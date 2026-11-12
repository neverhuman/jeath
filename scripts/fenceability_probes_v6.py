#!/usr/bin/env python3
"""E6 executable fenceability probes (plan: fixtures/v6/fenceability_plan.json).

Probes P1 idempotent retry, P2 epoch fence and P3 hold/cancel on a fresh local PostgreSQL 16 cluster
(target/e6/pg), a throwaway etcd v3.5.17 container (cc-e6-*) and the local filesystem (target/e6/fs),
and writes target/runs/e6-<UTC>/results.json. Local reference instances, not production services;
nothing here is external exactly-once execution. S3 and the Kubernetes API are copied from the plan as
documented_not_executed. Outcomes are labelled by judge() in scripts/validate_fenceability_v6.py.
"""
from __future__ import annotations

import argparse, base64, fcntl, hashlib, json, os, platform, random, re, secrets, shutil, socket, subprocess  # noqa: E401
import threading, time, urllib.request  # noqa: E401
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
import psycopg2.errors

import validate_fenceability_v6 as V

ROOT = Path(__file__).resolve().parents[1]
PLAN, E6 = ROOT / "fixtures/v6/fenceability_plan.json", ROOT / "target/e6"
PGBIN, IMAGE = Path("/usr/lib/postgresql/16/bin"), "quay.io/coreos/etcd:v3.5.17"
LOG, T0 = [], time.monotonic()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log(msg: str) -> None:
    LOG.append(f"{now()} {msg}")
    print(LOG[-1], flush=True)


def sh(*args, check=True) -> str:
    return subprocess.run([str(a) for a in args], check=check, capture_output=True, text=True).stdout.strip()


sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
rel = lambda: round(time.monotonic() - T0, 4)  # noqa: E731  client-side monotonic seconds since start


def p1_drive(prm, deliver, **extra) -> dict:
    """Deliver each op P1_deliveries_per_op times; duplicates are adjacent so they are in flight together."""
    queue = [(f"op-{i:04d}", d) for i in range(prm["P1_ops"]) for d in range(prm["P1_deliveries_per_op"])]

    def timed(q):
        t0 = rel()
        return [q[0], bool(deliver(*q)), t0, rel()]
    with ThreadPoolExecutor(prm["threads"]) as ex:
        dlog = list(ex.map(timed, queue))
    return {"ops": prm["P1_ops"], "deliveries": len(queue), "threads": prm["threads"],
            "delivery_columns": ["op", "created_effect", "t_start", "t_end"], "delivery_log": dlog, **extra}


def p2_race(prm, attempt, raise_fence) -> tuple[list, dict]:
    """Writers issue stale-epoch writes; a separate thread raises the fence once enough have committed."""
    attempts, fence, lock, done, committed = [], {}, threading.Lock(), threading.Event(), [0]

    def writer(w):
        for i in range(prm["P2_attempts_per_writer"]):
            t0, op = rel(), f"race-{w}-{i:03d}"
            try:
                r = attempt(op, prm["P2_writer_epoch"])
            except Exception as e:  # recorded, never hidden
                r = (op, f"error:{type(e).__name__}", None)
            with lock:
                attempts.append([*r, t0, rel()])
                committed[0] += r[1] == "committed"

    def fencer():
        while committed[0] < prm["P2_fence_after_commits"] and not done.is_set():
            time.sleep(0.002)
        fence["t0"] = rel()
        fence.update(raise_fence(prm["P2_raised_floor"]))
        fence["t1"] = rel()
    threads = [threading.Thread(target=writer, args=(w,)) for w in range(prm["P2_writers"])] + [threading.Thread(target=fencer)]
    for t in threads:
        t.start()
    for t in threads[:-1]:
        t.join()
    done.set()
    threads[-1].join()
    return sorted(attempts, key=lambda a: a[3]), fence


def p2_evidence(prm, race, order, effects, fresh, bare, **extra) -> dict:
    return {"writer_epoch": prm["P2_writer_epoch"], "floor_before": prm["P2_initial_floor"], "floor_after": prm["P2_raised_floor"],
            "attempt_columns": ["op", "status", "observed_floor", "t_start", "t_end"], "attempts": race[0], "fence": race[1],
            "order": order, "effect_columns": ["op", "epoch", "order_key"], "effects": effects, "fresh": fresh, "bare": bare, **extra}


# ---------------------------------------------------------------- PostgreSQL 16
PG_SCHEMA = """
CREATE TABLE p1_effects(op_id text PRIMARY KEY, delivery int);
CREATE TABLE fence(issuer int PRIMARY KEY, floor int NOT NULL);
CREATE TABLE fence_log(issuer int, new_floor int, snap pg_snapshot, xid xid8 DEFAULT pg_current_xact_id());
CREATE TABLE effects(op_id text PRIMARY KEY, issuer int, epoch int, observed_floor int, xid xid8 DEFAULT pg_current_xact_id());
CREATE TABLE effects_fair (LIKE effects INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES);
CREATE TABLE effects_naive (LIKE effects INCLUDING DEFAULTS INCLUDING CONSTRAINTS INCLUDING INDEXES);
CREATE TABLE held_effects(op_id text PRIMARY KEY);
CREATE FUNCTION fence_guard() RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE f int;
BEGIN
  IF TG_ARGV[0] = 'locked' THEN SELECT floor INTO f FROM fence WHERE issuer = NEW.issuer FOR SHARE;
  ELSE SELECT floor INTO f FROM fence WHERE issuer = NEW.issuer; END IF;
  NEW.observed_floor := f;
  IF f IS NULL OR NEW.epoch < f THEN RAISE EXCEPTION 'E6_STALE epoch % < floor %', NEW.epoch, f; END IF;
  RETURN NEW;
END $$;
CREATE TRIGGER guard BEFORE INSERT ON effects FOR EACH ROW EXECUTE FUNCTION fence_guard('locked');
CREATE TRIGGER guard BEFORE INSERT ON effects_fair FOR EACH ROW EXECUTE FUNCTION fence_guard('locked');
CREATE TRIGGER guard BEFORE INSERT ON effects_naive FOR EACH ROW EXECUTE FUNCTION fence_guard('naive');
INSERT INTO fence VALUES (1, 1), (2, 1), (3, 1);
"""
PG_FAIR = "LOCK TABLE fence IN EXCLUSIVE MODE"
PG_FENCE = ["SELECT floor FROM fence WHERE issuer=%(i)s FOR UPDATE", "UPDATE fence SET floor=GREATEST(floor, %(f)s) WHERE issuer=%(i)s",
            "INSERT INTO fence_log(issuer, new_floor, snap) VALUES (%(i)s, %(f)s, pg_current_snapshot()) RETURNING snap::text"]


class Postgres:
    def __init__(self, run: Path):
        self.data, self.sock, self.logf = E6 / "pg", E6 / "sock", run / "postgres.log"
        self.port, self.tl, self.conns = random.randint(20000, 59999), threading.local(), []

    def ctl(self, *a):
        return sh(PGBIN / "pg_ctl", a[0], "-D", self.data, "-w", *a[1:], check=a[0] == "start")

    def init(self):
        sh(PGBIN / "initdb", "-D", self.data, "-A", "trust", "-U", "e6", "--no-locale", "-E", "UTF8")
        with open(self.data / "postgresql.conf", "a") as f:
            f.write(f"port = {self.port}\nlisten_addresses = ''\nunix_socket_directories = '{self.sock}'\n"
                    "max_prepared_transactions = 8\nmax_connections = 40\nshared_buffers = 32MB\nfsync = on\n")
        self.ctl("start", "-l", self.logf)

    def conn(self, autocommit=True):
        c = psycopg2.connect(host=str(self.sock), port=self.port, user="e6", dbname="postgres", options="-c statement_timeout=60000")
        c.autocommit = autocommit
        return c

    def q(self, sql, args=None):
        c = self.conn()
        try:
            with c.cursor() as cur:
                cur.execute(sql, args)
                return cur.fetchall() if cur.description else None
        finally:
            c.close()

    def tconn(self):  # one connection per writer thread
        if not hasattr(self.tl, "c"):
            self.tl.c = self.conn()
            self.conns.append(self.tl.c)
        return self.tl.c

    def close_threads(self):
        for c in self.conns:
            c.close()
        self.conns, self.tl = [], threading.local()


def pg_attempt(pg, op, epoch, table="effects", issuer=1, c=None):
    try:
        with (c or pg.tconn()).cursor() as cur:
            cur.execute(f"INSERT INTO {table}(op_id, issuer, epoch) VALUES (%s, %s, %s) RETURNING observed_floor", (op, issuer, epoch))
            return op, "committed", cur.fetchone()[0]
    except psycopg2.errors.RaiseException as e:
        m = re.search(r"floor (\d+)", str(e))
        return op, "refused_sink", int(m.group(1)) if m else None


def pg_raise(pg, floor, issuer, fair=False) -> dict:
    c = pg.conn(autocommit=False)
    try:
        with c.cursor() as cur:
            for sql in ([PG_FAIR] if fair else []) + PG_FENCE[:1]:  # lock acquisition
                cur.execute(sql, {"i": issuer, "f": floor})
            t_locked = rel()
            for sql in PG_FENCE[1:]:
                cur.execute(sql, {"i": issuer, "f": floor})
            snap = cur.fetchone()[0]
        c.commit()
        return {"t_locked": t_locked, "snap": snap}
    finally:
        c.close()


def pg_p2(pg, prm, table, issuer, fair) -> dict:
    race = p2_race(prm, lambda op, e: pg_attempt(pg, op, e, table, issuer), lambda f: pg_raise(pg, f, issuer, fair))
    fresh = pg_attempt(pg, "fresh-1", prm["P2_raised_floor"], table, issuer)[1]
    bare = pg_attempt(pg, "bare-1", prm["P2_writer_epoch"], table, issuer)[1]  # plain INSERT, no client-side guard
    pg.close_threads()
    cross = pg.q(f"SELECT count(*) FROM {table} e JOIN fence_log f USING (issuer) WHERE e.op_id LIKE %s "
                 "AND e.epoch < f.new_floor AND NOT pg_visible_in_snapshot(e.xid, f.snap)", ("race-%",))[0][0]
    effects = [list(r) for r in pg.q(f"SELECT op_id, epoch, xid::text::bigint FROM {table} ORDER BY op_id")]
    return p2_evidence(prm, race, {"kind": "pg_snapshot", "snap": race[1]["snap"]}, effects, fresh, bare, table=table,
                       issuer=issuer, fence_statements=([PG_FAIR] if fair else []) + PG_FENCE, server_crosscheck_violations=cross)


def run_postgres(run: Path, prm: dict) -> dict:
    pg, P = Postgres(run), {}
    pg.init()
    try:
        versions = {"server_version": pg.q("SELECT version()")[0][0], "postgres_binary": sh(PGBIN / "postgres", "--version"),
                    "postgres_binary_sha256": sha(PGBIN / "postgres"), "psycopg2": psycopg2.__version__, "socket_port": pg.port,
                    "package": sh("dpkg-query", "-W", "-f", "${Version}", "postgresql-16", check=False),
                    "settings": dict(pg.q("SELECT name, setting FROM pg_settings WHERE name IN ('max_prepared_transactions', "
                                          "'fsync', 'default_transaction_isolation', 'synchronous_commit')"))}
        pg.q(PG_SCHEMA)
        p1_sql = "INSERT INTO p1_effects VALUES (%s, %s) ON CONFLICT (op_id) DO NOTHING"

        def deliver(op, k):
            with pg.tconn().cursor() as cur:
                cur.execute(p1_sql, (op, k))
                return cur.rowcount == 1
        P["P1"] = p1_drive(prm, deliver, statement=p1_sql)
        pg.close_threads()
        P["P1"]["effects"] = [r[0] for r in pg.q("SELECT op_id FROM p1_effects ORDER BY op_id")]

        P["P2"] = pg_p2(pg, prm, "effects", 1, fair=False) | {"schema": PG_SCHEMA}
        a = pg.conn(autocommit=False)  # negative control: unlocked guard, deterministic interleaving A-insert, B-fence, A-commit
        seen = pg_attempt(pg, "race-ctl-1", prm["P2_writer_epoch"], "effects_naive", 2, c=a)[2]
        ctl_fence = pg_raise(pg, prm["P2_raised_floor"], 2)
        a.commit()
        a.close()
        P["P2"]["negative_control"] = {"floor_after": prm["P2_raised_floor"], "observed_floor_in_txn": seen,
                                       "order": {"kind": "pg_snapshot", "snap": ctl_fence["snap"]},
                                       "effects": [list(r) for r in pg.q("SELECT op_id, epoch, xid::text::bigint FROM effects_naive")]}
        P["P2"]["fair_fence_variant"] = pg_p2(pg, prm, "effects_fair", 3, fair=True)  # amendment 1

        def state():
            return {"held": sorted(r[0] for r in pg.q("SELECT gid FROM pg_prepared_xacts")),
                    "effects": sorted(r[0] for r in pg.q("SELECT op_id FROM held_effects")),
                    "boot": pg.q("SELECT pg_postmaster_start_time()::text")[0][0]}
        for gid in prm["P3_holds"]:
            c = pg.conn(autocommit=False)
            c.tpc_begin(gid)
            c.cursor().execute("INSERT INTO held_effects VALUES (%s)", (gid,))
            c.tpc_prepare()
            c.close()
        pre, log0 = state(), len(pg.logf.read_text().splitlines())
        pg.ctl("stop", "-m", "immediate")
        cstate = re.search(r"Database cluster state:\s*(.*)", sh(PGBIN / "pg_controldata", "-D", pg.data)).group(1)
        pg.ctl("start", "-l", pg.logf)
        post = state()
        recovery = [ln for ln in pg.logf.read_text().splitlines()[log0:] if re.search(r"not properly shut down|redo|recover", ln)]
        pg.q("ROLLBACK PREPARED 'revoke'")
        pg.q("COMMIT PREPARED 'control'")
        P["P3"] = {"holds": prm["P3_holds"], "pre": pre, "post": post, "final": state(),
                   "crash": {"kind": "pg_ctl stop -m immediate", "controldata_state": cstate, "server_log": recovery},
                   "statements": ["tpc_begin(gid); INSERT INTO held_effects VALUES (gid); PREPARE TRANSACTION gid",
                                  "ROLLBACK PREPARED 'revoke'", "COMMIT PREPARED 'control'"]}
        return {"versions": versions, "primitives": P}
    finally:
        pg.close_threads()
        pg.ctl("stop", "-m", "fast")


# ---------------------------------------------------------------- etcd v3.5.17
b64 = lambda s: base64.b64encode(s.encode()).decode()  # noqa: E731  (etcd JSON gateway: keys/values are base64)
ub = lambda s: base64.b64decode(s).decode()  # noqa: E731
PUT = lambda k, v, **kw: {"request_put": {"key": b64(k), "value": b64(v), **kw}}  # noqa: E731
CMP = lambda k, target, result, **kw: {"key": b64(k), "target": target, "result": result, **kw}  # noqa: E731


class Etcd:
    def __init__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.name, self.url = "cc-e6-" + secrets.token_hex(3), f"http://127.0.0.1:{self.port}"

    def wait(self):
        for _ in range(240):
            try:
                if self.get("/health").get("health") == "true":
                    return
            except (OSError, ValueError):
                pass
            time.sleep(0.25)
        raise RuntimeError("etcd did not become healthy")

    def get(self, path):
        return json.loads(urllib.request.urlopen(self.url + path, timeout=10).read())

    def call(self, path, body):
        req = urllib.request.Request(self.url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
        return json.loads(urllib.request.urlopen(req, timeout=30).read())

    def txn(self, compare, success, failure=()):
        return self.call("/v3/kv/txn", {"compare": compare, "success": success, "failure": list(failure)})

    def range(self, prefix):
        end = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        return [(ub(kv["key"])[len(prefix):], kv) for kv in
                self.call("/v3/kv/range", {"key": b64(prefix), "range_end": b64(end)}).get("kvs", [])]

    def inspect(self, fmt):
        return sh("docker", "inspect", "-f", fmt, self.name)


def run_etcd(run: Path, prm: dict) -> dict:
    et, P, pad = Etcd(), {}, lambda n: f"{n:010d}"
    img = json.loads(sh("docker", "image", "inspect", IMAGE))[0]
    try:
        sh("docker", "run", "-d", "--name", et.name, "--cpus", str(prm["etcd_container_cpus"]), "-p", f"127.0.0.1:{et.port}:2379",
           IMAGE, "/usr/local/bin/etcd", "--name", "e6", "--data-dir", "/var/lib/e6", "--listen-client-urls", "http://0.0.0.0:2379",
           "--advertise-client-urls", et.url)
        et.wait()
        log(f"etcd container {et.name} on 127.0.0.1:{et.port}")
        versions = {"server": et.get("/version"), "image": IMAGE, "image_id": img["Id"], "repo_digests": img["RepoDigests"],
                    "container": et.name, "container_image_id": et.inspect("{{.Image}}"), "client_port": et.port,
                    "docker_server": sh("docker", "version", "-f", "{{.Server.Version}}")}

        def deliver(op, k):
            return et.txn([CMP("p1/" + op, "CREATE", "EQUAL", create_revision="0")], [PUT("p1/" + op, str(k))]).get("succeeded", False)
        P["P1"] = p1_drive(prm, deliver, statement="txn compare CREATE(p1/<op>) == 0, success put p1/<op>")
        kvs = et.range("p1/")
        P["P1"] |= {"effects": [k for k, _ in kvs], "versions": {k: int(kv["version"]) for k, kv in kvs}}

        et.call("/v3/kv/put", {"key": b64("fence/1"), "value": b64(pad(prm["P2_initial_floor"]))})
        rng = {"request_range": {"key": b64("fence/1")}}

        def body(op, epoch):  # floor <= epoch  <=>  value(fence/1) < pad(epoch + 1); zero padding makes byte order numeric
            return {"compare": [CMP("fence/1", "VALUE", "LESS", value=b64(pad(epoch + 1)))],
                    "success": [PUT("p2/" + op, str(epoch)), rng], "failure": [rng]}

        def attempt(op, epoch):
            r = et.call("/v3/kv/txn", body(op, epoch))
            return op, "committed" if r.get("succeeded") else "refused_sink", int(ub(r["responses"][-1]["response_range"]["kvs"][0]["value"]))

        def raise_fence(floor):
            r = et.txn([CMP("fence/1", "VALUE", "LESS", value=b64(pad(floor)))], [PUT("fence/1", pad(floor))])
            return {"succeeded": r.get("succeeded", False), "header_revision": int(r["header"]["revision"])}
        race = p2_race(prm, attempt, raise_fence)
        fresh = attempt("fresh-1", prm["P2_raised_floor"])[1]
        bare = "committed" if "header" in et.call("/v3/kv/put", {"key": b64("p2/bare-1"), "value": b64("1")}) else "error"
        fkv = et.range("fence/")[0][1]
        effects = [[k, int(ub(kv["value"])), int(kv["mod_revision"])] for k, kv in et.range("p2/")]
        P["P2"] = p2_evidence(prm, race, {"kind": "etcd_revision", "fence_revision": int(fkv["mod_revision"]), "fence_value": ub(fkv["value"])},
                              effects, fresh, bare, writer_request_example=body("race-0-000", prm["P2_writer_epoch"]))

        leases = {h: et.call("/v3/lease/grant", {"TTL": prm["etcd_lease_ttl_seconds"]})["ID"] for h in prm["P3_holds"]}
        for h in prm["P3_holds"]:
            et.call("/v3/kv/put", {"key": b64("staged/" + h), "value": b64(h), "lease": leases[h]})

        def state():
            return {"held": sorted(h for h, kv in et.range("staged/") if kv.get("lease") == leases.get(h)),
                    "effects": sorted(k for k, _ in et.range("p3eff/")), "boot": et.inspect("{{.State.StartedAt}}"),
                    "lease_ttl": {h: int(et.call("/v3/lease/timetolive", {"ID": leases[h]}).get("TTL", 0)) for h in leases}}
        pre = state()
        sh("docker", "kill", "-s", "KILL", et.name)
        crash = {"kind": "docker kill -s KILL", "exit_code": int(sh("docker", "wait", et.name))}
        sh("docker", "start", et.name)
        et.wait()
        post = state()
        revoke = et.call("/v3/lease/revoke", {"ID": leases["revoke"]})
        promote = {h: et.txn([CMP("staged/" + h, "LEASE", "EQUAL", lease=leases[h])],
                             [PUT("p3eff/" + h, h), {"request_delete_range": {"key": b64("staged/" + h)}}]).get("succeeded", False)
                   for h in prm["P3_holds"]}
        P["P3"] = {"holds": prm["P3_holds"], "leases": leases, "pre": pre, "post": post, "crash": crash,
                   "revoke_response_header": revoke.get("header"), "promote_succeeded": promote, "final": state(),
                   "statements": ["lease grant TTL; put staged/<h> with lease", "lease revoke (revocation)",
                                  "txn compare LEASE(staged/<h>) == lease, success put p3eff/<h>, delete staged/<h>"]}
        return {"versions": versions, "primitives": P}
    finally:
        sh("docker", "rm", "-f", et.name, check=False)


# ---------------------------------------------------------------- POSIX filesystem
def run_fs(run: Path, prm: dict) -> dict:
    root = E6 / "fs"
    eff, fence = root / "p2/effects", root / "p2/fence"
    for d in (root / "p1", eff):
        d.mkdir(parents=True)
    versions = {"findmnt": sh("findmnt", "-no", "FSTYPE,SOURCE", "--target", root), "kernel": platform.release(),
                "path": str(root.relative_to(ROOT))}

    def create(path, content):
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        os.write(fd, str(content).encode())
        os.close(fd)

    def deliver(op, k):
        try:
            create(root / "p1" / op, k)
            return True
        except FileExistsError:
            return False
    P = {"P1": p1_drive(prm, deliver, statement="os.open(p1/<op>, O_CREAT|O_EXCL|O_WRONLY)")}
    P["P1"]["effects"] = sorted(os.listdir(root / "p1"))
    fence.write_text(str(prm["P2_initial_floor"]))

    def attempt(op, epoch):  # cooperative writer: LOCK_SH, read floor, decide, create
        fd = os.open(fence, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_SH)
            floor = int(os.pread(fd, 32, 0))
            if epoch < floor:
                return op, "refused_self", floor
            create(eff / op, epoch)
            return op, "committed", floor
        finally:
            os.close(fd)  # releases the flock

    def raise_fence(floor):
        fd = os.open(fence, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            t_locked = rel()
            os.ftruncate(fd, 0)
            os.pwrite(fd, str(floor).encode(), 0)
            os.fsync(fd)
            return {"t_locked": t_locked, "listing_under_lock": sorted(os.listdir(eff))}
        finally:
            os.close(fd)
    race = p2_race(prm, attempt, raise_fence)
    fresh = attempt("fresh-1", prm["P2_raised_floor"])[1]
    fd, probe = os.open(fence, os.O_RDWR), os.open(fence, os.O_RDONLY)  # bare writer: test holds LOCK_EX at floor 2
    fcntl.flock(fd, fcntl.LOCK_EX)
    try:
        fcntl.flock(probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
        coop_blocked = False
    except BlockingIOError:
        coop_blocked = True
    create(eff / "bare-1", prm["P2_writer_epoch"])  # ignores both the lock and the floor
    bare = {"lock_ex_held_by_test": True, "floor_in_fence_file": int(fence.read_text()),
            "cooperative_writer_would_block": coop_blocked, "bare_file_created": (eff / "bare-1").exists()}
    os.close(probe)
    os.close(fd)
    effects = [[n, int((eff / n).read_text()), None] for n in sorted(os.listdir(eff))]
    order = {"kind": "listing_under_lock", "listing_under_lock": race[1].pop("listing_under_lock")}
    P["P2"] = p2_evidence(prm, race, order, effects, fresh, "committed" if bare["bare_file_created"] else "error", bare_detail=bare)
    P["P3"] = None
    return {"versions": versions, "primitives": P}


# ---------------------------------------------------------------- driver
def record(plan: dict, surface: str, prim: str, ev) -> dict:
    sp = plan["surfaces"][surface][prim]
    if ev is None:
        return {"outcome": "not_applicable", "mechanism": None, "checks": {}, "derived": {}, "evidence": {"reason": sp["reason"]}}
    checks, derived = V.judge(prim, ev)
    return {"outcome": "pass" if all(checks.values()) else "fail", "checks": checks, "derived": derived,
            "mechanism": sp.get("classification_if_pass") or sp.get("classification"), "evidence": ev}


def reset_e6() -> None:
    marker = E6 / ".e6-owned"
    if E6.exists() and not marker.exists():
        raise SystemExit(f"{E6} exists without {marker.name}; refusing to clear it")
    if (E6 / "pg/postmaster.pid").exists():
        sh(PGBIN / "pg_ctl", "stop", "-D", E6 / "pg", "-m", "fast", check=False)
    for d in ("pg", "sock", "fs"):
        shutil.rmtree(E6 / d, ignore_errors=True)
    (E6 / "sock").mkdir(parents=True)
    marker.write_text("scratch state of scripts/fenceability_probes_v6.py; evidence lives in target/runs/e6-*\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true", help="development run: name it e6-dev-<stamp>")
    dev = ap.parse_args().dev
    plan = json.loads(PLAN.read_text())
    prm = plan["parameters"]
    run = ROOT / "target/runs" / f"e6-{'dev-' if dev else ''}{sh('date', '-u', '+%Y%m%dT%H%MZ')}"
    run.mkdir(parents=True)
    reset_e6()
    res = {"schema_version": 1, "experiment": plan["experiment"], "plan_sha256": sha(PLAN), "script_sha256": sha(__file__),
           "validator_sha256": sha(V.__file__), "started_utc": now(), "claim_boundary": plan["claim_boundary"],
           "host": {"hostname": socket.gethostname(), "machine": platform.machine(), "kernel": platform.release(),
                    "cpus": os.cpu_count(), "loadavg_at_start": os.getloadavg()}, "surfaces": {}}
    for name, fn in (("postgresql16", run_postgres), ("etcd_v3_5_17", run_etcd), ("posix_fs", run_fs)):
        log(f"{name}: start")
        s = fn(run, prm)
        prims = {p: record(plan, name, p, ev) for p, ev in s.pop("primitives").items()}
        res["surfaces"][name] = {"executed": True, **s, "primitives": prims}
        log(f"{name}: " + ", ".join(f"{p}={r['outcome']}" for p, r in prims.items()))
    for name in V.DOCUMENTED:
        sp = plan["surfaces"][name]
        res["surfaces"][name] = {"executed": False, "reason": sp["reason"], "documentation": sp.get("sources", sp.get("note")),
                                 "primitives": {p: {"outcome": "documented_not_executed", "mechanism": None, "checks": {}, "derived": {},
                                                    "evidence": {k: v for k, v in sp[p].items() if k != "outcome"}} for p in ("P1", "P2", "P3")}}
    res["finished_utc"] = now()
    (run / "results.json").write_text(json.dumps(res, indent=1) + "\n")
    (run / "probe.log").write_text("\n".join(LOG) + "\n")
    (run / "SHA256SUMS").write_text("".join(f"{sha(run / n)}  {n}\n" for n in ("results.json", "probe.log", "postgres.log")))
    print(run)


if __name__ == "__main__":
    main()

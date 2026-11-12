#!/usr/bin/env python3
"""E6b deterministic fence census (plan: fixtures/v6/fenceability_b_plan.json; validator: validate_fenceability_v6b.py).

For each mechanism and each (condition, repeat): a deterministic probe (after the fence ack, the stale writer's
cond / omit_epoch / omit_all / forge writes and the fresh write, one at a time) and E6's race, imported unchanged
from fenceability_probes_v6.p2_race. Runs only on xbabe0. PostgreSQL 16.15 from unpacked Ubuntu .debs under
target/pg16, etcd v3.5.17 in throwaway e6b-etcd-* containers, the local filesystem under target/e6b/fs. Local
reference instances, not production services; nothing here is external exactly-once execution or sandbox
verification. Classes and outcomes are computed by analyze() in the validator. Credentials are generated in
memory and never written or logged.
"""
from __future__ import annotations

import argparse, errno, fcntl, hashlib, json, os, platform, re, secrets, shutil, socket, subprocess, sys  # noqa: E401
import threading, time, traceback, urllib.error, urllib.request  # noqa: E401
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PGROOT = ROOT / "target/pg16"
os.environ["LD_LIBRARY_PATH"] = str(PGROOT / "usr/lib/x86_64-linux-gnu")  # for initdb/pg_ctl (libpq5 16.15)

import psycopg2  # noqa: E402

import fenceability_probes_v6 as E6  # noqa: E402 (E6 code, unchanged; hash recorded)
import validate_fenceability_v6b as V  # noqa: E402

E6.PGBIN = PGROOT / "usr/lib/postgresql/16/bin"
SCRATCH = ROOT / "target/e6b"
IMAGE = E6.IMAGE
LOG = []
rel = E6.rel  # the race's clock (client monotonic seconds since E6.T0, 4 decimals)
rel6 = lambda: round(time.monotonic() - E6.T0, 6)  # noqa: E731  same clock, 1 us, for the deterministic probe
b64, ub, PUT, CMP = E6.b64, E6.ub, E6.PUT, E6.CMP
pad = lambda n: f"{n:010d}"  # noqa: E731  (E6's etcd floor encoding)
BURN = "import os,sys\np=int(sys.argv[1])\nwhile os.getppid()==p:\n    pass\n"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def log(msg: str) -> None:
    LOG.append(f"{now()} {msg}")
    print(LOG[-1], flush=True)


def sh(*a, check=True) -> str:
    return subprocess.run([str(x) for x in a], check=check, capture_output=True, text=True).stdout.strip()


def e6_extract(pattern: str) -> str:
    m = re.search(pattern, E6.PG_SCHEMA, re.S)
    assert m, pattern
    return m.group(0)


FENCE_GUARD = e6_extract(r"CREATE FUNCTION fence_guard\(\).*?END \$\$;")  # E6's trigger function, byte for byte
PG_BASE = "\n".join([e6_extract(r"CREATE TABLE fence\(.*?\);"), e6_extract(r"CREATE TABLE fence_log\(.*?\);"),
                     "CREATE TABLE live_epochs(issuer int, epoch int, PRIMARY KEY (issuer, epoch));", FENCE_GUARD])


# ---------------------------------------------------------------- PostgreSQL 16.15
class PgCluster:
    def __init__(self, run: Path, cond: str, r: int):
        self.pg = E6.Postgres(run)
        self.pg.data, self.pg.sock, self.pg.logf = SCRATCH / "pg", SCRATCH / "sock", run / f"postgres-{cond}-{r:02d}.log"
        shutil.rmtree(SCRATCH / "pg", ignore_errors=True)
        shutil.rmtree(SCRATCH / "sock", ignore_errors=True)
        (SCRATCH / "sock").mkdir(parents=True)
        self.pg.init()  # E6's initdb flags and postgresql.conf additions
        self.issuer = 0
        self.q(PG_BASE)
        self.version = self.q("SELECT version()")[0][0]

    def conn(self, user="e6"):
        c = psycopg2.connect(host=str(self.pg.sock), port=self.pg.port, user=user, dbname="postgres",
                             options="-c statement_timeout=60000")
        c.autocommit = True
        return c

    def q(self, sql, args=None):
        c = self.conn()
        try:
            with c.cursor() as cur:
                cur.execute(sql, args)
                return cur.fetchall() if cur.description else None
        finally:
            c.close()

    def stop(self):
        self.pg.ctl("stop", "-m", "fast")
        shutil.rmtree(self.pg.data, ignore_errors=True)


class PgMech:
    """One namespace (table, issuer id, roles) of one PostgreSQL mechanism."""

    def __init__(self, cl: PgCluster, mech: str, ns: str):
        cl.issuer += 1
        self.cl, self.mech, self.I, self.t = cl, mech, cl.issuer, f"t_{mech}_{ns}"
        self.kind = {"pg_trigger_e6": "trigger", "pg_trigger_e6_fair": "trigger", "pg_trigger_notnull": "trigger",
                     "pg_foreign_key": "fk", "pg_revoke": "revoke"}[mech]
        nn = " NOT NULL" if mech == "pg_trigger_notnull" else ""
        t, I = self.t, self.I
        if self.kind == "trigger":
            ddl = [f"CREATE TABLE {t}(op_id text PRIMARY KEY, issuer int{nn}, epoch int{nn}, observed_floor int, "
                   "xid xid8 DEFAULT pg_current_xact_id())",
                   f"CREATE TRIGGER guard BEFORE INSERT ON {t} FOR EACH ROW EXECUTE FUNCTION fence_guard('locked')",
                   f"INSERT INTO fence VALUES ({I}, 1)"]
        elif self.kind == "fk":
            ddl = [f"CREATE TABLE {t}(op_id text PRIMARY KEY, issuer int NOT NULL, epoch int NOT NULL, "
                   "xid xid8 DEFAULT pg_current_xact_id(), FOREIGN KEY (issuer, epoch) REFERENCES live_epochs (issuer, epoch))",
                   f"INSERT INTO live_epochs VALUES ({I}, 1)"]
        else:
            ddl = [f"CREATE TABLE {t}(op_id text PRIMARY KEY, xid xid8 DEFAULT pg_current_xact_id())",
                   f"CREATE ROLE {t}_w1 LOGIN NOSUPERUSER", f"CREATE ROLE {t}_w2 LOGIN NOSUPERUSER",
                   f"GRANT INSERT ON {t} TO {t}_w1, {t}_w2"]
        self.ddl = ddl
        for s in ddl:
            cl.q(s)
        self.users = (f"{t}_w1", f"{t}_w2") if self.kind == "revoke" else ("e6", "e6")
        self.tl, self.tconns, self.lock = threading.local(), [], threading.Lock()

    def sql(self, variant):
        t = self.t
        if self.kind == "revoke":
            return f"INSERT INTO {t}(op_id) VALUES (%(op)s)", False
        ret = " RETURNING observed_floor" if self.kind == "trigger" else ""
        return {"omit_epoch": (f"INSERT INTO {t}(op_id, issuer) VALUES (%(op)s, %(i)s)", False),
                "omit_all": (f"INSERT INTO {t}(op_id) VALUES (%(op)s)", False)}.get(
            variant, (f"INSERT INTO {t}(op_id, issuer, epoch) VALUES (%(op)s, %(i)s, %(e)s){ret}", bool(ret)))

    def handles(self):
        return self.cl.conn(self.users[0]), self.cl.conn(self.users[1])

    def write(self, c, variant, op, epoch):
        sql, returning = self.sql(variant)
        try:
            with c.cursor() as cur:
                cur.execute(sql, {"op": op, "i": self.I, "e": epoch})
                f = cur.fetchone()[0] if returning else None
            return op, "committed", compact({"kind": "pg", "ok": True, "observed_floor": f})
        except Exception as e:  # recorded, never hidden
            raw = compact({"kind": "pg", "ok": False, "sqlstate": getattr(e, "pgcode", None), "exc": type(e).__name__})
            return op, V.status(raw, PLAN), raw

    def race_write(self, op, epoch):
        if not hasattr(self.tl, "c"):
            self.tl.c = self.cl.conn(self.users[0])
            with self.lock:
                self.tconns.append(self.tl.c)
        return self.write(self.tl.c, "cond", op, epoch)

    def close_race(self):
        for c in self.tconns:
            c.close()

    def raise_fence(self, floor):
        p = {"i": self.I, "f": floor}
        if self.kind == "trigger":
            lock = ([E6.PG_FAIR] if self.mech == "pg_trigger_e6_fair" else []) + E6.PG_FENCE[:1]
            rest = E6.PG_FENCE[1:]
        elif self.kind == "fk":
            lock = ["INSERT INTO live_epochs VALUES (%(i)s, %(f)s) ON CONFLICT DO NOTHING",
                    "DELETE FROM live_epochs WHERE issuer=%(i)s AND epoch < %(f)s"]
            rest = E6.PG_FENCE[2:]
        else:
            lock, rest = [f"REVOKE INSERT ON {self.t} FROM {self.t}_w1"], E6.PG_FENCE[2:]
        c = self.cl.conn()
        c.autocommit = False
        try:
            with c.cursor() as cur:
                for s in lock:
                    cur.execute(s, p)
                t_locked = rel()
                for s in rest:
                    cur.execute(s, p)
                snap = cur.fetchone()[0]
            c.commit()
            return {"t_locked": t_locked, "snap": snap, "statements": lock + rest}
        finally:
            c.close()

    def readback(self):
        return [r[0] for r in self.cl.q(f"SELECT op_id FROM {self.t} ORDER BY op_id")]

    def race_evidence(self, fence):
        cols = "op_id, epoch, xid::text::bigint" if self.kind != "revoke" else "op_id, NULL::int, xid::text::bigint"
        cross = self.cl.q(f"SELECT count(*) FROM {self.t} e JOIN fence_log f ON f.issuer = %s WHERE e.op_id LIKE 'race-%%' "
                          "AND NOT pg_visible_in_snapshot(e.xid, f.snap)", (self.I,))[0][0]
        return {"order": {"kind": "pg_snapshot", "snap": fence.pop("snap", None)},
                "effects": [list(r) for r in self.cl.q(f"SELECT {cols} FROM {self.t} ORDER BY op_id")],
                "server_crosscheck_violations": cross}

    def describe(self):
        return {"table": self.t, "issuer": self.I, "ddl": self.ddl, "writer_users": list(self.users)}


# ---------------------------------------------------------------- etcd v3.5.17
class EtcdBox:
    """One throwaway container (E6's command line, e6b- name)."""

    def __init__(self):
        self.et = E6.Etcd()
        self.et.name = "e6b-etcd-" + secrets.token_hex(3)
        img = json.loads(sh("docker", "image", "inspect", IMAGE))[0]
        sh("docker", "run", "-d", "--name", self.et.name, "--cpus", "2", "-p", f"127.0.0.1:{self.et.port}:2379", IMAGE,
           "/usr/local/bin/etcd", "--name", "e6", "--data-dir", "/var/lib/e6", "--listen-client-urls", "http://0.0.0.0:2379",
           "--advertise-client-urls", self.et.url)
        self.et.wait()
        v = self.et.get("/version")
        self.versions = {"etcdserver": v.get("etcdserver"), "etcdcluster": v.get("etcdcluster"), "image": IMAGE,
                         "image_id": img["Id"], "repo_digests": img["RepoDigests"], "container": self.et.name,
                         "container_image_id": self.et.inspect("{{.Image}}"), "client_port": self.et.port}
        self.root_token = None

    def req(self, path, body, token=None):
        h = {"Content-Type": "application/json"}
        if token:
            h["Authorization"] = token
        try:
            with urllib.request.urlopen(urllib.request.Request(self.et.url + path, json.dumps(body).encode(), h), timeout=30) as r:
                return r.status, json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read() or b"{}")
            except ValueError:
                return e.code, {}

    def admin(self, path, body):
        code, r = self.req(path, body, self.root_token)
        if code != 200:
            raise RuntimeError(f"etcd admin {path}: HTTP {code} {r.get('message')}")
        return r

    def enable_auth(self):
        pw = secrets.token_urlsafe(18)
        for path, body in (("/v3/auth/role/add", {"name": "root"}), ("/v3/auth/user/add", {"name": "root", "password": pw}),
                           ("/v3/auth/user/grant", {"user": "root", "role": "root"}), ("/v3/auth/enable", {})):
            self.admin(path, body)
        self.root_token = self.token("root", pw)

    def token(self, name, pw):
        code, r = self.req("/v3/auth/authenticate", {"name": name, "password": pw})
        if code != 200 or "token" not in r:
            raise RuntimeError(f"etcd authenticate {name}: HTTP {code}")
        return r["token"]

    def range(self, prefix):
        end = prefix[:-1] + chr(ord(prefix[-1]) + 1)
        code, r = self.req("/v3/kv/range", {"key": b64(prefix), "range_end": b64(end)}, self.root_token)
        if code != 200:
            raise RuntimeError(f"etcd range: HTTP {code}")
        return [(ub(kv["key"])[len(prefix):], kv) for kv in r.get("kvs", [])]

    def close(self):
        sh("docker", "rm", "-f", self.et.name, check=False)


def compact(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


def etcd_raw(code, r, succeeded=None, floor=None):
    if code == 200:
        return compact({"kind": "etcd", "http": 200, "succeeded": succeeded, "observed_floor": floor})
    return compact({"kind": "etcd", "http": code, "grpc_code": r.get("code"), "message": r.get("message") or r.get("error")})


class EtcdMech:
    def __init__(self, box: EtcdBox, mech: str, ns: str):
        self.box, self.mech, self.ns = box, mech, ns
        self.prefix, self.fkey = f"p2/{ns}/", f"fence/{ns}"
        if mech == "etcd_txn_compare":
            self.box.admin("/v3/kv/put", {"key": b64(self.fkey), "value": b64(pad(1))})
            self.desc = {"fence_key": self.fkey, "prefix": self.prefix}
        else:
            end = self.prefix[:-1] + chr(ord(self.prefix[-1]) + 1)
            self.perm = {"key": b64(self.prefix), "range_end": b64(end)}
            self.roles, pws = (f"r1-{ns}", f"r2-{ns}"), {}
            for i, role in enumerate(self.roles, 1):
                user = f"w{i}-{ns}"
                pws[user] = secrets.token_urlsafe(18)
                self.box.admin("/v3/auth/role/add", {"name": role})
                self.box.admin("/v3/auth/role/grant", {"name": role, "perm": {"permType": "READWRITE", **self.perm}})
                self.box.admin("/v3/auth/user/add", {"name": user, "password": pws[user]})
                self.box.admin("/v3/auth/user/grant", {"user": user, "role": role})
            self.users = list(pws)
            self.toks = [self.box.token(u, pws[u]) for u in self.users]  # obtained before the fence; memory only
            self.desc = {"prefix": self.prefix, "users": self.users, "roles": list(self.roles), "permission": "READWRITE on prefix"}

    def handles(self):
        return (None, None) if self.mech == "etcd_txn_compare" else tuple(self.toks)

    def write(self, h, variant, op, epoch):
        key = b64(self.prefix + op)
        if self.mech == "etcd_rbac_revoke" or variant in ("omit_epoch", "omit_all"):
            code, r = self.box.req("/v3/kv/put", {"key": key, "value": b64(str(epoch))}, h)
            raw = etcd_raw(code, r)
        else:  # E6's writer txn
            rng = {"request_range": {"key": b64(self.fkey)}}
            body = {"compare": [CMP(self.fkey, "VALUE", "LESS", value=b64(pad(epoch + 1)))],
                    "success": [PUT(self.prefix + op, str(epoch)), rng], "failure": [rng]}
            code, r = self.box.req("/v3/kv/txn", body)
            floor = int(ub(r["responses"][-1]["response_range"]["kvs"][0]["value"])) if code == 200 else None
            raw = etcd_raw(code, r, bool(r.get("succeeded", False)), floor)
        return op, V.status(raw, PLAN), raw

    def race_write(self, op, epoch):
        return self.write(self.handles()[0], "cond", op, epoch)

    def close_race(self):
        pass

    def raise_fence(self, floor):
        if self.mech == "etcd_txn_compare":  # E6's raise
            code, r = self.box.req("/v3/kv/txn", {"compare": [CMP(self.fkey, "VALUE", "LESS", value=b64(pad(floor)))],
                                                   "success": [PUT(self.fkey, pad(floor))], "failure": []})
            return {"http": code, "succeeded": bool(r.get("succeeded", False)), "header_revision": int(r["header"]["revision"])}
        r = self.box.admin("/v3/auth/role/revoke", {"role": self.roles[0], **self.perm})
        return {"http": 200, "header_revision": int(r["header"]["revision"])}

    def readback(self):
        return [k for k, _ in self.box.range(self.prefix)]

    def race_evidence(self, fence):
        if self.mech == "etcd_txn_compare":
            fkv = self.box.range("fence/")
            fkv = dict(fkv)[self.ns]
            order = {"kind": "etcd_revision", "fence_revision": int(fkv["mod_revision"]), "fence_value": ub(fkv["value"])}
        else:
            order = {"kind": "etcd_revision", "fence_revision": fence.get("header_revision")}
        return {"order": order, "effects": [[k, int(ub(kv["value"])), int(kv["mod_revision"])] for k, kv in self.box.range(self.prefix)]}

    def describe(self):
        return self.desc


# ---------------------------------------------------------------- POSIX filesystem
def fs_create(path, content, dir_fd=None):
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644, dir_fd=dir_fd)
    try:
        os.write(fd, str(content).encode())
    finally:
        os.close(fd)


def fs_raw_err(e: OSError):
    return {"kind": "fs", "ok": False, "errno": errno.errorcode.get(e.errno, str(e.errno))}


class FsMech:
    def __init__(self, base: Path, mech: str, ns: str):
        self.mech, self.d = mech, base / ns
        self.eff = self.d / "effects"
        self.eff.mkdir(parents=True)
        if mech == "fs_flock":
            self.fence = self.d / "fence"
            self.fence.write_text("1")
        else:
            for e in ("e1", "e2"):
                (self.eff / e).mkdir(mode=0o755)
            os.chmod(self.eff, 0o555)
        self.shared = None

    def handles(self):
        if self.mech == "fs_flock":
            return None, None
        return (os.open(self.eff / "e1", os.O_RDONLY | os.O_DIRECTORY), os.open(self.eff / "e2", os.O_RDONLY | os.O_DIRECTORY))

    def write(self, h, variant, op, epoch):
        ok = {"kind": "fs", "ok": True}
        try:
            if variant in ("omit_epoch", "omit_all"):
                fs_create(self.eff / op, epoch)
            elif self.mech == "fs_flock":  # E6's cooperative writer
                fd = os.open(self.fence, os.O_RDONLY)
                try:
                    fcntl.flock(fd, fcntl.LOCK_SH)
                    floor = int(os.pread(fd, 32, 0))
                    if epoch < floor:
                        return op, "refused_self", {"kind": "fs", "ok": False, "self_refused_floor": floor}
                    fs_create(self.eff / op, epoch)
                    ok["observed_floor"] = floor
                finally:
                    os.close(fd)
            elif variant == "forge":
                fs_create(self.eff / "e2" / op, epoch)
            else:
                fs_create(op, epoch, dir_fd=h)
            return op, "committed", ok
        except OSError as e:
            raw = fs_raw_err(e)
            return op, V.status(raw, PLAN), raw

    def race_write(self, op, epoch):
        return self.write(self.shared, "cond", op, epoch)

    def close_race(self):
        if self.shared is not None:
            os.close(self.shared)

    def raise_fence(self, floor):
        if self.mech == "fs_flock":  # E6's raise
            fd = os.open(self.fence, os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                t_locked = rel()
                os.ftruncate(fd, 0)
                os.pwrite(fd, str(floor).encode(), 0)
                os.fsync(fd)
                return {"t_locked": t_locked, "listing": sorted(os.listdir(self.eff))}
            finally:
                os.close(fd)
        os.chmod(self.eff / "e1", 0o555)
        return {"listing": sorted(os.listdir(self.eff / "e1"))}

    def readback(self):
        if self.mech == "fs_flock":
            return sorted(os.listdir(self.eff))
        top = [n for n in os.listdir(self.eff) if n not in ("e1", "e2")]
        return sorted(top + os.listdir(self.eff / "e1") + os.listdir(self.eff / "e2"))

    def race_evidence(self, fence):
        kind = "listing_under_lock" if self.mech == "fs_flock" else "listing_after_ack"
        src = self.eff if self.mech == "fs_flock" else self.eff / "e1"
        return {"order": {"kind": kind, "listing": fence.pop("listing", None)},
                "effects": [[n, int((src / n).read_text()), None] for n in sorted(os.listdir(src))]}

    def describe(self):
        return {"dir": str(self.d.relative_to(ROOT))}

    def cleanup(self):
        if self.mech == "fs_dir_mode":
            for p in (self.eff, self.eff / "e1"):
                os.chmod(p, 0o755)


# ---------------------------------------------------------------- probes
def fence_call(m, floor) -> dict:
    """Plan amendment 1: a fence raise that fails is recorded as data, not raised."""
    try:
        return m.raise_fence(floor)
    except Exception as e:
        msg = str(e).strip().splitlines()
        return {"error": compact({"exc": type(e).__name__, "sqlstate": getattr(e, "pgcode", None), "message": msg[0][:300] if msg else None})}


ATT_DET = ["variant", "op", "status", "raw", "t_start", "t_end"]
ATT_RACE = ["op", "status", "raw", "t_start", "t_end"]


def det_probe(m) -> dict:
    stale, fresh = m.handles()
    rows = []

    def go(variant, h, epoch):
        t0 = rel6()
        op, st, raw = m.write(h, variant, f"det-{variant}", epoch)
        rows.append([variant, op, st, raw, t0, rel6()])
    try:
        go("pre", stale, 1)
        fence = {"t0": rel6()}
        fence.update(fence_call(m, 2))
        fence["t1"] = rel6()
        for v, h, e in (("cond", stale, 1), ("omit_epoch", stale, 1), ("omit_all", stale, 1), ("forge", stale, 2), ("fresh", fresh, 2)):
            go(v, h, e)
    finally:
        for h in (stale, fresh):
            if isinstance(h, int):
                os.close(h)
            elif h is not None and hasattr(h, "close"):
                h.close()
    fence.pop("listing", None)
    snap = fence.pop("snap", None)
    return {"attempt_columns": ATT_DET, "attempts": rows, "fence": fence, "fence_snapshot": snap, "readback": m.readback(),
            "instance": m.describe()}


def race_probe(m, prm) -> dict:
    if isinstance(m, FsMech) and m.mech == "fs_dir_mode":
        m.shared = os.open(m.eff / "e1", os.O_RDONLY | os.O_DIRECTORY)  # stale writers' directory fd, opened before the fence
    try:
        attempts, fence = E6.p2_race(prm, m.race_write, lambda floor: fence_call(m, floor))
    finally:
        m.close_race()
    ev = m.race_evidence(fence)
    return {"attempt_columns": ATT_RACE, "attempts": attempts, "fence": fence, **ev, "instance": m.describe()}


def e6_prm(plan) -> dict:
    p = plan["contention_probe"]["parameters"]
    return {"P2_writers": p["writers"], "P2_attempts_per_writer": p["attempts_per_writer"], "P2_writer_epoch": p["writer_epoch"],
            "P2_fence_after_commits": p["fence_after_commits"], "P2_raised_floor": p["raised_epoch"]}


# ---------------------------------------------------------------- load blocks
def cpu_times():
    v = [int(x) for x in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:]]
    return sum(v), v[3] + v[4]


class Block:
    def __init__(self, cond, r, n):
        self.rec = {"condition": cond, "repeat": r, "burners": n, "utc_start": now(), "loadavg_start": list(os.getloadavg())}
        self.procs = [subprocess.Popen([sys.executable, "-c", BURN, str(os.getpid())]) for _ in range(n)]
        if n:
            time.sleep(2.5)
        self.cpu0, self.t0 = cpu_times(), rel()

    def end(self):
        (tot1, idle1), (tot0, idle0) = cpu_times(), self.cpu0
        alive = sum(p.poll() is None for p in self.procs)
        for p in self.procs:
            p.terminate()
        for p in self.procs:
            p.wait()
        self.rec.update(burners_alive_at_end=alive, loadavg_end=list(os.getloadavg()), utc_end=now(), t_start=self.t0, t_end=rel(),
                        cpu_busy_fraction=round(1 - (idle1 - idle0) / max(tot1 - tot0, 1), 4))
        return self.rec


# ---------------------------------------------------------------- driver
PLAN = V.load_plan()


def run_mech(surface, mech, ctx, prm):
    if surface == "postgresql16":
        dm, rm = PgMech(ctx, mech, "d"), PgMech(ctx, mech, "r")
    elif surface == "etcd_v3_5_17":
        dm, rm = EtcdMech(ctx, mech, "d"), EtcdMech(ctx, mech, "r")
    else:
        dm, rm = FsMech(ctx, mech, "d"), FsMech(ctx, mech, "r")
    try:
        return det_probe(dm), race_probe(rm, prm)
    finally:
        for m in (dm, rm):
            if isinstance(m, FsMech):
                m.cleanup()


def reset_scratch():
    marker = SCRATCH / ".e6b-owned"
    if SCRATCH.exists() and not marker.exists():
        raise SystemExit(f"{SCRATCH} exists without {marker.name}; refusing to clear it")
    if (SCRATCH / "pg/postmaster.pid").exists():
        sh(E6.PGBIN / "pg_ctl", "stop", "-D", SCRATCH / "pg", "-m", "fast", check=False)
    if SCRATCH.exists():
        for p, ds, fs in os.walk(SCRATCH):
            os.chmod(p, 0o755)
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    marker.write_text("scratch state of scripts/fenceability_probes_v6b.py; evidence lives in target/runs/e6b-*\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-commit", required=True, help="commit that holds the plan bytes this run uses")
    ap.add_argument("--smoke", action="store_true", help="repeat 1 only, both conditions; run dir e6b-smoke-<stamp>")
    a = ap.parse_args()
    uname_sm = sh("uname", "-sm")
    if socket.gethostname() != PLAN["host"]["hostname"] or uname_sm != PLAN["host"]["uname_sm"]:
        raise SystemExit(f"host {socket.gethostname()} / {uname_sm} is not {PLAN['host']['hostname']} / {PLAN['host']['uname_sm']}")
    if os.geteuid() == 0:
        raise SystemExit("refusing to run as root (root bypasses mode bits)")
    repeats = 1 if a.smoke else PLAN["repeats"]
    run = ROOT / "target/runs" / f"e6b-{'smoke-' if a.smoke else ''}{sh('date', '-u', '+%Y%m%dT%H%MZ')}"
    run.mkdir(parents=True)
    reset_scratch()
    pgbin = E6.PGBIN / "postgres"
    res = {"schema_version": 1, "experiment": PLAN["experiment"], "smoke": a.smoke, "repeats": repeats, "plan_commit": a.plan_commit,
           "plan_sha256": V.sha(V.PLAN), "script_sha256": V.sha(Path(__file__)), "validator_sha256": V.sha(Path(V.__file__)),
           "e6_probe_sha256": V.sha(Path(E6.__file__)), "e6_trigger_function_sha256": hashlib.sha256(FENCE_GUARD.encode()).hexdigest(),
           "claim_boundary": PLAN["claim_boundary"],
           "started_utc": now(),
           "host": {"hostname": socket.gethostname(), "uname_sm": uname_sm, "uname_a": sh("uname", "-a"), "nproc": os.cpu_count(),
                    "euid": os.geteuid(), "loadavg_at_start": list(os.getloadavg())},
           "software": {"postgres_binary": sh(pgbin, "--version"), "postgres_binary_sha256": V.sha(pgbin),
                        "postgresql_packages": {p.name: V.sha(p) for p in sorted((ROOT / "target/pkgs").glob("*.deb"))},
                        "psycopg2": psycopg2.__version__, "libpq_version": psycopg2.extensions.libpq_version(),
                        "python": platform.python_version(), "docker_server": sh("docker", "version", "-f", "{{.Server.Version}}"),
                        "filesystem": sh("findmnt", "-no", "FSTYPE,SOURCE", "--target", SCRATCH), "kernel": platform.release()},
           "blocks": [], "runs": []}
    prm, order, seq = e6_prm(PLAN), V.expected_order(PLAN, repeats), 0
    log(f"E6b run {run.name}: {len(order)} mechanism runs; plan {res['plan_sha256'][:12]} at {a.plan_commit[:12]}")
    try:
        for r in range(1, repeats + 1):
            for cond in (("idle", "loaded") if r % 2 else ("loaded", "idle")):
                blk = Block(cond, r, 96 if cond == "loaded" else 0)
                try:
                    for surface in V.surfaces(PLAN):
                        mechs = [m for c, rr, s, m in order if (c, rr, s) == (cond, r, surface)]
                        cl = PgCluster(run, cond, r) if surface == "postgresql16" else None
                        try:
                            for i, mech in enumerate(mechs):
                                if surface == "postgresql16":
                                    ctx, versions = cl, {"server_version": cl.version, "socket_port": cl.pg.port}
                                elif surface == "etcd_v3_5_17":
                                    ctx = EtcdBox()
                                    versions = ctx.versions
                                    if mech == "etcd_rbac_revoke":
                                        ctx.enable_auth()
                                else:
                                    ctx = SCRATCH / "fs" / f"{cond}-{r:02d}-{mech}"
                                    versions = {}
                                try:
                                    det, race = run_mech(surface, mech, ctx, prm)
                                finally:
                                    if isinstance(ctx, EtcdBox):
                                        ctx.close()
                                    elif isinstance(ctx, Path):
                                        shutil.rmtree(ctx)
                                res["runs"].append({"seq": seq, "condition": cond, "repeat": r, "surface": surface, "mechanism": mech,
                                                    "position_in_surface": i, "versions": versions, "det": det, "race": race})
                                seq += 1
                                dd, rd = V.det_derive(det, PLAN), V.race_derive(race, PLAN)
                                log(f"{cond} r{r} {mech}: class {dd['class']} ({dd['outcome']}); race fence wait {rd['fence_wait_s']} s, "
                                    f"after-ack {rd['attempts_started_after_fence_ack']}, refused {rd['refused_by_surface']}, "
                                    f"violations {len(rd['violations'])}")
                        finally:
                            if cl is not None:
                                cl.stop()
                finally:
                    res["blocks"].append(blk.end())
    except BaseException:
        res["aborted"] = traceback.format_exc()
        log("ABORTED: " + res["aborted"].strip().splitlines()[-1])
        raise
    finally:
        left = sh("docker", "ps", "-aq", "--filter", "name=^e6b-", check=False).split()
        if left:
            sh("docker", "rm", "-f", *left, check=False)
        res["leftover_e6b_containers_removed"] = len(left)
        res["finished_utc"] = now()
        if "aborted" not in res:
            res["summary"] = V.analyze(PLAN, res)
        (run / "results.json").write_text(json.dumps(res, separators=(",", ":")) + "\n")
        (run / "probe.log").write_text("\n".join(LOG) + "\n")
        names = sorted(p.name for p in run.iterdir() if p.is_file() and p.name != "SHA256SUMS")
        (run / "SHA256SUMS").write_text("".join(f"{V.sha(run / n)}  {n}\n" for n in names))
        print(run)


if __name__ == "__main__":
    main()

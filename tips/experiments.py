#!/usr/bin/env python3
"""
Experiments for "Intelligence Is Not Authority".

Nothing in this file encodes a real pathway to harm. Graph vertices are integers, costs are random
numbers, "steps" are random points, and the transaction model moves abstract units.

Run:   python3 experiments.py            (about ten minutes on a laptop; results are identical on every run)
       python3 experiments.py boundary scores    (a subset: boundary, graphs, audit, copula, monitors, scores)
Needs: numpy, scipy, networkx, scikit-learn
Out:   results.json

  1. Boundary     exhaustive check of a two-actor transaction boundary, ten weakened variants (two of them
                  lifecycle faults: a ledger erased on restart or on renewal) and one weakened specification
  2. Interdiction width versus count on layered graphs with about eight million pathways
  3. Coverage     a missing-mass estimate under honest search, a shared blind spot, and withholding
  4. Audit        how often planted pathways expose a withholding searcher
  5. Correlation  joint miss rate of k monitors under graded correlation (closed form)
  6. Monitors     do learned monitors miss independently when stacked on one step, or on distinct steps?
  7. Scores       what four reward rules select (rank by take, conferred rank, caretaker, floor), and what a gate does to the first
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")          # single-threaded numerics keep runs bit-for-bit reproducible
import itertools
import json
import math
import warnings
from collections import deque

import numpy as np
import networkx as nx
from scipy import stats
from scipy.special import roots_hermitenorm
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
SEED = 20261110

# --------------------------------------------------------------------------
# Experiment 1: a boundary checked for every policy
# --------------------------------------------------------------------------
ALLOWANCE = 2
ACTORS, IDS, AMOUNTS, DESTS = (0, 1), (0, 1), (1, 2), (1, 0)      # dest 1 = approved, 0 = unapproved
PREPS = [("prep", i, j, a, d) for i in ACTORS for j in IDS for a in AMOUNTS for d in DESTS]
ACTIONS = PREPS + [("commit", 0), ("commit", 1), ("revoke",), ("advance",), ("restart",), ("renew",)]
VIOLATIONS = ("overspend", "replay", "revoked", "stale", "destination")
# total spend in the world, used-id bits, spend0, spend1, live, version, pending0, pending1, enforcer's ledger
INIT = (0, 0, 0, 0, 1, 0, None, None, 0)
VARIANTS = ("none", "reject_all", "per_action", "per_actor", "prep_time", "no_replay", "no_revoke", "no_version",
            "restart_reset", "renew_reset", "complete")


def violations(state, actor, spec=VIOLATIONS):
    """Which forbidden effects (among those the specification declares) would executing this request cause?"""
    z, used, z0, z1, live, ver, p0, p1, ledger = state
    j, a, d, pv = (p0, p1)[actor]
    bit = 1 << (2 * actor + j)
    out = []
    if z + a > ALLOWANCE: out.append("overspend")           # judged on the world, not on the ledger
    if used & bit: out.append("replay")
    if not live: out.append("revoked")
    if pv != ver: out.append("stale")
    if d != 1: out.append("destination")
    return [b for b in out if b in spec]


def allows(variant, state, actor):
    """Does this enforcer let the pending request execute?"""
    if variant == "none": return True
    if variant == "reject_all": return False
    z, used, z0, z1, live, ver, p0, p1, ledger = state
    j, a, d, pv = (p0, p1)[actor]
    bit = 1 << (2 * actor + j)
    budget = {"per_action": a <= ALLOWANCE, "per_actor": (z0, z1)[actor] + a <= ALLOWANCE,
              "prep_time": True}.get(variant, ledger + a <= ALLOWANCE)
    return (d == 1 and budget
            and (variant == "no_replay" or not used & bit)
            and (variant == "no_revoke" or live)
            and (variant == "no_version" or pv == ver))


def transition(variant, state, action, spec=VIOLATIONS):
    """Return (next_state, violations). A denied request leaves the state unchanged."""
    z, used, z0, z1, live, ver, p0, p1, ledger = state
    kind = action[0]
    if kind == "revoke": return (z, used, z0, z1, 0, ver, p0, p1, ledger), []
    if kind == "advance":                                   # the environment changes once; the grant lapses until renewed
        return (state if ver == 1 else (z, used, z0, z1, 0, 1, p0, p1, ledger)), []
    if kind == "restart":                                   # the enforcing process restarts; spent units stay spent
        return (z, used, z0, z1, live, ver, p0, p1, 0 if variant == "restart_reset" else ledger), []
    if kind == "renew":                                     # a trusted controller grants permission again
        return (z, used, z0, z1, 1, ver, p0, p1, 0 if variant == "renew_reset" else ledger), []
    if kind == "prep":
        _, i, j, a, d = action
        req = (j, a, d, ver)
        trial = (z, used, z0, z1, live, ver, req if i == 0 else p0, req if i == 1 else p1, ledger)
        if variant == "prep_time" and violations(trial, i): return state, []      # refused at preparation
        return trial, []
    i = action[1]
    if (p0, p1)[i] is None or not allows(variant, state, i): return state, []
    bad = violations(state, i, spec)
    j, a, d, pv = (p0, p1)[i]
    nxt = (z + a, used | (1 << (2 * i + j)), z0 + (a if i == 0 else 0), z1 + (a if i == 1 else 0), live, ver, p0, p1,
           ledger + a)
    return nxt, bad


def explore(variant, spec=VIOLATIONS):
    """Breadth-first search over every reachable accepted state and every action."""
    parent, queue, checks, witness = {INIT: None}, deque([INIT]), 0, {}
    while queue:
        s = queue.popleft()
        for act in ACTIONS:
            checks += 1
            t, bad = transition(variant, s, act, spec)
            if bad:
                for b in bad:
                    if b not in witness:
                        trace, cur = [act], s
                        while parent[cur] is not None:
                            cur, a = parent[cur]; trace.append(a)
                        witness[b] = trace[::-1]
                continue                                   # a violating successor is terminal
            if t not in parent:
                parent[t] = (s, act); queue.append(t)
    return len(parent), checks, witness, max(s[0] for s in parent)


def legit_tasks():
    ids = [(i, j) for i in ACTORS for j in IDS]
    single = [[(i, j, a)] for (i, j) in ids for a in AMOUNTS]
    pairs = [[(i, j, 1), (k, l, 1)] for (i, j) in ids for (k, l) in ids if (i, j) != (k, l)]
    return single + pairs


def completes(variant, task):
    s = INIT
    for (i, j, a) in task:
        s, _ = transition(variant, s, ("prep", i, j, a, 1))
        before = s[0]
        s, bad = transition(variant, s, ("commit", i))
        if bad or s[0] != before + a: return False
    return True


def experiment_boundary():
    out = {}
    T = legit_tasks()

    def record(variant, spec):
        n, checks, wit, top = explore(variant, spec)
        return {"safe_states": n, "checks": checks, "tasks_done": sum(completes(variant, t) for t in T), "tasks": len(T),
                "max_spend": top, "actions": len(ACTIONS),
                "violations": {k: len(w) for k, w in sorted(wit.items())},
                "witness": {k: [" ".join(map(str, a)) for a in w] for k, w in sorted(wit.items())}}

    for v in VARIANTS:
        out[v] = record(v, VIOLATIONS)
    # Specification-omission control: the cumulative-budget predicate is left out of the declared specification,
    # and the per-action enforcer is checked against what remains. The check passes; the world overspends.
    out["spec_omission"] = record("per_action", tuple(b for b in VIOLATIONS if b != "overspend"))
    return out


SRC, SNK = (-1, 0), (99, 0)     # source and sink labels (integer tuples keep runs reproducible)
STAGES = 8
OUT_DEGREE = 6
N_GRAPHS = 200
BETA = 0.5          # searchers favour cheap pathways: P(path) ~ exp(-BETA * cost)
MAX_BUDGET = 32
FAMILIES = ("narrow", "hub", "wide")


# --------------------------------------------------------------------------
# Experiment 2: layered pathway graphs
# --------------------------------------------------------------------------
def make_graph(family, rng):
    """Return stage sizes, adjacency matrices A[i] (stage i -> i+1) and vertex costs."""
    if family == "narrow":
        sizes = [30] * STAGES
        sizes[int(rng.integers(2, 6))] = 6          # one narrow stage, away from the ends
    else:
        sizes = [27] * STAGES
    adj = []
    for i in range(STAGES - 1):
        a, b = sizes[i], sizes[i + 1]
        d = min(OUT_DEGREE, b)
        if family == "hub":                           # heavy-tailed preference for targets
            w = 1.0 / np.arange(1, b + 1) ** 1.5
            w = rng.permutation(w)
            w = w / w.sum()
        else:
            w = np.full(b, 1.0 / b)
        A = np.zeros((a, b))
        for v in range(a):
            A[v, rng.choice(b, size=d, replace=False, p=w)] = 1.0
        adj.append(A)
    costs = [rng.uniform(1.0, 10.0, size=n) for n in sizes]
    return sizes, adj, costs


def count_paths(sizes, adj, alive):
    c = alive[-1].astype(float)
    for i in range(STAGES - 2, -1, -1):
        c = alive[i] * (adj[i] @ c)
    return float(c.sum())


def cheapest(sizes, adj, costs, alive):
    INF = 1e18
    c = np.where(alive[-1], costs[-1], INF)
    for i in range(STAGES - 2, -1, -1):
        nxt = np.where(adj[i] > 0, c[None, :], INF).min(axis=1)
        c = np.where(alive[i], costs[i] + nxt, INF)
    m = c.min()
    return math.inf if m >= INF / 2 else float(m)


def sample_paths(sizes, adj, costs, n, rng):
    """Exact samples from P(path) proportional to exp(-BETA * cost)."""
    Z = [None] * STAGES
    Z[-1] = np.exp(-BETA * costs[-1])
    for i in range(STAGES - 2, -1, -1):
        Z[i] = np.exp(-BETA * costs[i]) * (adj[i] @ Z[i + 1])
    paths = np.zeros((n, STAGES), dtype=int)
    p0 = Z[0] / Z[0].sum()
    paths[:, 0] = rng.choice(sizes[0], size=n, p=p0)
    for i in range(STAGES - 1):
        for r in range(n):
            w = adj[i][paths[r, i]] * Z[i + 1]
            paths[r, i + 1] = rng.choice(sizes[i + 1], p=w / w.sum())
    return paths


def node_cut(edges):
    """Minimum s-t vertex cut of a graph given as a set of ((stage, v), (stage, v)) edges."""
    H = nx.DiGraph()
    H.add_edges_from(sorted(edges))
    return nx.minimum_node_cut(H, SRC, SNK)


def full_edges(sizes, adj):
    E = set()
    for v in range(sizes[0]):
        E.add((SRC, (0, v)))
    for v in range(sizes[-1]):
        E.add(((STAGES - 1, v), SNK))
    for i in range(STAGES - 1):
        for a, b in zip(*np.nonzero(adj[i])):
            E.add(((i, int(a)), (i + 1, int(b))))
    return E


def reported_edges(paths):
    E = set()
    for p in paths:
        E.add((SRC, (0, int(p[0]))))
        E.add(((STAGES - 1, int(p[-1])), SNK))
        for i in range(STAGES - 1):
            E.add(((i, int(p[i])), (i + 1, int(p[i + 1]))))
    return E


def usage(paths):
    u = {}
    for p in paths:
        for i, v in enumerate(p):
            u[(i, int(v))] = u.get((i, int(v)), 0) + 1
    return u


def evaluate(order, sizes, adj, costs, total, base):
    """Apply interdictions in order; return blocked share, severed flag, margin multiple per budget."""
    alive = [np.ones(n, dtype=bool) for n in sizes]
    blocked, severed, gain = [0.0], [0.0], [1.0]
    for b in range(1, MAX_BUDGET + 1):
        if b <= len(order):
            i, v = order[b - 1]
            alive[i][v] = False
        left = count_paths(sizes, adj, alive)
        blocked.append(1.0 - left / total)
        severed.append(1.0 if left == 0 else 0.0)
        gain.append(cheapest(sizes, adj, costs, alive) / base)
    return blocked, severed, gain


def experiment_interdiction(rng):
    out = {}
    for fam in FAMILIES:
        acc = {s: {"blocked": [], "severed": [], "gain": []} for s in ("cut", "patch_specific", "patch_random")}
        counts, widths, rep_cut_sizes, seen_share = [], [], [], []
        sever_by_n = {n: [] for n in (30, 100, 300, 1000, 3000)}
        for _ in range(N_GRAPHS):
            sizes, adj, costs = make_graph(fam, rng)
            alive = [np.ones(n, dtype=bool) for n in sizes]
            total = count_paths(sizes, adj, alive)
            base = cheapest(sizes, adj, costs, alive)
            counts.append(total)
            widths.append(len(node_cut(full_edges(sizes, adj))))

            paths = sample_paths(sizes, adj, costs, 300, rng)
            uniq = np.unique(paths, axis=0)
            seen_share.append(len(uniq) / total)
            use = usage(paths)

            # cut-guided: minimum cut of the reported graph, busiest vertex first
            cut = sorted(node_cut(reported_edges(paths)), key=lambda x: (-use[x], x))
            rep_cut_sizes.append(len(cut))
            b, s, g = evaluate(cut, sizes, adj, costs, total, base)
            acc["cut"]["blocked"].append(b); acc["cut"]["severed"].append(s); acc["cut"]["gain"].append(g)

            # patching: take reported pathways cheapest first, remove one vertex from each open one
            pc = np.array([sum(costs[i][p[i]] for i in range(STAGES)) for p in uniq])
            order_paths = uniq[np.argsort(pc)]
            for name in ("patch_specific", "patch_random"):
                removed, order = set(), []
                for p in order_paths:
                    if len(order) >= MAX_BUDGET:
                        break
                    verts = [(i, int(p[i])) for i in range(STAGES)]
                    if any(v in removed for v in verts):
                        continue
                    if name == "patch_specific":       # the vertex most peculiar to this pathway
                        v = min(verts, key=lambda x: (use[x], x))
                    else:                               # any vertex on this pathway
                        v = verts[int(rng.integers(STAGES))]
                    removed.add(v); order.append(v)
                b, s, g = evaluate(order, sizes, adj, costs, total, base)
                acc[name]["blocked"].append(b); acc[name]["severed"].append(s); acc[name]["gain"].append(g)

            # does the cut of the reported graph sever the whole graph, by number of reports?
            for n in sever_by_n:
                pn = sample_paths(sizes, adj, costs, n, rng)
                al = [np.ones(k, dtype=bool) for k in sizes]
                for (i, v) in node_cut(reported_edges(pn)):
                    al[i][v] = False
                sever_by_n[n].append(1.0 if count_paths(sizes, adj, al) == 0 else 0.0)

        fam_out = {
            "paths_millions": float(np.mean(counts) / 1e6),
            "width_median": float(np.median(widths)),
            "reported_cut_median": float(np.median(rep_cut_sizes)),
            "seen_share": float(np.mean(seen_share)),
            "sever_by_n": {str(n): float(np.mean(v)) for n, v in sever_by_n.items()},
        }
        for name, d in acc.items():
            gain = np.array(d["gain"])
            fam_out[name] = {
                "blocked_mean": np.mean(d["blocked"], axis=0).tolist(),
                "severed_share": np.mean(d["severed"], axis=0).tolist(),
                "gain_median": [None if math.isinf(x) else float(x) for x in np.median(gain, axis=0)],
            }
        out[fam] = fam_out
    return out


# --------------------------------------------------------------------------
# Experiment 3: coverage estimates
# --------------------------------------------------------------------------
def experiment_coverage(rng, K=2000, searchers=4, reports=3000, trials=200):
    res = {c: {"est": [], "true": [], "critical": []} for c in ("honest", "blind", "withheld")}
    n = searchers * reports
    for _ in range(trials):
        a = rng.permutation(1.0 / np.arange(1, K + 1))     # heavy-tailed discoverability
        a = a / a.sum()                                     # the outside attacker's search distribution
        critical = np.zeros(K, dtype=bool)
        critical[rng.choice(K, size=K // 20, replace=False)] = True      # 5% are critical
        blind = np.zeros(K, dtype=bool)
        blind[rng.choice(K, size=int(0.15 * K), replace=False)] = True   # 15% no searcher can see
        for cond in res:
            p = a.copy()
            if cond == "blind":
                p[blind] = 0.0
            if cond == "withheld":
                p[critical] = 0.0      # searchers file the reports of an honest searcher in a world without them
            p = p / p.sum()
            draws = rng.choice(K, size=n, p=p)
            cnt = np.bincount(draws, minlength=K)
            n1 = int((cnt == 1).sum())
            found = cnt > 0
            res[cond]["est"].append(1.0 - n1 / n)                         # Good-Turing coverage estimate
            res[cond]["true"].append(float(a[found].sum()))               # mass an attacker would still find covered
            res[cond]["critical"].append(float(a[found & critical].sum() / a[critical].sum()))
    return {c: {k: float(np.mean(v)) for k, v in d.items()} for c, d in res.items()}


# --------------------------------------------------------------------------
# Experiment 4: planted-pathway audit
# --------------------------------------------------------------------------
def experiment_audit(recall=0.8, alpha=0.01):
    """Exact power of a one-sided Fisher test that recall on planted critical pathways
    is lower than on planted ordinary ones, when a withholding searcher reports a planted
    critical pathway only if it recognises it as planted (probability r)."""
    out = {}
    rs = [round(x, 2) for x in np.linspace(0.0, 1.0, 21)]
    for P in (40, 200):
        xs = np.arange(P + 1)
        # reject[xc, xo] = True if Fisher one-sided p <= alpha
        reject = np.zeros((P + 1, P + 1), dtype=bool)
        for xc in xs:
            for xo in xs:
                if xc >= xo:
                    continue
                # P(X <= xc) under hypergeometric with margins fixed
                pval = stats.hypergeom.cdf(xc, 2 * P, xc + xo, P)
                reject[xc, xo] = pval <= alpha
        po = stats.binom.pmf(xs, P, recall)
        power = []
        for r in rs:
            pc = stats.binom.pmf(xs, P, recall * r)
            power.append(float(pc @ reject @ po))
        out[str(P)] = {"r": rs, "power": power}
    return out


# --------------------------------------------------------------------------
# Experiment 5: saturation under graded correlation
# --------------------------------------------------------------------------
def experiment_copula(m=0.1):
    """k monitors each miss with probability m. Their misses share a latent factor with
    correlation rho (Gaussian copula). Joint miss rate = E[ Phi((t + sqrt(rho) f)/sqrt(1-rho))^k ]."""
    x, w = roots_hermitenorm(200)
    w = w / w.sum()
    t = stats.norm.ppf(m)
    out = {}
    for rho in (0.0, 0.3, 0.6, 0.9):
        row = []
        for k in range(1, 9):
            if rho == 0.0:
                row.append(m ** k)
            else:
                cond = stats.norm.cdf((t + math.sqrt(rho) * x) / math.sqrt(1 - rho))
                row.append(float((w * cond ** k).sum()))
        out[str(rho)] = row
    return out


# --------------------------------------------------------------------------
# Experiment 6: do learned monitors miss independently?
#
# A step is a point in 20 dimensions. A hidden random function decides whether it is hazardous.
# Monitors are classifiers, each trained on its own labelled sample and set to a 5% false-alarm rate.
#   same      six neural networks stacked on ONE step (the attacker's surrogate is a seventh)
#   mixed     six other model families stacked on ONE step (the attacker never sees them)
#   distinct  the mixed monitors, each reading a DIFFERENT step of a six-step pathway
# --------------------------------------------------------------------------
DIM, HIDDEN, N_TRAIN, N_CAL, N_PATH, GATES = 20, 8, 2000, 4000, 1000, 6
FALSE_ALARM, HAZ_SHARE = 0.05, 0.25
ATTACK_ITERS, ATTACK_CANDS = 20, 12
BUDGETS = (0, 1.0, 2.0, 3.0)                # how far an edit may move a step
N_TASKS = 24


class Task:
    def __init__(self, rng):
        self.rng = rng
        self.W = rng.normal(size=(HIDDEN, DIM)) / math.sqrt(DIM) * 2.0
        self.b = rng.normal(size=HIDDEN) * 0.5
        self.a = rng.normal(size=HIDDEN)
        self.tau = np.quantile(self.h(rng.normal(size=(50000, DIM))), 1 - HAZ_SHARE)

    def h(self, X):
        return np.tanh(X @ self.W.T + self.b) @ self.a

    def hazardous(self, X):
        return self.h(X) > self.tau

    def sample(self, n, hazard=None):
        out = []
        while sum(len(o) for o in out) < n:
            X = self.rng.normal(size=(4 * n, DIM))
            if hazard is not None:
                X = X[self.hazardous(X) == hazard]
            out.append(X)
        return np.concatenate(out)[:n]


class Monitor:
    """A classifier with a threshold set for a 5% false-alarm rate on benign steps."""
    def __init__(self, model, task):
        X = task.sample(N_TRAIN)
        self.model = model.fit(X, task.hazardous(X))
        self.thr = np.quantile(self.score(task.sample(N_CAL, hazard=False)), 1 - FALSE_ALARM)

    def score(self, X):
        m = self.model
        return m.decision_function(X) if hasattr(m, "decision_function") else m.predict_proba(X)[:, 1]

    def misses(self, X):
        return self.score(X) <= self.thr


def mlp(seed):
    return MLPClassifier(hidden_layer_sizes=(48, 48), max_iter=120, random_state=seed)


def mixed_models(seed):
    return [SVC(kernel="rbf", C=3.0, gamma="scale"),
            HistGradientBoostingClassifier(max_iter=150, random_state=seed),
            RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1),
            QuadraticDiscriminantAnalysis(reg_param=0.05),
            KNeighborsClassifier(n_neighbors=15),
            LogisticRegression(max_iter=500)]


def attack(task, surrogate, X0, rho):
    """Edit hazardous steps to lower the surrogate's score. An edit must keep the step at least as
    hazardous as it was (the payload still works) and may move it at most rho from the original."""
    if rho == 0:
        return X0.copy()
    X, h0 = X0.copy(), task.h(X0)
    best = surrogate.score(X)
    n = len(X)
    for _ in range(ATTACK_ITERS):
        cand = X[:, None, :] + (rho / 4) * task.rng.normal(size=(n, ATTACK_CANDS, DIM)) / math.sqrt(DIM) * 2
        flat = cand.reshape(-1, DIM)
        ok = (task.h(flat) >= np.repeat(h0, ATTACK_CANDS)) & (np.linalg.norm(flat - np.repeat(X0, ATTACK_CANDS, axis=0), axis=1) <= rho)
        s = np.where(ok, surrogate.score(flat), np.inf).reshape(n, ATTACK_CANDS)
        j = s.argmin(axis=1)
        better = s[np.arange(n), j] < best
        X[better] = cand[np.arange(n), j][better]
        best = np.where(better, s[np.arange(n), j], best)
    return X


def stacked(miss):
    """miss: gates x steps boolean. Joint miss and product of marginals, averaged over subsets of each size."""
    marg = miss.mean(axis=1)
    joint, prod = [], []
    for k in range(1, GATES + 1):
        subs = list(itertools.combinations(range(GATES), k))
        joint.append(float(np.mean([miss[list(s)].all(axis=0).mean() for s in subs])))
        prod.append(float(np.mean([np.prod(marg[list(s)]) for s in subs])))
    return {"marginal": float(marg.mean()), "joint": joint, "product": prod}


def one_task(seed):
    rng = np.random.default_rng(seed)
    task = Task(rng)
    same = [Monitor(mlp(seed * 10 + i), task) for i in range(GATES)]
    mixed = [Monitor(m, task) for m in mixed_models(seed)]
    surrogate = Monitor(mlp(seed * 10 + 9), task)
    X = task.sample(N_PATH, hazard=True)                               # one step, read by every stacked gate
    steps = [task.sample(N_PATH, hazard=True) for _ in range(GATES)]       # six steps, one gate each
    out = {"surrogate_pass": {}, "same": {}, "mixed": {}, "distinct": {}}
    for rho in BUDGETS:
        Xa = attack(task, surrogate, X, rho)
        out["surrogate_pass"][rho] = float(surrogate.misses(Xa).mean())
        out["same"][rho] = stacked(np.array([g.misses(Xa) for g in same]))
        out["mixed"][rho] = stacked(np.array([g.misses(Xa) for g in mixed]))
        out["distinct"][rho] = stacked(np.array([g.misses(attack(task, surrogate, S, rho)) for g, S in zip(mixed, steps)]))
    return out


def experiment_monitors(n_tasks=N_TASKS):
    reps = [one_task(SEED + s) for s in range(n_tasks)]
    def agg(get):
        v = np.array([get(r) for r in reps], dtype=float)
        half = 1.96 * v.std(axis=0, ddof=1) / math.sqrt(len(reps))
        return v.mean(axis=0).tolist(), half.tolist()
    out = {"tasks": len(reps), "budgets": list(BUDGETS),
           "surrogate_pass": {str(r): agg(lambda x: x["surrogate_pass"][r])[0] for r in BUDGETS}}
    for pipe in ("same", "mixed", "distinct"):
        out[pipe] = {}
        for r in BUDGETS:
            d = {"marginal": agg(lambda x: x[pipe][r]["marginal"])[0]}
            for key in ("joint", "product"):
                d[key], d[key + "_ci"] = agg(lambda x: x[pipe][r][key])
            out[pipe][str(r)] = d
    return out


# --------------------------------------------------------------------------
# Experiment 7: what the score selects (warrior, conferred rank, caretaker, floor)
# --------------------------------------------------------------------------
SC_N, SC_GENS, SC_REPS, SC_SWEEP_REPS = 2000, 80, 40, 20
SC_MEAN, SC_SIGMA, SC_HARM_P, SC_HARM, SC_TOP, SC_MUT = 0.8, 1.5, 0.05, 20.0, 0.10, 0.03
SC_CAP_GAIN, SC_CAP_MUT, SC_FLOOR = 0.3, 0.05, 0.20
SC_RULES = ("taken_rank", "conferred_rank", "caretaker", "floor")
SC_TOPS = (0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
SC_HARMS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
SC_GATES = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def evolve(rule, top, harm_p, gate, reps, rng):
    """Each agent carries two heritable traits: a capacity c, which multiplies everything it produces, and a share s of
    effort put into a gamble; the rest of its effort goes to a safe action that pays one unit per unit of capacity.
    The gamble pays the agent less on average (SC_MEAN) with high variance and, with probability harm_p * s, imposes a
    large loss on the people the agent serves. A gate refuses each agent's gamble with probability `gate`; a refused
    gamble pays nothing and harms nobody. Four rules allocate reproduction.
      taken_rank      the top fraction `top` by own take reproduce (a warrior's contest)
      conferred_rank  the same contest, but those who were harmed confer nothing on whoever harmed them
      caretaker       reproduction in proportion to the welfare delivered to those served, capped at what they need (1)
      floor           the lowest fifth by welfare delivered leave no offspring; everyone else reproduces equally
    Returns mean gamble share and mean capacity per generation, and harm and welfare in the last generation."""
    share = np.zeros((reps, SC_GENS + 1)); cap = np.zeros((reps, SC_GENS + 1))
    harm_rate = np.zeros(reps); welfare = np.zeros(reps)
    for r in range(reps):
        s = np.clip(0.5 + 0.1 * rng.standard_normal(SC_N), 0, 1)
        c = rng.standard_normal(SC_N)
        share[r, 0] = s.mean(); cap[r, 0] = c.mean()
        for g in range(SC_GENS):
            refused = rng.random(SC_N) < gate
            gamble = np.where(refused, 0.0, SC_MEAN + SC_SIGMA * rng.standard_normal(SC_N))
            take = np.exp(SC_CAP_GAIN * c) * ((1 - s) + s * gamble)
            harmed = ~refused & (rng.random(SC_N) < harm_p * s)
            delivered = take - SC_HARM * harmed
            if rule == "taken_rank":
                w = (take >= np.quantile(take, 1 - top)).astype(float)
            elif rule == "conferred_rank":
                w = ((take >= np.quantile(take, 1 - top)) & ~harmed).astype(float)
            elif rule == "caretaker":
                w = np.clip(delivered, 0, 1)
            else:
                w = (delivered >= np.quantile(delivered, SC_FLOOR)).astype(float)
            if g == SC_GENS - 1:
                harm_rate[r] = harmed.mean(); welfare[r] = delivered.mean()
            if w.sum() == 0: w = np.ones(SC_N)
            parents = rng.choice(SC_N, size=SC_N, p=w / w.sum())
            s = np.clip(s[parents] + SC_MUT * rng.standard_normal(SC_N), 0, 1)
            c = c[parents] + SC_CAP_MUT * rng.standard_normal(SC_N)
            share[r, g + 1] = s.mean(); cap[r, g + 1] = c.mean()
    gain = cap[:, -1] - cap[:, 0]
    return {"share": share.mean(0).round(4).tolist(), "capacity": cap.mean(0).round(4).tolist(),
            "final_share": float(share[:, -1].mean()), "final_share_sd": float(share[:, -1].std(ddof=1)),
            "capacity_gain": float(gain.mean()), "capacity_gain_sd": float(gain.std(ddof=1)),
            "harm_rate": float(harm_rate.mean()), "welfare": float(welfare.mean())}


def experiment_scores():
    out = {"rules": {}, "by_top": {}, "by_harm": {}, "by_gate": {}}
    keep = ("final_share", "final_share_sd", "capacity_gain", "harm_rate")
    for i, rule in enumerate(SC_RULES):
        out["rules"][rule] = evolve(rule, SC_TOP, SC_HARM_P, 0.0, SC_REPS, np.random.default_rng(SEED + 7000 + i))
    for i, top in enumerate(SC_TOPS):          # how steep is the prize?
        d = evolve("taken_rank", top, SC_HARM_P, 0.0, SC_SWEEP_REPS, np.random.default_rng(SEED + 7100 + i))
        out["by_top"][str(top)] = {k: d[k] for k in keep}
    for i, hp in enumerate(SC_HARMS):          # how often does the gamble harm someone who can then withhold standing?
        d = evolve("conferred_rank", SC_TOP, hp, 0.0, SC_SWEEP_REPS, np.random.default_rng(SEED + 7200 + i))
        out["by_harm"][str(hp)] = {k: d[k] for k in keep}
    for i, gate in enumerate(SC_GATES):        # how often is the gamble refused before it pays?
        d = evolve("taken_rank", SC_TOP, SC_HARM_P, gate, SC_SWEEP_REPS, np.random.default_rng(SEED + 7300 + i))
        out["by_gate"][str(gate)] = {k: d[k] for k in keep}
    out["params"] = {"n": SC_N, "generations": SC_GENS, "replicates": SC_REPS, "sweep_replicates": SC_SWEEP_REPS,
                     "gamble_mean": SC_MEAN, "sigma": SC_SIGMA, "harm_p": SC_HARM_P, "harm": SC_HARM, "top": SC_TOP,
                     "mutation": SC_MUT, "capacity_gain": SC_CAP_GAIN, "capacity_mutation": SC_CAP_MUT, "floor": SC_FLOOR}
    return out


if __name__ == "__main__":
    import sys
    only = set(sys.argv[1:])          # optional: run a subset, e.g. "python3 experiments.py boundary scores"
    try:
        results = json.load(open("results.json")) if only else {}
    except FileNotFoundError:
        results = {}
    results["seed"] = SEED
    if not only or "boundary" in only:
        results["boundary"] = experiment_boundary()
    if not only or "graphs" in only:  # these two share one random stream and always run together
        rng = np.random.default_rng(SEED)
        results["interdiction"] = experiment_interdiction(rng)
        results["coverage"] = experiment_coverage(rng)
    if not only or "audit" in only:
        results["audit"] = experiment_audit()
    if not only or "copula" in only:
        results["copula"] = experiment_copula()
    if not only or "monitors" in only:
        results["monitors"] = experiment_monitors()
    if not only or "scores" in only:
        results["scores"] = experiment_scores()
    with open("results.json", "w") as f:
        json.dump(results, f, indent=1)

    for v, r in results["boundary"].items():
        print(f"{v:13s} states={r['safe_states']:6d} checks={r['checks']:7d} tasks={r['tasks_done']}/{r['tasks']} "
              f"max spend={r['max_spend']} shortest violation={r['violations']}")
    for fam in FAMILIES:
        d = results["interdiction"][fam]
        print(f"{fam:7s} paths={d['paths_millions']:.2f}M width={d['width_median']:.0f} six cuts: blocked "
              f"{d['cut']['blocked_mean'][6]:.3f} severed {d['cut']['severed_share'][6]:.3f}; 32 patches: blocked "
              f"{d['patch_specific']['blocked_mean'][32]:.3f} / {d['patch_random']['blocked_mean'][32]:.3f}")
    print("coverage:", {c: {k: round(v, 3) for k, v in d.items()} for c, d in results["coverage"].items()})
    M = results["monitors"]
    for r in M["budgets"]:
        print(f"edit budget {r}:", "  ".join(f"{p} marginal {M[p][str(r)]['marginal']:.3f} six-gate joint "
              f"{M[p][str(r)]['joint'][5]:.4f} (product {M[p][str(r)]['product'][5]:.4f})" for p in ("same", "mixed", "distinct")))
    S = results["scores"]
    for rule, d in S["rules"].items():
        print(f"{rule:15s} gamble share after {SC_GENS} generations: {d['final_share']:.3f} (sd {d['final_share_sd']:.3f})  "
              f"capacity gain {d['capacity_gain']:.2f} sd  harm rate {d['harm_rate']:.4f}  welfare delivered {d['welfare']:.3f}")
    print("prize share -> gamble share:", {k: round(v["final_share"], 3) for k, v in S["by_top"].items()})
    print("prize share -> capacity gain:", {k: round(v["capacity_gain"], 2) for k, v in S["by_top"].items()})
    print("harm probability -> gamble share (conferred rank):", {k: round(v["final_share"], 3) for k, v in S["by_harm"].items()})
    print("gate strength -> gamble share (warrior):", {k: round(v["final_share"], 3) for k, v in S["by_gate"].items()})

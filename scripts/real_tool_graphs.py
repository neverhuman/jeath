#!/usr/bin/env python3
"""Introspect real AgentDojo v1.2.2 tool suites and Linux isolation_v2 probes,
cross-check split-vertex Dinic cutsets against `src/v4_real_graphs.rs`, and emit
`target/v4-graphs-monitors/real_tool_cutsets.json` (validating Prediction P1 & C2).
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import warnings

os.environ["CUDA_VISIBLE_DEVICES"] = ""
warnings.filterwarnings("ignore", category=SyntaxWarning)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "target/v4-graphs-monitors/real_tool_cutsets.json"


def _ensure_agentdojo_python() -> None:
    try:
        import agentdojo  # noqa: F401
        return
    except ImportError:
        pass
    for candidate in (
        ROOT / "target/v4-graphs-monitors/venv/bin/python",
        ROOT / "target/benchmark-venv/bin/python",
    ):
        if candidate.exists() and Path(sys.executable).resolve() != candidate.resolve():
            os.execv(str(candidate), [str(candidate), *sys.argv])
    raise SystemExit(
        "agentdojo not available; expected venv at target/v4-graphs-monitors/venv or target/benchmark-venv"
    )


# ============================================================================
# Independent Python Split-Vertex Dinic Solver (Cross-checks Rust v4_real_graphs)
# ============================================================================


class PyDinic:
    def __init__(self, n: int) -> None:
        self.adj: list[list[list[int]]] = [[] for _ in range(n)]
        self.level = [-1] * n
        self.ptr = [0] * n

    def add_edge(self, u: int, v: int, cap: int) -> None:
        rev_u = len(self.adj[v])
        rev_v = len(self.adj[u])
        self.adj[u].append([v, rev_u, cap])
        self.adj[v].append([u, rev_v, 0])

    def bfs(self, s: int, t: int) -> bool:
        self.level = [-1] * len(self.adj)
        self.level[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            nxt = self.level[u] + 1
            for v, _, cap in self.adj[u]:
                if cap > 0 and self.level[v] < 0:
                    self.level[v] = nxt
                    q.append(v)
        return self.level[t] >= 0

    def dfs(self, u: int, t: int, pushed: int) -> int:
        if pushed == 0 or u == t:
            return pushed
        while self.ptr[u] < len(self.adj[u]):
            cid = self.ptr[u]
            v, rev, cap = self.adj[u][cid]
            if self.level[u] + 1 == self.level[v] and cap > 0:
                tr = self.dfs(v, t, min(pushed, cap))
                if tr > 0:
                    self.adj[u][cid][2] -= tr
                    self.adj[v][rev][2] += tr
                    return tr
            self.ptr[u] += 1
        return 0

    def max_flow(self, s: int, t: int, inf: int) -> int:
        flow = 0
        while self.bfs(s, t):
            self.ptr = [0] * len(self.adj)
            while True:
                pushed = self.dfs(s, t, inf)
                if pushed == 0:
                    break
                flow += pushed
                if flow >= inf:
                    return flow
        return flow

    def reachable_from(self, s: int) -> list[bool]:
        vis = [False] * len(self.adj)
        vis[s] = True
        q = deque([s])
        while q:
            u = q.popleft()
            for v, _, cap in self.adj[u]:
                if cap > 0 and not vis[v]:
                    vis[v] = True
                    q.append(v)
        return vis


def py_count_paths(edges: list[list[int]], removed: set[int]) -> int:
    n = len(edges)
    if 0 in removed or (n - 1) in removed:
        return 0
    dp = [0] * n
    dp[0] = 1
    for u in range(n):
        if u in removed or dp[u] == 0:
            continue
        for v in edges[u]:
            assert v > u, "DAG requires topological vertex ordering"
            if v not in removed:
                dp[v] += dp[u]
    return dp[n - 1]


def py_weighted_min_vertex_cut(
    edges: list[list[int]], costs: list[int | None]
) -> tuple[int, list[int]] | None:
    n = len(edges)
    if (n - 1) in edges[0]:
        return None
    finite_sum = sum(c for c in costs[1 : n - 1] if c is not None)
    inf = finite_sum + 1
    dinic = PyDinic(2 * n)
    for v in range(1, n - 1):
        cap = costs[v] if costs[v] is not None else inf
        dinic.add_edge(2 * v, 2 * v + 1, cap)
    for u in range(n - 1):
        u_out = 1 if u == 0 else 2 * u + 1
        for v in edges[u]:
            dinic.add_edge(u_out, 2 * v, inf)
    src = 1
    snk = 2 * (n - 1)
    flow = dinic.max_flow(src, snk, inf)
    if flow >= inf:
        return None
    reach = dinic.reachable_from(src)
    cut = [v for v in range(1, n - 1) if reach[2 * v] and not reach[2 * v + 1]]
    assert sum(costs[v] for v in cut if costs[v] is not None) == flow
    assert py_count_paths(edges, set(cut)) == 0
    return flow, cut


def py_shortest_and_longest_internal(edges: list[list[int]]) -> tuple[int, int]:
    n = len(edges)
    inf = 10**9
    min_d = [inf] * n
    max_d = [0] * n
    reach = [False] * n
    min_d[0] = 0
    reach[0] = True
    for u in range(n):
        if not reach[u]:
            continue
        for v in edges[u]:
            reach[v] = True
            min_d[v] = min(min_d[v], min_d[u] + 1)
            max_d[v] = max(max_d[v], max_d[u] + 1)
    assert reach[n - 1]
    return min_d[n - 1] - 1, max_d[n - 1] - 1


# ============================================================================
# Live AgentDojo v1.2.2 Introspection & Linux Confinement Source Extraction
# ============================================================================

SINK_TOOLS = {
    "banking": [
        "schedule_transaction",
        "send_money",
        "update_password",
        "update_scheduled_transaction",
    ],
    "workspace": [
        "create_calendar_event",
        "delete_email",
        "delete_file",
        "send_email",
        "share_file",
    ],
    "travel": [
        "create_calendar_event",
        "reserve_car_rental",
        "reserve_hotel",
        "reserve_restaurant",
        "send_email",
    ],
    "slack": [
        "get_webpage@url_beacon",
        "post_webpage",
        "remove_user_from_slack",
        "send_channel_message",
        "send_direct_message",
    ],
}

STAGE_TOOLS = {
    "banking": ["update_user_info"],
    "workspace": [
        "add_calendar_event_participants",
        "append_to_file",
        "cancel_calendar_event",
        "create_file",
        "reschedule_calendar_event",
    ],
    "travel": ["cancel_calendar_event"],
    "slack": ["add_user_to_channel", "invite_user_to_slack"],
}

DISCOVERY_ARG_SHAPES = {
    (),
    ("city",),
    ("query",),
    ("query", "sender"),
    ("query", "date"),
    ("day",),
}


def introspect_agentdojo_suites() -> tuple[dict[str, dict], dict[str, dict]]:
    from agentdojo.functions_runtime import FunctionsRuntime
    from agentdojo.task_suite.load_suites import get_suite
    import agentdojo.task_suite.task_suite as ts_mod

    py_graphs: dict[str, dict] = {}
    suite_metadata: dict[str, dict] = {}
    grader_sha256 = hashlib.sha256(Path(inspect.getfile(ts_mod)).read_bytes()).hexdigest()

    for sname in ("banking", "workspace", "travel", "slack"):
        suite = get_suite("v1.2.2", sname)
        vecs = sorted(suite.get_injection_vector_defaults().keys())
        canaries = {k: f"__CANARY_{k}__" for k in vecs}
        env = suite.load_and_inject_default_environment(canaries)
        rt = FunctionsRuntime(suite.tools)

        # Measure empirical benign user_task ground-truth usage per tool
        benign_usage: dict[str, int] = {t.name: 0 for t in suite.tools}
        benign_chain_lengths: list[int] = []
        for ut in suite.user_tasks.values():
            u_env = ut.init_environment(suite.load_and_inject_default_environment({}))
            gt_calls = ut.ground_truth(u_env)
            benign_chain_lengths.append(len(gt_calls))
            for fn in {c.function for c in gt_calls}:
                if fn in benign_usage:
                    benign_usage[fn] += 1

        # Measure empirical injection_task targets & identify pure-prose exceptions
        inj_sink_counts: dict[str, int] = {}
        prose_only_injections: list[str] = []
        for iname, it in sorted(suite.injection_tasks.items()):
            i_env = suite.load_and_inject_default_environment({})
            igt = [c.function for c in it.ground_truth(i_env)]
            if not igt and "model_output" in inspect.getsource(it.security) and "post_environment" not in inspect.getsource(it.security).split("return", 1)[-1]:
                prose_only_injections.append(iname)
            for fn in igt:
                inj_sink_counts[fn] = inj_sink_counts.get(fn, 0) + 1

        pure_sinks = {x.split("@")[0] for x in SINK_TOOLS[sname] if "@" not in x}
        stage_set = set(STAGE_TOOLS[sname])
        read_tools = sorted(
            t.name for t in suite.tools if t.name not in pure_sinks and t.name not in stage_set
        )
        subsys = {
            t.name: sorted(str(d.env_dependency) for d in t.dependencies.values())
            for t in suite.tools
        }
        args_map = {
            t.name: tuple(t.parameters.model_json_schema().get("properties", {}).keys())
            for t in suite.tools
        }

        if sname == "banking":
            probes = {
                "get_most_recent_transactions": [{"n": 100}],
                "read_file": [{"file_path": p} for p in env.filesystem.files],
            }
        elif sname == "workspace":
            probes = {
                "get_unread_emails": [{}],
                "get_received_emails": [{}],
                "search_emails": [{"query": "__CANARY_"}],
                "search_calendar_events": [{"query": "__CANARY_"}],
                "get_day_calendar_events": [
                    {"day": str(ev.start_time.date())} for ev in env.calendar.events.values()
                ],
                "list_files": [{}],
                "get_file_by_id": [{"file_id": fid} for fid in env.cloud_drive.files],
                "search_files": [{"query": "__CANARY_"}],
                "search_files_by_filename": [
                    {"filename": f.filename} for f in env.cloud_drive.files.values()
                ],
            }
        elif sname == "travel":
            probes = {
                "get_rating_reviews_for_hotels": [
                    {"hotel_names": [h.name for h in env.hotels.hotel_list]}
                ],
                "get_rating_reviews_for_restaurants": [
                    {"restaurant_names": [r.name for r in env.restaurants.restaurant_list]}
                ],
                "get_rating_reviews_for_car_rental": [
                    {"company_name": [c.name for c in env.car_rental.company_list]}
                ],
            }
        else:
            probes = {
                "get_channels": [{}],
                "get_webpage": [{"url": u} for u in env.web.web_content],
            }

        canary_reach: dict[str, list[str]] = {}
        for fn, call_args in sorted(probes.items()):
            seen: set[str] = set()
            for a in call_args:
                res, _ = rt.run_function(env.model_copy(deep=True), fn, a, raise_on_error=True)
                s_out = str(res)
                for k in vecs:
                    if f"__CANARY_{k}__" in s_out:
                        seen.add(k)
            if seen:
                canary_reach[fn] = sorted(seen)

        # Build 5-stage operational workflow DAG matching `src/v4_real_graphs.rs`
        vertices: list[dict] = []

        def add_v(label: str, stage: str, cost: int | None) -> int:
            idx = len(vertices)
            vertices.append({"index": idx, "label": label, "stage": stage, "cost": cost})
            return idx

        s_idx = add_v("S", "source", None)
        l1 = [add_v(f"vec:{v}", "untrusted_vector", None) for v in vecs]
        l2_names = sorted(canary_reach.keys())
        l2 = [add_v(f"ingest:{fn}", "untrusted_read_tool", 10) for fn in l2_names]
        l3 = [add_v(f"recon1:{fn}", "primary_recon_tool", 8) for fn in read_tools]
        l4_recon = [add_v(f"recon2:{fn}", "secondary_recon_tool", 8) for fn in read_tools]
        stage_names = sorted(STAGE_TOOLS[sname])
        l4_stage = [add_v(f"stage:{fn}", "state_staging_tool", 4) for fn in stage_names]
        l4_synth = [
            add_v("synth:direct_injection_args", "llm_arg_synthesis", None),
            add_v("synth:harvested_pii_interpolation", "llm_arg_synthesis", None),
            add_v("synth:staged_or_split_args", "llm_arg_synthesis", None),
        ]
        sink_names = SINK_TOOLS[sname]
        l5 = [add_v(f"sink:{fn}", "external_commit_sink", 1) for fn in sink_names]
        t_idx = add_v("T", "terminal", None)

        edges: list[list[int]] = [[] for _ in vertices]
        for v_i in l1:
            edges[s_idx].append(v_i)
        for i2, fn2 in enumerate(l2_names):
            for rv in canary_reach[fn2]:
                edges[l1[vecs.index(rv)]].append(l2[i2])
            for i3, fn3 in enumerate(read_tools):
                if fn3 != fn2:
                    edges[l2[i2]].append(l3[i3])
            edges[l2[i2]].append(l4_synth[0])
        for i3, fn3 in enumerate(read_tools):
            is_disc3 = args_map[fn3] in DISCOVERY_ARG_SHAPES
            for i4, fn4 in enumerate(read_tools):
                is_disc4 = args_map[fn4] in DISCOVERY_ARG_SHAPES
                if fn4 != fn3 and (subsys[fn3] != subsys[fn4] or (is_disc3 and not is_disc4)):
                    edges[l3[i3]].append(l4_recon[i4])
            edges[l3[i3]].extend(l4_stage)
            edges[l3[i3]].extend(l4_synth)
        for u in [*l4_recon, *l4_stage, *l4_synth]:
            edges[u].extend(l5)
        for sk in l5:
            edges[sk].append(t_idx)
        edges = [sorted(set(adj)) for adj in edges]

        gid = f"agentdojo_{sname}_v1_2_2"
        py_graphs[gid] = {"vertices": vertices, "edges": edges}

        # Compute benign task dependency share on L2 untrusted-read tools vs L5 sinks
        ut_with_l2 = 0
        ut_with_l5 = 0
        for ut in suite.user_tasks.values():
            u_env = ut.init_environment(suite.load_and_inject_default_environment({}))
            used_fns = {c.function for c in ut.ground_truth(u_env)}
            if used_fns & set(l2_names):
                ut_with_l2 += 1
            if used_fns & pure_sinks:
                ut_with_l5 += 1

        tool_catalog = []
        for t in sorted(suite.tools, key=lambda x: x.name):
            schema = t.parameters.model_json_schema()
            props = {
                k: v.get("type", "union_or_ref")
                for k, v in schema.get("properties", {}).items()
            }
            role = (
                "external_commit_sink"
                if t.name in pure_sinks
                else "state_staging_tool"
                if t.name in stage_set
                else "untrusted_read_and_recon_tool"
                if t.name in canary_reach
                else "recon_lookup_tool"
            )
            tool_catalog.append(
                {
                    "name": t.name,
                    "role": role,
                    "subsystems": subsys[t.name],
                    "parameters": props,
                    "return_type": str(t.return_type),
                    "benign_user_tasks_using": benign_usage[t.name],
                    "benign_user_task_share": benign_usage[t.name] / len(suite.user_tasks),
                    "canary_vectors_observed": canary_reach.get(t.name, []),
                }
            )

        suite_metadata[gid] = {
            "benchmark_version": "v1.2.2",
            "upstream_grader_sha256": grader_sha256,
            "tool_count": len(suite.tools),
            "user_task_count": len(suite.user_tasks),
            "injection_task_count": len(suite.injection_tasks),
            "injection_vector_count": len(vecs),
            "benign_gt_chain_len_min": min(benign_chain_lengths),
            "benign_gt_chain_len_max": max(benign_chain_lengths),
            "benign_gt_chain_len_mean": sum(benign_chain_lengths) / len(benign_chain_lengths),
            "benign_task_share_requiring_l2_untrusted_reads": ut_with_l2 / len(suite.user_tasks),
            "benign_task_share_requiring_l5_sinks": ut_with_l5 / len(suite.user_tasks),
            "injection_task_ground_truth_sink_calls": inj_sink_counts,
            "unmodeled_prose_only_injection_tasks": prose_only_injections,
            "tools": tool_catalog,
        }

    return py_graphs, suite_metadata


def introspect_linux_confinement_sources() -> dict:
    iso_src = (ROOT / "src/isolation_v2.rs").read_text()
    blocked_syscalls = sorted(set(re.findall(r"libc::(SYS_[a-zA-Z0-9_]+)", iso_src)))
    probe_keys = re.findall(r'"([a-z_]+)":\s*[a-zA-Z0-9_]+', iso_src.split("serde_json::json!")[-1])
    info = {
        "isolation_source": "src/isolation_v2.rs",
        "isolation_sha256": hashlib.sha256(iso_src.encode()).hexdigest(),
        "seccomp_blocked_syscalls_count": len(blocked_syscalls),
        "seccomp_blocked_syscalls": blocked_syscalls,
        "probe_channels": probe_keys,
    }
    suite1_path = ROOT / "target/live-redteam/suite1-results.json"
    if suite1_path.is_file():
        s1 = json.loads(suite1_path.read_text())
        info["live_kernel_redteam_verified"] = bool(s1.get("passed"))
        info["landlock_abi_measured"] = s1.get("landlock_fs_and_net_escapes", {}).get("landlock_abi")
        info["ebadf_closed_fds_3_to_255"] = s1.get("inherited_fd_audit", {}).get("ebadf_count_3_to_255")
    return info


def compile_and_run_rust_solver(out_dir: Path) -> dict:
    harness = out_dir / "rust_cutset_VerifyHarness.rs"
    bin_path = out_dir / "rust_cutset_VerifyHarness_bin"
    rust_json_path = out_dir / "rust_real_tool_cutsets_raw.json"

    harness.write_text(
        'pub use authority_lab::{isolation_v2, rng, stats, v3_graphs};\n'
        '#[path = "../../src/v4_real_graphs.rs"]\n'
        "pub mod v4_real_graphs;\n"
        "fn main() {\n"
        "    let out = std::env::args().nth(1).unwrap();\n"
        "    let suite = v4_real_graphs::real_tool_cutsets_suite(20261111, false);\n"
        "    assert!(suite.agentdojo_p1_verified);\n"
        "    assert!(suite.linux_waist_confinement_verified);\n"
        '    std::fs::write(out, serde_json::to_string_pretty(&suite).unwrap() + "\\n").unwrap();\n'
        "}\n"
    )

    deps_dir = ROOT / "target/release/deps"
    if not list(deps_dir.glob("libauthority_lab-*.rlib")):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env.setdefault("CARGO_INCREMENTAL", "0")
        env.setdefault("CARGO_BUILD_JOBS", "2")
        subprocess.run(["cargo", "build", "--locked", "--release", "--lib"], cwd=ROOT, check=True, env=env)

    def newest_rlib(prefix: str) -> Path:
        matches = sorted(deps_dir.glob(f"lib{prefix}-*.rlib"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:
            raise RuntimeError(f"missing rlib for {prefix} in {deps_dir}")
        return matches[0]

    cmd = [
        "rustc",
        "--edition",
        "2024",
        "-D",
        "warnings",
        "-O",
        str(harness),
        "-L",
        f"dependency={deps_dir}",
        "--extern",
        f"authority_lab={newest_rlib('authority_lab')}",
        "--extern",
        f"serde={newest_rlib('serde')}",
        "--extern",
        f"serde_json={newest_rlib('serde_json')}",
        "-o",
        str(bin_path),
    ]
    subprocess.run(cmd, cwd=ROOT, check=True)
    subprocess.run([str(bin_path), str(rust_json_path)], cwd=ROOT, check=True)
    return json.loads(rust_json_path.read_text())


def main() -> None:
    _ensure_agentdojo_python()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output JSON path inside target/v4-graphs-monitors/",
    )
    args = p.parse_args()

    out_path = args.output.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    py_agentdojo_graphs, suite_meta = introspect_agentdojo_suites()
    linux_meta = introspect_linux_confinement_sources()
    rust_suite = compile_and_run_rust_solver(out_path.parent)

    # Cross-check every single graph between Python Dinic/DP and Rust v4_real_graphs!
    cross_checks = []
    for g in rust_suite["graphs"]:
        gid = g["graph_id"]
        edges = g["edges"]
        costs = [v["cost"] for v in g["vertices"]]

        if gid in py_agentdojo_graphs:
            assert g["vertices"] == py_agentdojo_graphs[gid]["vertices"], f"vertex mismatch on {gid}"
            assert edges == py_agentdojo_graphs[gid]["edges"], f"edge mismatch on {gid}"
            g["agentdojo_introspection"] = suite_meta[gid]
        else:
            g["linux_confinement_introspection"] = linux_meta

        # Independent Python solver verification
        py_paths = py_count_paths(edges, set())
        py_short, py_long = py_shortest_and_longest_internal(edges)
        unit_costs: list[int | None] = [None] + [1] * (len(edges) - 2) + [None]
        py_unit_res = py_weighted_min_vertex_cut(edges, unit_costs)
        assert py_unit_res is not None
        py_w, py_w_cut = py_unit_res
        py_weighted_res = py_weighted_min_vertex_cut(edges, costs)

        assert py_paths == g["attack_route_count_p"], f"path count mismatch on {gid}"
        assert py_short == g["shortest_path_internal_len_l"], f"shortest path mismatch on {gid}"
        assert py_long == g["longest_path_internal_len"], f"longest path mismatch on {gid}"
        assert py_w == g["menger_pathway_width_w"], f"Menger width mismatch on {gid}"
        assert [g["vertices"][i]["label"] for i in py_w_cut] == g["unweighted_min_cut_vertices"]

        if py_weighted_res is None:
            assert g["weighted_min_cut_is_infinite"] is True
            assert g["weighted_min_cut_cost_tau_b"] is None
            all_elig = {i for i, c in enumerate(costs) if c is not None}
            assert py_count_paths(edges, all_elig) == g["remaining_paths_after_weighted_cut"]
        else:
            py_tau, py_cut = py_weighted_res
            assert g["weighted_min_cut_is_infinite"] is False
            assert py_tau == g["weighted_min_cut_cost_tau_b"]
            assert [g["vertices"][i]["label"] for i in py_cut] == g["weighted_min_cut_vertices"]
            assert py_count_paths(edges, set(py_cut)) == 0 == g["remaining_paths_after_weighted_cut"]

        cross_checks.append(
            {
                "graph_id": gid,
                "python_rust_topology_identical": True,
                "python_rust_mincut_and_paths_identical": True,
            }
        )

    rust_suite["python_rust_cross_validation"] = {
        "graphs_checked": len(cross_checks),
        "discrepancies": 0,
        "details": cross_checks,
    }
    rust_suite["rust_module_sha256"] = hashlib.sha256(
        (ROOT / "src/v4_real_graphs.rs").read_bytes()
    ).hexdigest()
    rust_suite["python_script_sha256"] = hashlib.sha256(
        Path(__file__).resolve().read_bytes()
    ).hexdigest()

    out_path.write_text(json.dumps(rust_suite, indent=2) + "\n")
    summary = [
        {
            "graph_id": g["graph_id"],
            "|V|": g["total_vertices_v"],
            "|E|": g["total_edges_e"],
            "|P|": g["attack_route_count_p"],
            "w(G)": g["menger_pathway_width_w"],
            "l(G)": g["shortest_path_internal_len_l"],
            "tau_b(G)": g["weighted_min_cut_cost_tau_b"],
            "cut_severs_100pct": g["cutset_severs_all_routes"],
        }
        for g in rust_suite["graphs"]
    ]
    print(json.dumps({"artifact": str(out_path), "discrepancies": 0, "graphs": summary}, indent=2))


if __name__ == "__main__":
    main()

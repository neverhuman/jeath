//! Graph bottleneck & interdiction models for Wave 2 V3:
//! - H10 (`weighted_cuts.json`): Node-splitting weighted & eligibility-constrained Dinic max-flow
//!   cutset solver, `u128` path-count DP, exhaustive small-DAG brute-force cross-check, and
//!   `Superintelligence` Appendix A scaling families (`57,600` to `966,367,641,600` paths).
//! - H11 (`interdiction.json`): Pure-Rust port of `tips/experiments.py` Experiment 2 over
//!   8-stage sparse DAG ensembles (`narrow`, `hub`, `wide`, ~7.6M–8.4M pathways), Boltzmann
//!   low-cost path reporting (`beta = 0.5`), comparing reported-subgraph Dinic `cut` against
//!   `patch_specific` and `patch_random` across budgets `0..=32` and report counts
//!   `[30, 100, 300, 1000, 3000]`.
//!
//! Claim boundary (`AGENTS.md`): Exact combinatorial graph solvers and controlled synthetic DAG
//! ensembles; they do not certify completeness of real-world deployment attack graphs.

use crate::rng::Rng;
use serde::Serialize;
use std::collections::{BTreeMap, VecDeque};

// ============================================================================
// Dinic Max-Flow / Min-Cut Core (Zero external crates)
// ============================================================================

#[derive(Clone, Copy)]
struct FlowEdge {
    to: usize,
    rev: usize,
    cap: i64,
}

struct Dinic {
    adj: Vec<Vec<FlowEdge>>,
    level: Vec<i32>,
    ptr: Vec<usize>,
}

impl Dinic {
    fn new(n: usize) -> Self {
        Self {
            adj: vec![Vec::new(); n],
            level: vec![-1; n],
            ptr: vec![0; n],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, cap: i64) {
        let rev_from = self.adj[to].len();
        let rev_to = self.adj[from].len();
        self.adj[from].push(FlowEdge {
            to,
            rev: rev_from,
            cap,
        });
        self.adj[to].push(FlowEdge {
            to: from,
            rev: rev_to,
            cap: 0,
        });
    }

    fn bfs(&mut self, s: usize, t: usize) -> bool {
        self.level.fill(-1);
        self.level[s] = 0;
        let mut q = VecDeque::new();
        q.push_back(s);
        while let Some(v) = q.pop_front() {
            let next_lvl = self.level[v] + 1;
            for e in &self.adj[v] {
                if e.cap > 0 && self.level[e.to] < 0 {
                    self.level[e.to] = next_lvl;
                    q.push_back(e.to);
                }
            }
        }
        self.level[t] >= 0
    }

    fn dfs(&mut self, v: usize, t: usize, pushed: i64) -> i64 {
        if pushed == 0 || v == t {
            return pushed;
        }
        while self.ptr[v] < self.adj[v].len() {
            let cid = self.ptr[v];
            let to = self.adj[v][cid].to;
            let cap = self.adj[v][cid].cap;
            if self.level[v] + 1 == self.level[to] && cap > 0 {
                let tr = self.dfs(to, t, pushed.min(cap));
                if tr > 0 {
                    let rev = self.adj[v][cid].rev;
                    self.adj[v][cid].cap -= tr;
                    self.adj[to][rev].cap += tr;
                    return tr;
                }
            }
            self.ptr[v] += 1;
        }
        0
    }

    fn max_flow(&mut self, s: usize, t: usize, inf: i64) -> i64 {
        let mut flow = 0i64;
        while self.bfs(s, t) {
            self.ptr.fill(0);
            loop {
                let pushed = self.dfs(s, t, inf);
                if pushed == 0 {
                    break;
                }
                flow += pushed;
                if flow >= inf {
                    return flow;
                }
            }
        }
        flow
    }

    fn reachable_from(&self, s: usize) -> Vec<bool> {
        let mut vis = vec![false; self.adj.len()];
        let mut q = VecDeque::new();
        vis[s] = true;
        q.push_back(s);
        while let Some(u) = q.pop_front() {
            for e in &self.adj[u] {
                if e.cap > 0 && !vis[e.to] {
                    vis[e.to] = true;
                    q.push_back(e.to);
                }
            }
        }
        vis
    }
}

// ============================================================================
// H10: Weighted & Eligibility-Constrained Cutsets (`weighted_cuts.json`)
// ============================================================================

#[derive(Clone, Debug)]
pub struct WeightedDag {
    /// Topological adjacency list: `0` is start `S`, `edges.len() - 1` is terminal `T`.
    pub edges: Vec<Vec<usize>>,
    /// Blocking cost for internal vertices `1..n-1`: `Some(cost)` (`cost > 0`) for eligible
    /// control points, `None` for ungateable/ineligible vertices (`+infinity`).
    pub costs: Vec<Option<u64>>,
}

impl WeightedDag {
    pub fn complete_layered(widths: &[usize], singleton_cost: u64) -> Self {
        assert!(!widths.is_empty() && widths.iter().all(|&w| w > 0));
        assert!(singleton_cost > 0);
        let internal: usize = widths.iter().sum();
        let n = internal + 2; // 0 = S, 1..=internal = layers, internal + 1 = T
        let mut stages = Vec::with_capacity(widths.len() + 2);
        stages.push(vec![0usize]);
        let mut costs = vec![None; n];
        let mut next = 1usize;
        for &w in widths {
            let layer: Vec<usize> = (next..next + w).collect();
            let c = if w == 1 { singleton_cost } else { 1 };
            for &v in &layer {
                costs[v] = Some(c);
            }
            stages.push(layer);
            next += w;
        }
        stages.push(vec![next]);
        let mut edges = vec![Vec::new(); n];
        for pair in stages.windows(2) {
            for &u in &pair[0] {
                edges[u] = pair[1].clone();
            }
        }
        Self { edges, costs }
    }

    pub fn count_paths(&self, removed: &[usize]) -> u128 {
        let n = self.edges.len();
        let mut is_removed = vec![false; n];
        for &r in removed {
            if r < n {
                is_removed[r] = true;
            }
        }
        if is_removed[0] || is_removed[n - 1] {
            return 0;
        }
        let mut dp = vec![0u128; n];
        dp[0] = 1;
        for u in 0..n {
            if is_removed[u] || dp[u] == 0 {
                continue;
            }
            for &v in &self.edges[u] {
                assert!(v > u, "WeightedDag requires topological vertex ordering");
                if !is_removed[v] {
                    dp[v] = dp[v].checked_add(dp[u]).expect("u128 path count overflow");
                }
            }
        }
        dp[n - 1]
    }

    /// Computes minimum weighted internal vertex cut `\tau_b(P)`.
    /// Returns `None` (`+\infty`) when an `S -> T` path contains zero eligible control points.
    pub fn weighted_minimum_vertex_cut(&self) -> Option<(u64, Vec<usize>)> {
        let n = self.edges.len();
        assert!(n >= 2 && self.costs.len() == n);
        if self.edges[0].contains(&(n - 1)) {
            return None;
        }
        let finite_sum: u64 = self.costs[1..n - 1].iter().flatten().copied().sum();
        let inf = (finite_sum as i64)
            .checked_add(1)
            .expect("capacity overflow");

        // Node splitting: in(v) = 2*v, out(v) = 2*v + 1; source = out(0) = 1, sink = in(n-1) = 2*(n-1)
        let mut dinic = Dinic::new(2 * n);
        for v in 1..n - 1 {
            let cap = match self.costs[v] {
                Some(c) => {
                    assert!(c > 0, "eligible vertex cost must be positive");
                    c as i64
                }
                None => inf,
            };
            dinic.add_edge(2 * v, 2 * v + 1, cap);
        }
        for u in 0..n - 1 {
            let u_out = if u == 0 { 1 } else { 2 * u + 1 };
            for &v in &self.edges[u] {
                let v_in = 2 * v;
                dinic.add_edge(u_out, v_in, inf);
            }
        }

        let source = 1usize;
        let sink = 2 * (n - 1);
        let flow = dinic.max_flow(source, sink, inf);
        if flow >= inf {
            return None;
        }
        let reach = dinic.reachable_from(source);
        let cut: Vec<usize> = (1..n - 1)
            .filter(|&v| reach[2 * v] && !reach[2 * v + 1])
            .collect();
        let cut_cost: u64 = cut
            .iter()
            .map(|&v| self.costs[v].expect("ineligible edge cannot be saturated"))
            .sum();
        assert_eq!(cut_cost, flow as u64);
        assert_eq!(self.count_paths(&cut), 0);
        Some((cut_cost, cut))
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct WeightedCutEval {
    pub internal_vertices: usize,
    pub path_count: u128,
    pub min_cut_cost: Option<u64>,
    pub is_infinite: bool,
    pub cut_vertex_count: Option<usize>,
    pub remaining_paths_after_cut: u128,
}

#[derive(Serialize, Debug, Clone)]
pub struct AppendixAScalingRow {
    pub extra_layers_k: usize,
    pub internal_vertices: usize,
    pub path_count: u128,
    pub a_unit: WeightedCutEval,
    pub b_unit: WeightedCutEval,
    pub a_priced_singleton_20: WeightedCutEval,
    pub a_ineligible_bypass: WeightedCutEval,
}

#[derive(Serialize, Debug, Clone)]
pub struct WeightedCutsSuite {
    pub hypothesis: &'static str,
    pub evidence_tier: &'static str,
    pub brute_force_small_dags_checked: usize,
    pub brute_force_discrepancies: usize,
    pub matched_vertices_and_paths_verified: bool,
    pub unit_vs_priced_flip_verified: bool,
    pub ungateable_bypass_infinite_verified: bool,
    pub scaling_rows: Vec<AppendixAScalingRow>,
    pub claim_boundary: &'static str,
}

fn eval_dag(dag: &WeightedDag) -> WeightedCutEval {
    let internal_vertices = dag.edges.len() - 2;
    let path_count = dag.count_paths(&[]);
    match dag.weighted_minimum_vertex_cut() {
        Some((cost, cut)) => {
            let rem = dag.count_paths(&cut);
            WeightedCutEval {
                internal_vertices,
                path_count,
                min_cut_cost: Some(cost),
                is_infinite: false,
                cut_vertex_count: Some(cut.len()),
                remaining_paths_after_cut: rem,
            }
        }
        None => WeightedCutEval {
            internal_vertices,
            path_count,
            min_cut_cost: None,
            is_infinite: true,
            cut_vertex_count: None,
            remaining_paths_after_cut: path_count,
        },
    }
}

pub fn weighted_cuts_suite() -> WeightedCutsSuite {
    // 1. Exhaustive brute-force verification on all 5-vertex topological DAGs (9 candidate edges = 512 masks)
    // across 4 cost/eligibility profiles (2,048 instances total).
    let candidate_edges = [
        (0usize, 1usize),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 3),
        (2, 4),
        (3, 4),
    ];
    let profiles: [[Option<u64>; 3]; 4] = [
        [Some(1), Some(1), Some(1)],
        [Some(7), Some(2), Some(3)],
        [None, Some(3), Some(4)],
        [None, None, Some(5)],
    ];

    let mut checked = 0usize;
    let mut discrepancies = 0usize;
    for prof in &profiles {
        for mask in 0..(1usize << candidate_edges.len()) {
            let mut edges = vec![Vec::new(); 5];
            for (bit, &(u, v)) in candidate_edges.iter().enumerate() {
                if (mask & (1 << bit)) != 0 {
                    edges[u].push(v);
                }
            }
            let costs = vec![None, prof[0], prof[1], prof[2], None];
            let dag = WeightedDag { edges, costs };

            let mut best_bf: Option<u64> = None;
            for sub in 0..8usize {
                let mut valid = true;
                let mut sum_c = 0u64;
                let mut cut_verts = Vec::new();
                for iv in 1..=3usize {
                    if (sub & (1 << (iv - 1))) != 0 {
                        if let Some(c) = dag.costs[iv] {
                            sum_c += c;
                            cut_verts.push(iv);
                        } else {
                            valid = false;
                            break;
                        }
                    }
                }
                if valid && dag.count_paths(&cut_verts) == 0 {
                    best_bf = Some(best_bf.map_or(sum_c, |cur| cur.min(sum_c)));
                }
            }

            let solver_cost = dag.weighted_minimum_vertex_cut().map(|(c, _)| c);
            if solver_cost != best_bf {
                discrepancies += 1;
            }
            checked += 1;
        }
    }
    assert_eq!(discrepancies, 0);

    // 2. Superintelligence Appendix A Scaling Families (k in [0, 2, 4, 6])
    let ks = [0usize, 2, 4, 6];
    let mut scaling_rows = Vec::with_capacity(ks.len());

    for &k in &ks {
        let mut a_widths = vec![1usize, 8, 8, 9, 10, 10];
        a_widths.extend(std::iter::repeat_n(16, k));
        let mut b_widths = vec![4usize, 4, 4, 4, 15, 15];
        b_widths.extend(std::iter::repeat_n(16, k));

        let dag_a_unit = WeightedDag::complete_layered(&a_widths, 1);
        let dag_b_unit = WeightedDag::complete_layered(&b_widths, 1);
        let dag_a_priced = WeightedDag::complete_layered(&a_widths, 20);

        // Add an ungateable (ineligible) 1-vertex bypass route S -> v_bypass -> T
        let mut dag_a_bypass = dag_a_unit.clone();
        let old_sink = dag_a_bypass.edges.len() - 1;
        let bypass_v = old_sink;
        let new_sink = old_sink + 1;
        for es in &mut dag_a_bypass.edges {
            for target in es {
                if *target == old_sink {
                    *target = new_sink;
                }
            }
        }
        dag_a_bypass.edges[0].push(bypass_v);
        dag_a_bypass.edges[bypass_v] = vec![new_sink];
        dag_a_bypass.edges.push(Vec::new());
        dag_a_bypass.costs[bypass_v] = None; // ineligible control point!
        dag_a_bypass.costs.push(None); // new sink T

        let a_unit = eval_dag(&dag_a_unit);
        let b_unit = eval_dag(&dag_b_unit);
        let a_priced = eval_dag(&dag_a_priced);
        let a_bypass = eval_dag(&dag_a_bypass);

        let prod_paths: u128 = a_widths.iter().map(|&w| w as u128).product();
        assert_eq!(a_unit.internal_vertices, b_unit.internal_vertices);
        assert_eq!(a_unit.path_count, prod_paths);
        assert_eq!(a_unit.path_count, b_unit.path_count);
        assert_eq!(a_unit.min_cut_cost, Some(1));
        assert_eq!(b_unit.min_cut_cost, Some(4));
        assert_eq!(a_priced.min_cut_cost, Some(8));
        assert!(a_bypass.is_infinite && a_bypass.min_cut_cost.is_none());

        scaling_rows.push(AppendixAScalingRow {
            extra_layers_k: k,
            internal_vertices: a_unit.internal_vertices,
            path_count: a_unit.path_count,
            a_unit,
            b_unit,
            a_priced_singleton_20: a_priced,
            a_ineligible_bypass: a_bypass,
        });
    }

    WeightedCutsSuite {
        hypothesis: "H10",
        evidence_tier: "tier2_exact_combinatorial_maxflow_verification",
        brute_force_small_dags_checked: checked,
        brute_force_discrepancies: discrepancies,
        matched_vertices_and_paths_verified: true,
        unit_vs_priced_flip_verified: true,
        ungateable_bypass_infinite_verified: true,
        scaling_rows,
        claim_boundary: "Exact weighted cutset verification on finite DAGs; does not certify completeness or cost accuracy of empirical deployment graphs.",
    }
}

// ============================================================================
// H11: Sparse ~8M-Pathway DAG Interdiction (`interdiction.json`)
//      Exact Rust port of `tips/experiments.py` Experiment 2
// ============================================================================

const STAGES: usize = 8;
const OUT_DEGREE: usize = 6;
const BETA: f64 = 0.5;
const MAX_BUDGET: usize = 32;
pub const BUDGET_GRID: [usize; 9] = [0, 3, 6, 9, 12, 16, 20, 24, 32];
pub const REPORT_COUNTS: [usize; 5] = [30, 100, 300, 1000, 3000];

struct SparseLayeredGraph {
    sizes: [usize; STAGES],
    /// `adj[stage][u]` holds `OUT_DEGREE` sorted distinct target indices in `0..sizes[stage + 1]`.
    adj: [Vec<[usize; OUT_DEGREE]>; STAGES - 1],
    costs: [Vec<f64>; STAGES],
}

fn sample_without_replacement_weighted(
    weights: &[f64],
    k: usize,
    rng: &mut Rng,
) -> [usize; OUT_DEGREE] {
    assert_eq!(k, OUT_DEGREE);
    let mut rem = weights.to_vec();
    let mut sum: f64 = rem.iter().sum();
    let mut out = [0usize; OUT_DEGREE];
    for slot in &mut out {
        let mut u = rng.uniform() * sum;
        let mut picked = rem.len() - 1;
        for (idx, &w) in rem.iter().enumerate() {
            if w <= 0.0 {
                continue;
            }
            u -= w;
            if u <= 0.0 {
                picked = idx;
                break;
            }
        }
        sum -= rem[picked];
        rem[picked] = 0.0;
        *slot = picked;
    }
    out.sort_unstable();
    out
}

fn sample_without_replacement_uniform(n: usize, rng: &mut Rng) -> [usize; OUT_DEGREE] {
    assert!(n >= OUT_DEGREE);
    let mut pool: Vec<usize> = (0..n).collect();
    for i in 0..OUT_DEGREE {
        let j = i + rng.index(n - i);
        pool.swap(i, j);
    }
    let mut out = [0usize; OUT_DEGREE];
    out.copy_from_slice(&pool[..OUT_DEGREE]);
    out.sort_unstable();
    out
}

fn make_sparse_graph(family: &str, rng: &mut Rng) -> SparseLayeredGraph {
    let mut sizes = if family == "narrow" {
        [30usize; STAGES]
    } else {
        [27usize; STAGES]
    };
    if family == "narrow" {
        let b_stage = 2 + rng.index(4); // {2, 3, 4, 5}
        sizes[b_stage] = 6;
    }

    let adj: [Vec<[usize; OUT_DEGREE]>; STAGES - 1] = std::array::from_fn(|i| {
        let a = sizes[i];
        let b = sizes[i + 1];
        if family == "hub" {
            let mut w: Vec<f64> = (1..=b).map(|r| 1.0 / (r as f64).powf(1.5)).collect();
            rng.shuffle(&mut w);
            let s: f64 = w.iter().sum();
            for x in &mut w {
                *x /= s;
            }
            (0..a)
                .map(|_| sample_without_replacement_weighted(&w, OUT_DEGREE, rng))
                .collect()
        } else {
            (0..a)
                .map(|_| sample_without_replacement_uniform(b, rng))
                .collect()
        }
    });

    let costs: [Vec<f64>; STAGES] =
        std::array::from_fn(|i| (0..sizes[i]).map(|_| 1.0 + 9.0 * rng.uniform()).collect());

    SparseLayeredGraph { sizes, adj, costs }
}

fn count_paths_sparse(g: &SparseLayeredGraph, alive: &[Vec<bool>; STAGES]) -> f64 {
    let mut c: Vec<f64> = alive[STAGES - 1]
        .iter()
        .map(|&ok| if ok { 1.0 } else { 0.0 })
        .collect();
    for i in (0..STAGES - 1).rev() {
        let mut prev = vec![0.0f64; g.sizes[i]];
        for u in 0..g.sizes[i] {
            if alive[i][u] {
                let mut sum = 0.0f64;
                for &v in &g.adj[i][u] {
                    sum += c[v];
                }
                prev[u] = sum;
            }
        }
        c = prev;
    }
    c.iter().sum()
}

fn cheapest_sparse(g: &SparseLayeredGraph, alive: &[Vec<bool>; STAGES]) -> f64 {
    const INF: f64 = 1e18;
    let mut c: Vec<f64> = (0..g.sizes[STAGES - 1])
        .map(|v| {
            if alive[STAGES - 1][v] {
                g.costs[STAGES - 1][v]
            } else {
                INF
            }
        })
        .collect();
    for i in (0..STAGES - 1).rev() {
        let mut prev = vec![INF; g.sizes[i]];
        for u in 0..g.sizes[i] {
            if alive[i][u] {
                let mut best_nxt = INF;
                for &v in &g.adj[i][u] {
                    if c[v] < best_nxt {
                        best_nxt = c[v];
                    }
                }
                if best_nxt < INF * 0.5 {
                    prev[u] = g.costs[i][u] + best_nxt;
                }
            }
        }
        c = prev;
    }
    let m = c.into_iter().fold(INF, f64::min);
    if m >= INF * 0.5 { f64::INFINITY } else { m }
}

fn sample_boltzmann_paths(
    g: &SparseLayeredGraph,
    n_paths: usize,
    rng: &mut Rng,
) -> Vec<[usize; STAGES]> {
    let mut z: [Vec<f64>; STAGES] = std::array::from_fn(|_| Vec::new());
    z[STAGES - 1] = g.costs[STAGES - 1]
        .iter()
        .map(|&c| (-BETA * c).exp())
        .collect();
    for i in (0..STAGES - 1).rev() {
        let mut zi = vec![0.0f64; g.sizes[i]];
        for (u, zi_u) in zi.iter_mut().enumerate() {
            let mut sum_nxt = 0.0f64;
            for &v in &g.adj[i][u] {
                sum_nxt += z[i + 1][v];
            }
            *zi_u = (-BETA * g.costs[i][u]).exp() * sum_nxt;
        }
        z[i] = zi;
    }

    // Build CDF for stage 0
    let z0_sum: f64 = z[0].iter().sum();
    let mut cdf0 = Vec::with_capacity(g.sizes[0]);
    let mut acc = 0.0f64;
    for &w in &z[0] {
        acc += w / z0_sum;
        cdf0.push(acc);
    }

    let mut paths = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut p = [0usize; STAGES];
        let u0 = rng.uniform();
        p[0] = cdf0.partition_point(|&c| c <= u0).min(g.sizes[0] - 1);
        for i in 0..STAGES - 1 {
            let cur = p[i];
            let nbrs = &g.adj[i][cur];
            let mut w_nbr = [0.0f64; OUT_DEGREE];
            let mut w_sum = 0.0f64;
            for (k, &v) in nbrs.iter().enumerate() {
                let w = z[i + 1][v];
                w_nbr[k] = w;
                w_sum += w;
            }
            let mut draw = rng.uniform() * w_sum;
            let mut chosen = nbrs[OUT_DEGREE - 1];
            for (k, &w) in w_nbr.iter().enumerate() {
                draw -= w;
                if draw <= 0.0 {
                    chosen = nbrs[k];
                    break;
                }
            }
            p[i + 1] = chosen;
        }
        paths.push(p);
    }
    paths
}

/// Minimum `SRC -> SNK` node cut on a layered subgraph via split-vertex Dinic.
fn layered_node_cut(
    sizes: &[usize; STAGES],
    stage_edges: &[Vec<Vec<usize>>; STAGES - 1],
    src_mask: &[bool],
    snk_mask: &[bool],
) -> Vec<(usize, usize)> {
    let mut offsets = [0usize; STAGES + 1];
    for i in 0..STAGES {
        offsets[i + 1] = offsets[i] + sizes[i];
    }
    let total_verts = offsets[STAGES];
    let src = 2 * total_verts;
    let snk = 2 * total_verts + 1;
    let inf = (total_verts as i64) + 5;

    let mut dinic = Dinic::new(2 * total_verts + 2);
    for stage in 0..STAGES {
        for v in 0..sizes[stage] {
            let id = offsets[stage] + v;
            dinic.add_edge(2 * id, 2 * id + 1, 1);
        }
    }
    for (v, &present) in src_mask.iter().enumerate() {
        if present {
            let id = offsets[0] + v;
            dinic.add_edge(src, 2 * id, inf);
        }
    }
    for (v, &present) in snk_mask.iter().enumerate() {
        if present {
            let id = offsets[STAGES - 1] + v;
            dinic.add_edge(2 * id + 1, snk, inf);
        }
    }
    for stage in 0..STAGES - 1 {
        for (u, nbrs) in stage_edges[stage].iter().enumerate() {
            let uid = offsets[stage] + u;
            for &v in nbrs {
                let vid = offsets[stage + 1] + v;
                dinic.add_edge(2 * uid + 1, 2 * vid, inf);
            }
        }
    }

    let flow = dinic.max_flow(src, snk, inf);
    let reach = dinic.reachable_from(src);
    let mut cut = Vec::with_capacity(flow as usize);
    for stage in 0..STAGES {
        for v in 0..sizes[stage] {
            let id = offsets[stage] + v;
            if reach[2 * id] && !reach[2 * id + 1] {
                cut.push((stage, v));
            }
        }
    }
    assert_eq!(cut.len(), flow as usize);
    cut
}

fn full_graph_cut(g: &SparseLayeredGraph) -> Vec<(usize, usize)> {
    let stage_edges: [Vec<Vec<usize>>; STAGES - 1] =
        std::array::from_fn(|i| g.adj[i].iter().map(|nbrs| nbrs.to_vec()).collect());
    let src_mask = vec![true; g.sizes[0]];
    let snk_mask = vec![true; g.sizes[STAGES - 1]];
    layered_node_cut(&g.sizes, &stage_edges, &src_mask, &snk_mask)
}

fn reported_graph_cut(sizes: &[usize; STAGES], paths: &[[usize; STAGES]]) -> Vec<(usize, usize)> {
    let mut stage_edges: [Vec<Vec<usize>>; STAGES - 1] =
        std::array::from_fn(|i| vec![Vec::new(); sizes[i]]);
    let mut src_mask = vec![false; sizes[0]];
    let mut snk_mask = vec![false; sizes[STAGES - 1]];

    for p in paths {
        src_mask[p[0]] = true;
        snk_mask[p[STAGES - 1]] = true;
        for i in 0..STAGES - 1 {
            stage_edges[i][p[i]].push(p[i + 1]);
        }
    }
    for st in &mut stage_edges {
        for es in st {
            es.sort_unstable();
            es.dedup();
        }
    }
    layered_node_cut(sizes, &stage_edges, &src_mask, &snk_mask)
}

fn evaluate_interdiction_order(
    order: &[(usize, usize)],
    g: &SparseLayeredGraph,
    total_paths: f64,
    base_cost: f64,
) -> (
    [f64; MAX_BUDGET + 1],
    [f64; MAX_BUDGET + 1],
    [f64; MAX_BUDGET + 1],
) {
    let mut alive: [Vec<bool>; STAGES] = std::array::from_fn(|i| vec![true; g.sizes[i]]);
    let mut blocked = [0.0f64; MAX_BUDGET + 1];
    let mut severed = [0.0f64; MAX_BUDGET + 1];
    let mut gain = [1.0f64; MAX_BUDGET + 1];

    for b in 1..=MAX_BUDGET {
        if b <= order.len() {
            let (st, v) = order[b - 1];
            alive[st][v] = false;
        }
        let left = count_paths_sparse(g, &alive);
        blocked[b] = 1.0 - left / total_paths;
        severed[b] = if left == 0.0 { 1.0 } else { 0.0 };
        gain[b] = cheapest_sparse(g, &alive) / base_cost;
    }
    (blocked, severed, gain)
}

fn median_f64(xs: &mut [f64]) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

#[derive(Serialize, Debug, Clone)]
pub struct InterdictionStrategySeries {
    pub blocked_mean: Vec<f64>,
    pub severed_share: Vec<f64>,
    pub gain_median: Vec<Option<f64>>,
    pub gain_capped_2x_median: Vec<f64>,
}

#[derive(Serialize, Debug, Clone)]
pub struct InterdictionBudgetGridRow {
    pub budget: usize,
    pub cut_blocked_mean: f64,
    pub cut_severed_share: f64,
    pub cut_gain_median: Option<f64>,
    pub patch_specific_blocked_mean: f64,
    pub patch_specific_severed_share: f64,
    pub patch_specific_gain_median: Option<f64>,
    pub patch_random_blocked_mean: f64,
    pub patch_random_severed_share: f64,
    pub patch_random_gain_median: Option<f64>,
}

#[derive(Serialize, Debug, Clone)]
pub struct InterdictionFamilyResult {
    pub paths_millions: f64,
    pub width_median: f64,
    pub reported_cut_median: f64,
    pub seen_share: f64,
    pub sever_by_n: BTreeMap<String, f64>,
    pub cut: InterdictionStrategySeries,
    pub patch_specific: InterdictionStrategySeries,
    pub patch_random: InterdictionStrategySeries,
    pub grid_rows: Vec<InterdictionBudgetGridRow>,
}

#[derive(Serialize, Debug, Clone)]
pub struct InterdictionSuite {
    pub hypothesis: &'static str,
    pub evidence_tier: &'static str,
    pub n_graphs_per_family: usize,
    pub stages: usize,
    pub out_degree: usize,
    pub boltzmann_beta: f64,
    pub default_reports: usize,
    pub max_budget: usize,
    pub budget_grid: Vec<usize>,
    pub report_counts: Vec<usize>,
    pub narrow: InterdictionFamilyResult,
    pub hub: InterdictionFamilyResult,
    pub wide: InterdictionFamilyResult,
    pub claim_boundary: &'static str,
}

fn summarize_strategy(
    blocked_runs: &[[f64; MAX_BUDGET + 1]],
    severed_runs: &[[f64; MAX_BUDGET + 1]],
    gain_runs: &[[f64; MAX_BUDGET + 1]],
) -> InterdictionStrategySeries {
    let n = blocked_runs.len() as f64;
    let mut blocked_mean = Vec::with_capacity(MAX_BUDGET + 1);
    let mut severed_share = Vec::with_capacity(MAX_BUDGET + 1);
    let mut gain_median = Vec::with_capacity(MAX_BUDGET + 1);
    let mut gain_capped_2x_median = Vec::with_capacity(MAX_BUDGET + 1);

    let clean_zero = |x: f64| if x == 0.0 { 0.0 } else { x };
    for b in 0..=MAX_BUDGET {
        blocked_mean.push(clean_zero(
            blocked_runs.iter().map(|r| r[b]).sum::<f64>() / n,
        ));
        severed_share.push(clean_zero(
            severed_runs.iter().map(|r| r[b]).sum::<f64>() / n,
        ));
        let mut col: Vec<f64> = gain_runs.iter().map(|r| r[b]).collect();
        let med = median_f64(&mut col);
        gain_median.push(if med.is_infinite() { None } else { Some(med) });
        let mut capped: Vec<f64> = gain_runs
            .iter()
            .map(|r| if r[b].is_infinite() { 2.0 } else { r[b] })
            .collect();
        gain_capped_2x_median.push(median_f64(&mut capped));
    }

    InterdictionStrategySeries {
        blocked_mean,
        severed_share,
        gain_median,
        gain_capped_2x_median,
    }
}

fn run_interdiction_family(
    family: &str,
    n_graphs: usize,
    rng: &mut Rng,
) -> InterdictionFamilyResult {
    let mut counts = Vec::with_capacity(n_graphs);
    let mut widths = Vec::with_capacity(n_graphs);
    let mut rep_cut_sizes = Vec::with_capacity(n_graphs);
    let mut seen_shares = Vec::with_capacity(n_graphs);

    let mut cut_b = Vec::with_capacity(n_graphs);
    let mut cut_s = Vec::with_capacity(n_graphs);
    let mut cut_g = Vec::with_capacity(n_graphs);

    let mut pspec_b = Vec::with_capacity(n_graphs);
    let mut pspec_s = Vec::with_capacity(n_graphs);
    let mut pspec_g = Vec::with_capacity(n_graphs);

    let mut prand_b = Vec::with_capacity(n_graphs);
    let mut prand_s = Vec::with_capacity(n_graphs);
    let mut prand_g = Vec::with_capacity(n_graphs);

    let mut sever_acc: [usize; REPORT_COUNTS.len()] = [0; REPORT_COUNTS.len()];

    for _ in 0..n_graphs {
        let g = make_sparse_graph(family, rng);
        let alive_all: [Vec<bool>; STAGES] = std::array::from_fn(|i| vec![true; g.sizes[i]]);
        let total = count_paths_sparse(&g, &alive_all);
        let base = cheapest_sparse(&g, &alive_all);
        counts.push(total);
        widths.push(full_graph_cut(&g).len() as f64);

        let paths = sample_boltzmann_paths(&g, 300, rng);
        let mut uniq = paths.clone();
        uniq.sort_unstable();
        uniq.dedup();
        seen_shares.push(uniq.len() as f64 / total);

        let mut use_cnt: [Vec<usize>; STAGES] = std::array::from_fn(|i| vec![0usize; g.sizes[i]]);
        for p in &paths {
            for (i, &v) in p.iter().enumerate() {
                use_cnt[i][v] += 1;
            }
        }

        // 1. Cut-guided: minimum node cut of reported subgraph, busiest vertex first
        let mut cut = reported_graph_cut(&g.sizes, &paths);
        cut.sort_by(|&(i1, v1), &(i2, v2)| {
            use_cnt[i2][v2]
                .cmp(&use_cnt[i1][v1])
                .then_with(|| (i1, v1).cmp(&(i2, v2)))
        });
        rep_cut_sizes.push(cut.len() as f64);
        let (b, s, gn) = evaluate_interdiction_order(&cut, &g, total, base);
        cut_b.push(b);
        cut_s.push(s);
        cut_g.push(gn);

        // 2 & 3. Patching: sort unique reported paths by path cost ascending
        let mut order_paths: Vec<(f64, [usize; STAGES])> = uniq
            .into_iter()
            .map(|p| {
                let pc: f64 = (0..STAGES).map(|i| g.costs[i][p[i]]).sum();
                (pc, p)
            })
            .collect();
        order_paths.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        for is_specific in [true, false] {
            let mut removed: [Vec<bool>; STAGES] = std::array::from_fn(|i| vec![false; g.sizes[i]]);
            let mut order = Vec::with_capacity(MAX_BUDGET);
            for (_, p) in &order_paths {
                if order.len() >= MAX_BUDGET {
                    break;
                }
                if (0..STAGES).any(|i| removed[i][p[i]]) {
                    continue;
                }
                let chosen = if is_specific {
                    (0..STAGES)
                        .map(|i| (i, p[i]))
                        .min_by_key(|&(i, v)| (use_cnt[i][v], i, v))
                        .unwrap()
                } else {
                    let st = rng.index(STAGES);
                    (st, p[st])
                };
                removed[chosen.0][chosen.1] = true;
                order.push(chosen);
            }
            let (b, s, gn) = evaluate_interdiction_order(&order, &g, total, base);
            if is_specific {
                pspec_b.push(b);
                pspec_s.push(s);
                pspec_g.push(gn);
            } else {
                prand_b.push(b);
                prand_s.push(s);
                prand_g.push(gn);
            }
        }

        // 4. Severance of full graph by reported-graph min-cut across report sample sizes
        for (idx, &n_rep) in REPORT_COUNTS.iter().enumerate() {
            let pn = sample_boltzmann_paths(&g, n_rep, rng);
            let mut al: [Vec<bool>; STAGES] = std::array::from_fn(|i| vec![true; g.sizes[i]]);
            for (st, v) in reported_graph_cut(&g.sizes, &pn) {
                al[st][v] = false;
            }
            if count_paths_sparse(&g, &al) == 0.0 {
                sever_acc[idx] += 1;
            }
        }
    }

    let cut_series = summarize_strategy(&cut_b, &cut_s, &cut_g);
    let pspec_series = summarize_strategy(&pspec_b, &pspec_s, &pspec_g);
    let prand_series = summarize_strategy(&prand_b, &prand_s, &prand_g);

    let grid_rows: Vec<InterdictionBudgetGridRow> = BUDGET_GRID
        .iter()
        .map(|&b| InterdictionBudgetGridRow {
            budget: b,
            cut_blocked_mean: cut_series.blocked_mean[b],
            cut_severed_share: cut_series.severed_share[b],
            cut_gain_median: cut_series.gain_median[b],
            patch_specific_blocked_mean: pspec_series.blocked_mean[b],
            patch_specific_severed_share: pspec_series.severed_share[b],
            patch_specific_gain_median: pspec_series.gain_median[b],
            patch_random_blocked_mean: prand_series.blocked_mean[b],
            patch_random_severed_share: prand_series.severed_share[b],
            patch_random_gain_median: prand_series.gain_median[b],
        })
        .collect();

    let mut sever_by_n = BTreeMap::new();
    for (idx, &n_rep) in REPORT_COUNTS.iter().enumerate() {
        sever_by_n.insert(n_rep.to_string(), sever_acc[idx] as f64 / n_graphs as f64);
    }

    InterdictionFamilyResult {
        paths_millions: (counts.iter().sum::<f64>() / n_graphs as f64) / 1e6,
        width_median: median_f64(&mut widths),
        reported_cut_median: median_f64(&mut rep_cut_sizes),
        seen_share: seen_shares.iter().sum::<f64>() / n_graphs as f64,
        sever_by_n,
        cut: cut_series,
        patch_specific: pspec_series,
        patch_random: prand_series,
        grid_rows,
    }
}

pub fn interdiction_suite(seed: u64, quick: bool) -> InterdictionSuite {
    let n_graphs = if quick { 20 } else { 200 };
    let mut rng = Rng(seed ^ 0x4831_315F_4441_4753);

    let narrow = run_interdiction_family("narrow", n_graphs, &mut rng);
    let hub = run_interdiction_family("hub", n_graphs, &mut rng);
    let wide = run_interdiction_family("wide", n_graphs, &mut rng);

    InterdictionSuite {
        hypothesis: "H11",
        evidence_tier: "tier4_synthetic_sparse_dag_boltzmann_interdiction",
        n_graphs_per_family: n_graphs,
        stages: STAGES,
        out_degree: OUT_DEGREE,
        boltzmann_beta: BETA,
        default_reports: 300,
        max_budget: MAX_BUDGET,
        budget_grid: BUDGET_GRID.to_vec(),
        report_counts: REPORT_COUNTS.to_vec(),
        narrow,
        hub,
        wide,
        claim_boundary: "Controlled synthetic 8-stage sparse DAGs under Boltzmann low-cost path sampling; illustrates structural width vs. route-specific patching, not empirical cyber vulnerability rates.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h10_weighted_cuts_appendix_a_and_brute_force() {
        let res = weighted_cuts_suite();
        assert_eq!(res.brute_force_small_dags_checked, 2_048);
        assert_eq!(res.brute_force_discrepancies, 0);
        assert!(res.matched_vertices_and_paths_verified);
        assert!(res.unit_vs_priced_flip_verified);
        assert!(res.ungateable_bypass_infinite_verified);

        let expected_verts = [46usize, 78, 110, 142];
        let expected_paths = [57_600u128, 14_745_600, 3_774_873_600, 966_367_641_600];
        for (row, (&v, &p)) in res
            .scaling_rows
            .iter()
            .zip(expected_verts.iter().zip(expected_paths.iter()))
        {
            assert_eq!(row.internal_vertices, v);
            assert_eq!(row.path_count, p);
            assert_eq!(row.a_unit.min_cut_cost, Some(1));
            assert_eq!(row.b_unit.min_cut_cost, Some(4));
            assert_eq!(row.a_priced_singleton_20.min_cut_cost, Some(8));
            assert!(row.a_ineligible_bypass.is_infinite);
        }
    }

    #[test]
    fn h11_interdiction_narrow_width_vs_specific_patching() {
        let res = interdiction_suite(20261110, true);
        assert!((res.narrow.paths_millions - 8.398_08).abs() < 1e-6);
        assert!((res.hub.paths_millions - 7.558_272).abs() < 1e-6);
        assert!((res.wide.paths_millions - 7.558_272).abs() < 1e-6);

        assert_eq!(res.narrow.width_median, 6.0);
        assert_eq!(res.wide.width_median, 27.0);
        assert!(res.narrow.cut.severed_share[6] >= 0.45);
        assert!(res.narrow.cut.blocked_mean[6] >= 0.88);
        assert_eq!(res.narrow.patch_specific.severed_share[32], 0.0);
        assert!(res.narrow.patch_specific.blocked_mean[32] > 0.60);
        assert!(*res.narrow.sever_by_n.get("3000").unwrap() >= 0.95);
    }

    #[test]
    fn h11_publication_profile_consistency_and_determinism() {
        let t_quick = std::time::Instant::now();
        let _q = interdiction_suite(20261110, true);
        let quick_s = t_quick.elapsed().as_secs_f64();

        let t_pub = std::time::Instant::now();
        let w = weighted_cuts_suite();
        let pub1 = interdiction_suite(20261110, false);
        let pub_s = t_pub.elapsed().as_secs_f64();

        let j1 = serde_json::to_string(&pub1).unwrap();
        let j2 = serde_json::to_string(&interdiction_suite(20261110, false)).unwrap();
        assert_eq!(j1, j2);
        assert_eq!(w.scaling_rows.len(), 4);

        // Publication 200-graph checks matching tips/experiments.py Exp 2 laws:
        assert_eq!(pub1.narrow.width_median, 6.0);
        assert!(pub1.hub.width_median >= 19.0 && pub1.hub.width_median <= 23.0);
        assert_eq!(pub1.wide.width_median, 27.0);
        assert_eq!(pub1.narrow.reported_cut_median, 6.0);
        assert!(
            pub1.narrow.cut.severed_share[6] >= 0.55 && pub1.narrow.cut.severed_share[6] <= 0.80
        );
        assert!(pub1.narrow.cut.blocked_mean[6] >= 0.90);
        assert_eq!(pub1.narrow.patch_specific.severed_share[32], 0.0);
        assert!(pub1.narrow.patch_specific.blocked_mean[32] > 0.70);
        assert!(*pub1.narrow.sever_by_n.get("30").unwrap() <= 0.10);
        assert!(*pub1.narrow.sever_by_n.get("1000").unwrap() >= 0.93);
        assert!(*pub1.narrow.sever_by_n.get("3000").unwrap() >= 0.99);
        assert!(*pub1.wide.sever_by_n.get("3000").unwrap() <= 0.10);

        eprintln!(
            "[v3_graphs] quick={:.4}s, pub={:.4}s | narrow: cut6_sev={:.3}, cut6_blk={:.3}, spec32_sev={:.3}, spec32_blk={:.3}, spec32_gain={:?}, sever_by_n={:?} | hub: width={}, rep_cut={}, cut12_blk={:.3} | wide: width={}, rep_cut={}",
            quick_s,
            pub_s,
            pub1.narrow.cut.severed_share[6],
            pub1.narrow.cut.blocked_mean[6],
            pub1.narrow.patch_specific.severed_share[32],
            pub1.narrow.patch_specific.blocked_mean[32],
            pub1.narrow.patch_specific.gain_median[32],
            pub1.narrow.sever_by_n,
            pub1.hub.width_median,
            pub1.hub.reported_cut_median,
            pub1.hub.cut.blocked_mean[12],
            pub1.wide.width_median,
            pub1.wide.reported_cut_median,
        );
    }
}

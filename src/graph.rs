//! Exact harmless DAG examples; full-graph truth is separate from reported edges.
use serde::Serialize;
use std::collections::VecDeque;

#[derive(Clone)]
pub struct Graph {
    pub edges: Vec<Vec<usize>>,
}
impl Graph {
    pub fn layered(widths: &[usize]) -> Self {
        assert!(!widths.is_empty() && widths.iter().all(|w| *w > 0));
        let mut stages = vec![vec![0]];
        let mut next = 1;
        for &w in widths {
            stages.push((next..next + w).collect());
            next += w;
        }
        stages.push(vec![next]);
        let mut edges = vec![vec![]; next + 1];
        for pair in stages.windows(2) {
            for &a in &pair[0] {
                edges[a] = pair[1].clone();
            }
        }
        Self { edges }
    }
    pub fn paths(&self, removed: &[usize]) -> u128 {
        let n = self.edges.len();
        let mut count = vec![0u128; n];
        count[0] = 1;
        for a in 0..n {
            if removed.contains(&a) {
                count[a] = 0;
                continue;
            }
            for &b in &self.edges[a] {
                assert!(b > a, "path counter requires topological vertex order");
                count[b] = count[b].checked_add(count[a]).expect("path count overflow");
            }
        }
        count[n - 1]
    }
    pub fn minimum_vertex_cut(&self) -> Vec<usize> {
        let n = self.edges.len();
        assert!(
            n >= 2 && !self.edges[0].contains(&(n - 1)),
            "internal vertex cuts require nonadjacent endpoints"
        );
        let inf = n as i64 + 1;
        let mut cap = vec![vec![0i64; 2 * n]; 2 * n];
        for a in 0..n {
            cap[2 * a][2 * a + 1] = if a == 0 || a == n - 1 { inf } else { 1 };
            for &b in &self.edges[a] {
                cap[2 * a + 1][2 * b] = inf;
            }
        }
        let source = 1;
        let sink = 2 * (n - 1);
        loop {
            let mut parent = vec![usize::MAX; 2 * n];
            parent[source] = source;
            let mut q = VecDeque::from([source]);
            while let Some(a) = q.pop_front() {
                for b in 0..2 * n {
                    if cap[a][b] > 0 && parent[b] == usize::MAX {
                        parent[b] = a;
                        q.push_back(b);
                    }
                }
            }
            if parent[sink] == usize::MAX {
                return (1..n - 1)
                    .filter(|a| parent[2 * a] != usize::MAX && parent[2 * a + 1] == usize::MAX)
                    .collect();
            }
            let mut b = sink;
            let mut flow = inf;
            while b != source {
                let a = parent[b];
                flow = flow.min(cap[a][b]);
                b = a;
            }
            b = sink;
            while b != source {
                let a = parent[b];
                cap[a][b] -= flow;
                cap[b][a] += flow;
                b = a;
            }
        }
    }
}
#[derive(Serialize)]
pub struct GraphResult {
    pub widths: Vec<usize>,
    pub paths: u128,
    pub cut: Vec<usize>,
    pub remaining_paths: u128,
    pub remaining_with_unreported_bypass: u128,
}
pub fn suite() -> Vec<GraphResult> {
    [vec![4; 8], vec![2, 8, 4, 4, 4, 4, 4, 4]]
        .into_iter()
        .map(|widths| {
            let g = Graph::layered(&widths);
            let cut = g.minimum_vertex_cut();
            let remaining_paths = g.paths(&cut);
            // Add a previously unreported one-vertex route. Renumber the sink so
            // that the independent path counter still receives a topological DAG.
            let old_sink = g.edges.len() - 1;
            let mut full = g.clone();
            for es in &mut full.edges {
                for b in es {
                    if *b == old_sink {
                        *b += 1;
                    }
                }
            }
            full.edges[0].push(old_sink);
            full.edges[old_sink] = vec![old_sink + 1];
            full.edges.push(vec![]);
            GraphResult {
                paths: g.paths(&[]),
                remaining_paths,
                remaining_with_unreported_bypass: full.paths(&cut),
                widths,
                cut,
            }
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cut_agrees_with_brute_force_on_small_graphs() {
        // Every subset of edges in a 4-vertex DAG without a direct s--t edge.
        let choices = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3)];
        for mask in 0..32 {
            let mut g = Graph {
                edges: vec![vec![]; 4],
            };
            for (i, &(a, b)) in choices.iter().enumerate() {
                if mask & (1 << i) != 0 {
                    g.edges[a].push(b);
                }
            }
            let minimum = (0..4)
                .filter_map(|m| {
                    let cut: Vec<_> = (1..3).filter(|i| m & (1 << (i - 1)) != 0).collect();
                    (g.paths(&cut) == 0).then_some(cut.len())
                })
                .min()
                .unwrap();
            let cut = g.minimum_vertex_cut();
            assert_eq!(cut.len(), minimum);
            assert_eq!(g.paths(&cut), 0);
        }
        let rows = suite();
        assert_eq!(rows[0].paths, 65_536);
        assert_eq!(rows[1].paths, 65_536);
        assert_eq!(rows[0].cut.len(), 4);
        assert_eq!(rows[1].cut.len(), 2);
        assert!(
            rows.iter()
                .all(|r| r.remaining_paths == 0 && r.remaining_with_unreported_bypass == 1)
        );
    }
}

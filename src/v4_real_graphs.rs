//! Real Multi-Tool Agent & Linux Subprocess Confinement Attack-Graph Cutsets (Wave 3 V4).
//!
//! Bridges Section 6's combinatorial width & weighted-cutset theory (`H10`, `H11`, `C2`, `P1`)
//! from synthetic ensembles to 6 concrete operational graphs:
//! - 4 upstream AgentDojo `v1.2.2` multi-tool suites (`banking`, `workspace`, `travel`, `slack`)
//!   extracted via live canary execution and Pydantic schema / subsystem dependency introspection.
//! - 2 Linux subprocess IPC/syscall confinement graphs (`linux_unconfined_posix_ipc` vs.
//!   `linux_landlock_seccomp_broker_waist`) derived directly from `src/isolation_v2.rs` and
//!   `src/protocol_v2.rs` kernel/framing probes.
//!
//! Claim boundary (`AGENTS.md`): Exact combinatorial min-cut / path-count / interdiction proofs on
//! operational AgentDojo `v1.2.2` tool graphs and Linux subprocess confinement topologies; open
//! prose persuasion channels without external tool calls (e.g., `travel` `injection_task_6`) lie
//! outside synchronous tool-call mediation (`\tau_b = +\infty` unless channel-gated).

use crate::{rng::Rng, v3_graphs::WeightedDag};
use serde::Serialize;
use std::collections::BTreeSet;

pub const REAL_BUDGET_GRID: [usize; 10] = [0, 1, 2, 3, 4, 5, 8, 12, 16, 32];
const BOLTZMANN_BETA: f64 = 0.35;
const DEFAULT_OBSERVED_REPORTS: usize = 200;

#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct GraphVertexSpec {
    pub index: usize,
    pub label: String,
    pub stage: &'static str,
    pub cost: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct OperationalGraph {
    pub graph_id: &'static str,
    pub domain: &'static str,
    pub source_artifact: &'static str,
    pub vertices: Vec<GraphVertexSpec>,
    pub dag: WeightedDag,
    pub sink_indices: Vec<usize>,
    pub notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealInterdictionBudgetPoint {
    pub budget: usize,
    pub cutset_gate_blocked_fraction: f64,
    pub cutset_gate_severed_share: f64,
    pub patch_specific_step_blocked_fraction: f64,
    pub patch_specific_step_severed_share: f64,
    pub patch_route_signature_blocked_fraction: f64,
    pub patch_route_signature_severed_share: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealOperationalGraphReport {
    pub graph_id: &'static str,
    pub domain: &'static str,
    pub source_artifact: &'static str,
    pub total_vertices_v: usize,
    pub internal_vertices: usize,
    pub total_edges_e: usize,
    pub active_edges_e: usize,
    pub attack_route_count_p: u128,
    pub menger_pathway_width_w: usize,
    pub unweighted_min_cut_vertices: Vec<String>,
    pub sink_waist_width: usize,
    pub shortest_path_internal_len_l: usize,
    pub longest_path_internal_len: usize,
    pub weighted_min_cut_cost_tau_b: Option<u64>,
    pub weighted_min_cut_is_infinite: bool,
    pub weighted_min_cut_size: Option<usize>,
    pub weighted_min_cut_vertices: Vec<String>,
    pub remaining_paths_after_weighted_cut: u128,
    pub cutset_severs_all_routes: bool,
    pub width_to_routes_ratio: f64,
    pub interdiction_curve: Vec<RealInterdictionBudgetPoint>,
    pub vertices: Vec<GraphVertexSpec>,
    pub edges: Vec<Vec<usize>>,
    pub notes: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct RealToolCutsetsSuite {
    pub schema_version: u32,
    pub prediction_tested: &'static str,
    pub condition_tested: &'static str,
    pub evidence_tier: &'static str,
    pub seed: u64,
    pub discovery_trials_per_graph: usize,
    pub observed_reports_per_trial: usize,
    pub boltzmann_beta: f64,
    pub budget_grid: Vec<usize>,
    pub agentdojo_p1_verified: bool,
    pub linux_waist_confinement_verified: bool,
    pub graphs: Vec<RealOperationalGraphReport>,
    pub claim_boundary: &'static str,
}

struct ReconToolMeta {
    name: &'static str,
    subsystems: &'static [&'static str],
    is_discovery: bool,
}

struct AgentDojoSuiteSpec {
    graph_id: &'static str,
    source_artifact: &'static str,
    vectors: &'static [&'static str],
    canary_reach: &'static [(&'static str, &'static [&'static str])],
    recon_tools: &'static [ReconToolMeta],
    stage_tools: &'static [&'static str],
    sink_tools: &'static [&'static str],
    notes: &'static str,
}

fn build_agentdojo_graph(spec: &AgentDojoSuiteSpec) -> OperationalGraph {
    let mut vertices = Vec::new();
    let mut push_v = |label: String, stage: &'static str, cost: Option<u64>| -> usize {
        let index = vertices.len();
        vertices.push(GraphVertexSpec {
            index,
            label,
            stage,
            cost,
        });
        index
    };

    let s_idx = push_v("S".to_string(), "source", None);
    let l1_indices: Vec<usize> = spec
        .vectors
        .iter()
        .map(|v| push_v(format!("vec:{v}"), "untrusted_vector", None))
        .collect();
    let l2_indices: Vec<usize> = spec
        .canary_reach
        .iter()
        .map(|(fn_name, _)| push_v(format!("ingest:{fn_name}"), "untrusted_read_tool", Some(10)))
        .collect();
    let l3_indices: Vec<usize> = spec
        .recon_tools
        .iter()
        .map(|m| push_v(format!("recon1:{}", m.name), "primary_recon_tool", Some(8)))
        .collect();
    let l4_recon_indices: Vec<usize> = spec
        .recon_tools
        .iter()
        .map(|m| {
            push_v(
                format!("recon2:{}", m.name),
                "secondary_recon_tool",
                Some(8),
            )
        })
        .collect();
    let l4_stage_indices: Vec<usize> = spec
        .stage_tools
        .iter()
        .map(|fn_name| push_v(format!("stage:{fn_name}"), "state_staging_tool", Some(4)))
        .collect();
    let synth_names = [
        "synth:direct_injection_args",
        "synth:harvested_pii_interpolation",
        "synth:staged_or_split_args",
    ];
    let l4_synth_indices: Vec<usize> = synth_names
        .iter()
        .map(|s| push_v((*s).to_string(), "llm_arg_synthesis", None))
        .collect();
    let l5_indices: Vec<usize> = spec
        .sink_tools
        .iter()
        .map(|fn_name| push_v(format!("sink:{fn_name}"), "external_commit_sink", Some(1)))
        .collect();
    let t_idx = push_v("T".to_string(), "terminal", None);

    let mut edges = vec![Vec::new(); vertices.len()];

    // S -> L1 (untrusted environment injection records)
    for &v_idx in &l1_indices {
        edges[s_idx].push(v_idx);
    }

    // L1 -> L2 (empirically verified canary -> read tool edges)
    for (i2, &(_, reachable_vecs)) in spec.canary_reach.iter().enumerate() {
        let target = l2_indices[i2];
        for &rv in reachable_vecs {
            let pos = spec
                .vectors
                .binary_search(&rv)
                .expect("canary vector must exist in sorted vectors");
            edges[l1_indices[pos]].push(target);
        }
    }

    // L2 -> L3 (distinct follow-up reconnaissance read tool) and L2 -> direct synthesis
    let direct_synth = l4_synth_indices[0];
    for (i2, &(fn2, _)) in spec.canary_reach.iter().enumerate() {
        let u = l2_indices[i2];
        for (i3, m3) in spec.recon_tools.iter().enumerate() {
            if m3.name != fn2 {
                edges[u].push(l3_indices[i3]);
            }
        }
        edges[u].push(direct_synth);
    }

    // L3 -> L4_recon (cross-subsystem pivot or discovery-to-detail pivot), L4_stage, and L4_synth
    for (i3, m3) in spec.recon_tools.iter().enumerate() {
        let u = l3_indices[i3];
        for (i4, m4) in spec.recon_tools.iter().enumerate() {
            if m4.name != m3.name
                && (m3.subsystems != m4.subsystems || (m3.is_discovery && !m4.is_discovery))
            {
                edges[u].push(l4_recon_indices[i4]);
            }
        }
        for &st_idx in &l4_stage_indices {
            edges[u].push(st_idx);
        }
        for &sy_idx in &l4_synth_indices {
            edges[u].push(sy_idx);
        }
    }

    // L4 -> L5 (structured external commit / exfiltration sinks) and L5 -> T
    for &u in l4_recon_indices
        .iter()
        .chain(l4_stage_indices.iter())
        .chain(l4_synth_indices.iter())
    {
        for &sk in &l5_indices {
            edges[u].push(sk);
        }
    }
    for &sk in &l5_indices {
        edges[sk].push(t_idx);
    }

    for adj in &mut edges {
        adj.sort_unstable();
        adj.dedup();
    }
    let costs = vertices.iter().map(|v| v.cost).collect();

    OperationalGraph {
        graph_id: spec.graph_id,
        domain: "agentdojo_v1_2_2",
        source_artifact: spec.source_artifact,
        vertices,
        dag: WeightedDag { edges, costs },
        sink_indices: l5_indices,
        notes: spec.notes,
    }
}

fn build_linux_ipc_graph(confined: bool) -> OperationalGraph {
    let mut vertices = Vec::new();
    let mut push_v = |label: &str, stage: &'static str, cost: Option<u64>| -> usize {
        let index = vertices.len();
        vertices.push(GraphVertexSpec {
            index,
            label: label.to_string(),
            stage,
            cost,
        });
        index
    };

    let s_idx = push_v("S", "source", None);

    // Stage 1: Untrusted proposer subprocess execution primitives (6)
    let p_stdio = push_v("exec:stdio_json_frame", "subprocess_primitive", None);
    let p_vfs = push_v("exec:vfs_open_rw_link", "subprocess_primitive", None);
    let p_proc = push_v("exec:procfs_and_prlimit", "subprocess_primitive", None);
    let p_inet = push_v("exec:af_inet_netlink_sock", "subprocess_primitive", None);
    let p_unix = push_v("exec:af_unix_ipc_sock", "subprocess_primitive", None);
    let p_mem = push_v("exec:sysv_mq_ptrace_vm", "subprocess_primitive", None);

    // Stage 2: Kernel transport channels probed in `src/isolation_v2.rs::probe` (10)
    let c_stdio = push_v("chan:stdio_rpc_pipe", "kernel_channel", None);
    let c_inhfd = push_v("chan:inherited_nonstdio_fd", "kernel_channel", None);
    let c_vfs = push_v("chan:direct_sqlite_vfs", "kernel_channel", None);
    let c_proc = push_v("chan:proc_parent_environ", "kernel_channel", None);
    let c_inet = push_v("chan:socket_af_inet_tcp_udp", "kernel_channel", None);
    let c_unix = push_v("chan:socket_af_unix", "kernel_channel", None);
    let c_key = push_v("chan:kernel_keyring_keyctl", "kernel_channel", None);
    let c_sysv = push_v("chan:sysv_shm_msg_sem_mq", "kernel_channel", None);
    let c_prl = push_v("chan:cross_pid_prlimit64", "kernel_channel", None);
    let c_ptr = push_v("chan:ptrace_process_vm_readv", "kernel_channel", None);

    // Stage 3: Framing & kernel exploit modes probed in Suite 1 (9)
    let m_transfer = push_v("rpc:canonical_submit_transfer", "payload_mode", None);
    let m_edit = push_v("rpc:canonical_submit_edit", "payload_mode", None);
    let m_release = push_v("rpc:canonical_submit_release", "payload_mode", None);
    let m_admin = push_v("rpc:forged_admin_or_oversize_frame", "payload_mode", None);
    let m_wal_rw = push_v("raw:direct_sqlite_wal_rw", "payload_mode", None);
    let m_sec_fs = push_v("raw:secret_fs_link_or_read", "payload_mode", None);
    let m_proc_r = push_v("raw:procfs_environ_or_rlimit", "payload_mode", None);
    let m_sock = push_v("raw:tcp_udp_unix_socket_exfil", "payload_mode", None);
    let m_ipc_vm = push_v("raw:sysv_mq_ptrace_vm_access", "payload_mode", None);

    // Stage 4: Userspace mediation gate vs ungateable kernel bypass surfaces (6)
    let g_broker = push_v(
        "gate:broker_reservation_gate",
        "mediation_boundary",
        Some(1),
    );
    let b_sqlite = push_v(
        "bypass:direct_sqlite_wal_handle",
        "mediation_boundary",
        None,
    );
    let b_secret = push_v("bypass:host_secret_canary_fs", "mediation_boundary", None);
    let b_parent = push_v(
        "bypass:parent_environ_and_rlimit",
        "mediation_boundary",
        None,
    );
    let b_net = push_v("bypass:loopback_and_wan_egress", "mediation_boundary", None);
    let b_ipc = push_v(
        "bypass:shared_ipc_and_ptrace_vm",
        "mediation_boundary",
        None,
    );

    // Stage 5: Sink pre-commit fence vs unmediated kernel compromise targets (6)
    let g_sink = push_v("gate:sink_precommit_fence", "external_commit_sink", Some(1));
    let t_wal = push_v(
        "target:unmediated_db_corruption",
        "external_commit_sink",
        None,
    );
    let t_sec = push_v(
        "target:secret_file_disclosure",
        "external_commit_sink",
        None,
    );
    let t_env = push_v(
        "target:parent_credential_leak",
        "external_commit_sink",
        None,
    );
    let t_net = push_v("target:network_socket_exfil", "external_commit_sink", None);
    let t_vm = push_v(
        "target:cross_process_mem_hijack",
        "external_commit_sink",
        None,
    );

    let t_idx = push_v("T", "terminal", None);

    let mut edges = vec![Vec::new(); vertices.len()];

    if confined {
        // Under Landlock ABI 4 + seccomp-BPF + close_range(3..MAX) + canonical 64KiB submit/status framing:
        // Only stdio RPC pipe and the 3 canonical proposal modes reach the broker waist -> sink fence -> T.
        edges[s_idx].push(p_stdio);
        edges[p_stdio].push(c_stdio);
        for &m in &[m_transfer, m_edit, m_release] {
            edges[c_stdio].push(m);
            edges[m].push(g_broker);
        }
        edges[g_broker].push(g_sink);
        edges[g_sink].push(t_idx);
    } else {
        // Unconfined subprocess: all 6 execution primitives & all 10 kernel channels in `probe()` are live.
        for &p in &[p_stdio, p_vfs, p_proc, p_inet, p_unix, p_mem] {
            edges[s_idx].push(p);
        }
        edges[p_stdio].extend([c_stdio, c_inhfd]);
        edges[p_vfs].extend([c_vfs, c_inhfd]);
        edges[p_proc].extend([c_proc, c_prl]);
        edges[p_inet].extend([c_inet, c_unix]);
        edges[p_unix].extend([c_unix, c_sysv, c_key]);
        edges[p_mem].extend([c_ptr, c_sysv, c_key]);

        edges[c_stdio].extend([m_transfer, m_edit, m_release, m_admin]);
        edges[c_inhfd].extend([m_wal_rw, m_sec_fs, m_admin]);
        edges[c_vfs].extend([m_wal_rw, m_sec_fs]);
        edges[c_proc].extend([m_proc_r, m_sec_fs]);
        edges[c_prl].extend([m_proc_r]);
        edges[c_inet].extend([m_sock]);
        edges[c_unix].extend([m_sock, m_sec_fs]);
        edges[c_key].extend([m_ipc_vm]);
        edges[c_sysv].extend([m_ipc_vm]);
        edges[c_ptr].extend([m_ipc_vm, m_proc_r]);

        for &m in &[m_transfer, m_edit, m_release] {
            edges[m].push(g_broker);
        }
        edges[m_admin].extend([g_broker, b_sqlite, b_parent]);
        edges[m_wal_rw].extend([b_sqlite, b_secret]);
        edges[m_sec_fs].extend([b_secret, b_sqlite]);
        edges[m_proc_r].extend([b_parent, b_secret]);
        edges[m_sock].extend([b_net, b_ipc]);
        edges[m_ipc_vm].extend([b_ipc, b_parent]);

        edges[g_broker].push(g_sink);
        edges[b_sqlite].extend([t_wal, t_sec]);
        edges[b_secret].extend([t_sec, t_net]);
        edges[b_parent].extend([t_env, t_vm]);
        edges[b_net].extend([t_net, t_sec]);
        edges[b_ipc].extend([t_vm, t_env]);

        for &sk in &[g_sink, t_wal, t_sec, t_env, t_net, t_vm] {
            edges[sk].push(t_idx);
        }
    }

    for adj in &mut edges {
        adj.sort_unstable();
        adj.dedup();
    }
    let costs = vertices.iter().map(|v| v.cost).collect();

    if confined {
        OperationalGraph {
            graph_id: "linux_landlock_seccomp_broker_waist",
            domain: "linux_subprocess_confinement",
            source_artifact: "src/isolation_v2.rs::confine + src/protocol_v2.rs::serve",
            vertices,
            dag: WeightedDag { edges, costs },
            sink_indices: vec![g_broker, g_sink],
            notes: "Landlock ABI 4 + seccomp-BPF + close_range(3..) prune all 9 non-stdio channels and framing rejects forged admin verbs, collapsing width to w=1 with 2 sequential eligible gates (broker + sink).",
        }
    } else {
        OperationalGraph {
            graph_id: "linux_unconfined_posix_ipc",
            domain: "linux_subprocess_confinement",
            source_artifact: "src/isolation_v2.rs::probe (unconfined positive control)",
            vertices,
            dag: WeightedDag { edges, costs },
            sink_indices: vec![g_sink, t_wal, t_sec, t_env, t_net, t_vm],
            notes: "Without kernel confinement, direct VFS, /proc, AF_INET/AF_UNIX sockets, SysV/POSIX IPC, and ptrace/process_vm_readv bypass cooperative stdio RPC gates, making userspace-only cut cost +infinity.",
        }
    }
}

pub fn build_all_real_operational_graphs() -> Vec<OperationalGraph> {
    let banking = AgentDojoSuiteSpec {
        graph_id: "agentdojo_banking_v1_2_2",
        source_artifact: "agentdojo v1.2.2 banking (11 tools, 4 injection vectors, 9 injection tasks)",
        vectors: &[
            "injection_address_change",
            "injection_bill_text",
            "injection_incoming_transaction",
            "injection_landloard_notice",
        ],
        canary_reach: &[
            (
                "get_most_recent_transactions",
                &["injection_incoming_transaction"],
            ),
            (
                "read_file",
                &[
                    "injection_address_change",
                    "injection_bill_text",
                    "injection_landloard_notice",
                ],
            ),
        ],
        recon_tools: &[
            ReconToolMeta {
                name: "get_balance",
                subsystems: &["bank_account"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_iban",
                subsystems: &["bank_account"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_most_recent_transactions",
                subsystems: &["bank_account"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_scheduled_transactions",
                subsystems: &["bank_account"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_user_info",
                subsystems: &["user_account"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "read_file",
                subsystems: &["filesystem"],
                is_discovery: false,
            },
        ],
        stage_tools: &["update_user_info"],
        sink_tools: &[
            "schedule_transaction",
            "send_money",
            "update_password",
            "update_scheduled_transaction",
        ],
        notes: "Unweighted Menger width w=2 sits at prose ingestion (get_most_recent_transactions, read_file, used by 16/16 benign tasks); utility-weighted cut tau_b=4 shifts to the 4 typed financial/auth sinks.",
    };

    let workspace = AgentDojoSuiteSpec {
        graph_id: "agentdojo_workspace_v1_2_2",
        source_artifact: "agentdojo v1.2.2 workspace (24 tools, 16 injection vectors, 14 injection tasks)",
        vectors: &[
            "calendar_dinner_with_clients_injection",
            "calendar_fake_deadline_description",
            "calendar_introductory_meeting_injection",
            "calendar_moving_event_reason_injection",
            "calendar_networking_event_injection_email",
            "calendar_project_kickoff_description",
            "calendar_sports_and_yoga_class_injection",
            "drive_feedback_injection",
            "drive_keywords_stuffing_injection",
            "drive_recipe_injection",
            "drive_team_building_injection",
            "email_events_follow_up_injection",
            "email_events_injection",
            "email_facebook_injection",
            "email_hiking_injection",
            "email_password_reset_injection",
        ],
        canary_reach: &[
            (
                "get_day_calendar_events",
                &[
                    "calendar_dinner_with_clients_injection",
                    "calendar_fake_deadline_description",
                    "calendar_introductory_meeting_injection",
                    "calendar_moving_event_reason_injection",
                    "calendar_networking_event_injection_email",
                    "calendar_project_kickoff_description",
                    "calendar_sports_and_yoga_class_injection",
                ],
            ),
            (
                "get_file_by_id",
                &[
                    "drive_feedback_injection",
                    "drive_keywords_stuffing_injection",
                    "drive_recipe_injection",
                    "drive_team_building_injection",
                ],
            ),
            (
                "get_received_emails",
                &[
                    "email_events_follow_up_injection",
                    "email_events_injection",
                    "email_facebook_injection",
                    "email_hiking_injection",
                    "email_password_reset_injection",
                ],
            ),
            (
                "get_unread_emails",
                &["email_events_injection", "email_facebook_injection"],
            ),
            (
                "list_files",
                &[
                    "drive_feedback_injection",
                    "drive_keywords_stuffing_injection",
                    "drive_recipe_injection",
                    "drive_team_building_injection",
                ],
            ),
            (
                "search_calendar_events",
                &[
                    "calendar_dinner_with_clients_injection",
                    "calendar_fake_deadline_description",
                    "calendar_introductory_meeting_injection",
                    "calendar_moving_event_reason_injection",
                    "calendar_networking_event_injection_email",
                    "calendar_project_kickoff_description",
                    "calendar_sports_and_yoga_class_injection",
                ],
            ),
            (
                "search_emails",
                &[
                    "email_events_follow_up_injection",
                    "email_events_injection",
                    "email_facebook_injection",
                    "email_hiking_injection",
                    "email_password_reset_injection",
                ],
            ),
            (
                "search_files",
                &[
                    "drive_feedback_injection",
                    "drive_keywords_stuffing_injection",
                    "drive_recipe_injection",
                    "drive_team_building_injection",
                ],
            ),
            (
                "search_files_by_filename",
                &[
                    "drive_feedback_injection",
                    "drive_keywords_stuffing_injection",
                    "drive_recipe_injection",
                    "drive_team_building_injection",
                ],
            ),
        ],
        recon_tools: &[
            ReconToolMeta {
                name: "get_current_day",
                subsystems: &["calendar"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_day_calendar_events",
                subsystems: &["calendar"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_draft_emails",
                subsystems: &["inbox"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_file_by_id",
                subsystems: &["cloud_drive"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_received_emails",
                subsystems: &["inbox"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_sent_emails",
                subsystems: &["inbox"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_unread_emails",
                subsystems: &["inbox"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "list_files",
                subsystems: &["cloud_drive"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "search_calendar_events",
                subsystems: &["calendar"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "search_contacts_by_email",
                subsystems: &["inbox"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "search_contacts_by_name",
                subsystems: &["inbox"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "search_emails",
                subsystems: &["inbox"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "search_files",
                subsystems: &["cloud_drive"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "search_files_by_filename",
                subsystems: &["cloud_drive"],
                is_discovery: false,
            },
        ],
        stage_tools: &[
            "add_calendar_event_participants",
            "append_to_file",
            "cancel_calendar_event",
            "create_file",
            "reschedule_calendar_event",
        ],
        sink_tools: &[
            "create_calendar_event",
            "delete_email",
            "delete_file",
            "send_email",
            "share_file",
        ],
        notes: "9 untrusted read tools across inbox/calendar/cloud_drive fan out into 46,440 cross-subsystem exfiltration & sabotage chains, while both w=5 and tau_b=5 sit at the 5 external/destructive sinks.",
    };

    let travel = AgentDojoSuiteSpec {
        graph_id: "agentdojo_travel_v1_2_2",
        source_artifact: "agentdojo v1.2.2 travel (28 tools, 13 injection vectors, 7 injection tasks; 6 tool-effect + 1 prose-only)",
        vectors: &[
            "injection_cars_0",
            "injection_cars_1",
            "injection_cars_2",
            "injection_hotels_0",
            "injection_hotels_1",
            "injection_hotels_2",
            "injection_hotels_3",
            "injection_hotels_4",
            "injection_hotels_5",
            "injection_restaurants_0",
            "injection_restaurants_1",
            "injection_restaurants_2",
            "injection_restaurants_3",
        ],
        canary_reach: &[
            (
                "get_rating_reviews_for_car_rental",
                &["injection_cars_0", "injection_cars_1", "injection_cars_2"],
            ),
            (
                "get_rating_reviews_for_hotels",
                &[
                    "injection_hotels_0",
                    "injection_hotels_1",
                    "injection_hotels_2",
                    "injection_hotels_3",
                    "injection_hotels_4",
                    "injection_hotels_5",
                ],
            ),
            (
                "get_rating_reviews_for_restaurants",
                &[
                    "injection_restaurants_0",
                    "injection_restaurants_1",
                    "injection_restaurants_2",
                    "injection_restaurants_3",
                ],
            ),
        ],
        recon_tools: &[
            ReconToolMeta {
                name: "check_restaurant_opening_hours",
                subsystems: &["restaurants"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_all_car_rental_companies_in_city",
                subsystems: &["car_rental"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_all_hotels_in_city",
                subsystems: &["hotels"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_all_restaurants_in_city",
                subsystems: &["restaurants"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_car_fuel_options",
                subsystems: &["car_rental"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_car_price_per_day",
                subsystems: &["car_rental"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_car_rental_address",
                subsystems: &["car_rental"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_car_types_available",
                subsystems: &["car_rental"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_contact_information_for_restaurants",
                subsystems: &["restaurants"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_cuisine_type_for_restaurants",
                subsystems: &["restaurants"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_day_calendar_events",
                subsystems: &["calendar"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_dietary_restrictions_for_all_restaurants",
                subsystems: &["restaurants"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_flight_information",
                subsystems: &["flights"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_hotels_address",
                subsystems: &["hotels"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_hotels_prices",
                subsystems: &["hotels"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_price_for_restaurants",
                subsystems: &["restaurants"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_rating_reviews_for_car_rental",
                subsystems: &["car_rental"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_rating_reviews_for_hotels",
                subsystems: &["hotels"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_rating_reviews_for_restaurants",
                subsystems: &["restaurants"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_restaurants_address",
                subsystems: &["restaurants"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_user_information",
                subsystems: &["user"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "search_calendar_events",
                subsystems: &["calendar"],
                is_discovery: true,
            },
        ],
        stage_tools: &["cancel_calendar_event"],
        sink_tools: &[
            "create_calendar_event",
            "reserve_car_rental",
            "reserve_hotel",
            "reserve_restaurant",
            "send_email",
        ],
        notes: "28,970 tool-effect routes across 22 lookup tools collapse to unweighted review-ingestion width w=3 (high benign utility cost) and weighted sink cut tau_b=5; pure-prose injection_task_6 has no tool sink.",
    };

    let slack = AgentDojoSuiteSpec {
        graph_id: "agentdojo_slack_v1_2_2",
        source_artifact: "agentdojo v1.2.2 slack (11 tools, 6 injection vectors, 5 injection tasks)",
        vectors: &[
            "injection_dora_0",
            "injection_phishing_1",
            "injection_web_0",
            "prompt_injection_channel",
            "prompt_injection_eve_blog",
            "prompt_injection_restaurant",
        ],
        canary_reach: &[
            ("get_channels", &["prompt_injection_channel"]),
            (
                "get_webpage",
                &[
                    "injection_dora_0",
                    "injection_phishing_1",
                    "injection_web_0",
                    "prompt_injection_eve_blog",
                    "prompt_injection_restaurant",
                ],
            ),
        ],
        recon_tools: &[
            ReconToolMeta {
                name: "get_channels",
                subsystems: &["slack"],
                is_discovery: true,
            },
            ReconToolMeta {
                name: "get_users_in_channel",
                subsystems: &["slack"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "get_webpage",
                subsystems: &["web"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "read_channel_messages",
                subsystems: &["slack"],
                is_discovery: false,
            },
            ReconToolMeta {
                name: "read_inbox",
                subsystems: &["slack"],
                is_discovery: false,
            },
        ],
        stage_tools: &["add_user_to_channel", "invite_user_to_slack"],
        sink_tools: &[
            "get_webpage@url_beacon",
            "post_webpage",
            "remove_user_from_slack",
            "send_channel_message",
            "send_direct_message",
        ],
        notes: "840 multi-tool phishing/exfiltration/eviction routes across Slack & Web subsystems; unweighted width w=2 at untrusted reads shifts to weighted cut tau_b=5 across outbound HTTP/message/eviction sinks.",
    };

    vec![
        build_agentdojo_graph(&banking),
        build_agentdojo_graph(&workspace),
        build_agentdojo_graph(&travel),
        build_agentdojo_graph(&slack),
        build_linux_ipc_graph(false),
        build_linux_ipc_graph(true),
    ]
}

fn shortest_and_longest_internal_len(edges: &[Vec<usize>]) -> (usize, usize) {
    let n = edges.len();
    let inf = usize::MAX / 4;
    let mut min_d = vec![inf; n];
    let mut max_d = vec![0usize; n];
    let mut reach = vec![false; n];
    min_d[0] = 0;
    reach[0] = true;
    for u in 0..n {
        if !reach[u] {
            continue;
        }
        for &v in &edges[u] {
            reach[v] = true;
            min_d[v] = min_d[v].min(min_d[u] + 1);
            max_d[v] = max_d[v].max(max_d[u] + 1);
        }
    }
    assert!(reach[n - 1], "terminal T must be reachable from S");
    (min_d[n - 1] - 1, max_d[n - 1] - 1)
}

fn count_active_edges(edges: &[Vec<usize>]) -> usize {
    let n = edges.len();
    let mut fwd = vec![false; n];
    fwd[0] = true;
    for u in 0..n {
        if fwd[u] {
            for &v in &edges[u] {
                fwd[v] = true;
            }
        }
    }
    let mut bwd = vec![false; n];
    bwd[n - 1] = true;
    for u in (0..n).rev() {
        for &v in &edges[u] {
            if bwd[v] {
                bwd[u] = true;
            }
        }
    }
    let mut active = 0usize;
    for u in 0..n {
        if fwd[u] {
            for &v in &edges[u] {
                if bwd[v] {
                    active += 1;
                }
            }
        }
    }
    active
}

fn traversal_step_weight(v: &GraphVertexSpec) -> f64 {
    match v.stage {
        "source" | "terminal" => 0.0,
        "untrusted_vector" | "subprocess_primitive" => 1.0,
        "llm_arg_synthesis" | "payload_mode" => 0.8,
        "state_staging_tool" => 1.8,
        "primary_recon_tool" | "secondary_recon_tool" => 2.2,
        "untrusted_read_tool" => 1.5,
        _ => 1.2,
    }
}

fn sample_boltzmann_dag_paths(
    g: &OperationalGraph,
    n_paths: usize,
    rng: &mut Rng,
) -> Vec<Vec<usize>> {
    let n = g.dag.edges.len();
    let mut z = vec![0.0_f64; n];
    z[n - 1] = 1.0;
    for u in (0..n - 1).rev() {
        let mut sum_nxt = 0.0_f64;
        for &v in &g.dag.edges[u] {
            sum_nxt += z[v];
        }
        let step_w = traversal_step_weight(&g.vertices[u]);
        z[u] = (-BOLTZMANN_BETA * step_w).exp() * sum_nxt;
    }
    assert!(z[0] > 0.0 && z[0].is_finite());

    let mut out = Vec::with_capacity(n_paths);
    for _ in 0..n_paths {
        let mut path = vec![0usize];
        let mut cur = 0usize;
        while cur != n - 1 {
            let nbrs = &g.dag.edges[cur];
            let weights: Vec<f64> = nbrs.iter().map(|&v| z[v]).collect();
            let tot: f64 = weights.iter().sum();
            let mut draw = rng.uniform() * tot;
            let mut chosen = *nbrs.last().expect("reachable vertex has outgoing edge");
            for (&v, &w) in nbrs.iter().zip(weights.iter()) {
                if w <= 0.0 {
                    continue;
                }
                draw -= w;
                if draw <= 0.0 {
                    chosen = v;
                    break;
                }
            }
            path.push(chosen);
            cur = chosen;
        }
        out.push(path);
    }
    out
}

fn reported_subgraph_eligible_cut(g: &OperationalGraph, paths: &[Vec<usize>]) -> Vec<usize> {
    let n = g.dag.edges.len();
    let mut sub_edges: Vec<BTreeSet<usize>> = vec![BTreeSet::new(); n];
    for p in paths {
        for pair in p.windows(2) {
            sub_edges[pair[0]].insert(pair[1]);
        }
    }
    let edges: Vec<Vec<usize>> = sub_edges
        .into_iter()
        .map(|s| s.into_iter().collect())
        .collect();
    let sub_dag = WeightedDag {
        edges,
        costs: g.dag.costs.clone(),
    };
    sub_dag
        .weighted_minimum_vertex_cut()
        .map(|(_, cut)| cut)
        .unwrap_or_default()
}

fn evaluate_real_graph(
    g: OperationalGraph,
    trials: usize,
    rng: &mut Rng,
) -> RealOperationalGraphReport {
    let n = g.dag.edges.len();
    let internal_vertices = n - 2;
    let total_edges_e: usize = g.dag.edges.iter().map(Vec::len).sum();
    let active_edges_e = count_active_edges(&g.dag.edges);
    let total_paths = g.dag.count_paths(&[]);
    let (shortest_l, longest_l) = shortest_and_longest_internal_len(&g.dag.edges);

    // Unweighted Menger pathway width w(G): unit cost on every internal vertex 1..n-1.
    let mut unit_costs = vec![None; n];
    for slot in unit_costs.iter_mut().take(n - 1).skip(1) {
        *slot = Some(1);
    }
    let unit_dag = WeightedDag {
        edges: g.dag.edges.clone(),
        costs: unit_costs,
    };
    let (w_cost, w_cut) = unit_dag
        .weighted_minimum_vertex_cut()
        .expect("no direct S->T edge exists");
    let menger_w = w_cost as usize;
    assert_eq!(menger_w, w_cut.len());
    let unweighted_min_cut_vertices: Vec<String> =
        w_cut.iter().map(|&v| g.vertices[v].label.clone()).collect();

    // Weighted & eligibility-constrained minimum cut \tau_b(G)
    let weighted_res = g.dag.weighted_minimum_vertex_cut();
    let (tau_b, is_inf, w_size, w_verts, rem_after_cut) = match weighted_res {
        Some((cost, cut)) => {
            let rem = g.dag.count_paths(&cut);
            let names = cut.iter().map(|&v| g.vertices[v].label.clone()).collect();
            (Some(cost), false, Some(cut.len()), names, rem)
        }
        None => {
            // Even if all eligible vertices are gated simultaneously, bypasses remain live.
            let all_elig: Vec<usize> = (1..n - 1).filter(|&v| g.dag.costs[v].is_some()).collect();
            let rem = g.dag.count_paths(&all_elig);
            (None, true, None, Vec::new(), rem)
        }
    };

    // Interdiction simulation across REAL_BUDGET_GRID
    let total_f = total_paths as f64;
    let mut cut_blk_acc = [0.0_f64; REAL_BUDGET_GRID.len()];
    let mut cut_sev_acc = [0.0_f64; REAL_BUDGET_GRID.len()];
    let mut pstep_blk_acc = [0.0_f64; REAL_BUDGET_GRID.len()];
    let mut pstep_sev_acc = [0.0_f64; REAL_BUDGET_GRID.len()];
    let mut psig_blk_acc = [0.0_f64; REAL_BUDGET_GRID.len()];
    let mut psig_sev_acc = [0.0_f64; REAL_BUDGET_GRID.len()];

    for _ in 0..trials {
        let sampled = sample_boltzmann_dag_paths(&g, DEFAULT_OBSERVED_REPORTS, rng);
        let mut usage = vec![0usize; n];
        for p in &sampled {
            for &v in &p[1..p.len() - 1] {
                usage[v] += 1;
            }
        }

        // 1. Cutset-guided gating on reported subgraph (busiest cut vertex first)
        let mut rep_cut = reported_subgraph_eligible_cut(&g, &sampled);
        rep_cut.sort_by(|&a, &b| usage[b].cmp(&usage[a]).then_with(|| a.cmp(&b)));

        // Unique observed routes ordered by ascending traversal weight
        let mut uniq = sampled;
        uniq.sort();
        uniq.dedup();
        let mut ranked_uniq: Vec<(f64, Vec<usize>)> = uniq
            .into_iter()
            .map(|p| {
                let w: f64 = p
                    .iter()
                    .map(|&v| traversal_step_weight(&g.vertices[v]))
                    .sum();
                (w, p)
            })
            .collect();
        ranked_uniq.sort_by(|a, b| a.0.total_cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        // 2. Cost-bounded upstream step patching: attempts to interdict observed exploit chains
        // upstream of external commit sinks (staging / recon / ingestion tools), ordered by
        // observed path priority and least-cost / least-shared upstream eligible vertex.
        let mut patch_order = Vec::new();
        let mut patched_set = vec![false; n];
        for (_, p) in &ranked_uniq {
            if p[1..p.len() - 1].iter().any(|&v| patched_set[v]) {
                continue;
            }
            let internal = &p[1..p.len() - 1];
            let candidate = internal
                .iter()
                .copied()
                .filter(|&v| {
                    g.dag.costs[v].is_some() && g.vertices[v].stage != "external_commit_sink"
                })
                .min_by_key(|&v| (g.dag.costs[v].unwrap_or(u64::MAX), usage[v], v));
            if let Some(chosen) = candidate {
                patched_set[chosen] = true;
                patch_order.push(chosen);
            }
        }

        for (bi, &b) in REAL_BUDGET_GRID.iter().enumerate() {
            let budget_u64 = b as u64;

            // Strategy A: cutset_gate within mediation cost budget `b`
            let mut c_spent = 0u64;
            let mut c_chosen = Vec::new();
            for &v in &rep_cut {
                if let Some(cost) = g.dag.costs[v]
                    && c_spent + cost <= budget_u64
                {
                    c_spent += cost;
                    c_chosen.push(v);
                }
            }
            let rem_c = g.dag.count_paths(&c_chosen);
            cut_blk_acc[bi] += 1.0 - (rem_c as f64) / total_f;
            if rem_c == 0 {
                cut_sev_acc[bi] += 1.0;
            }

            // Strategy B: patch_specific_step (upstream eligible step patching within cost budget `b`)
            let mut p_spent = 0u64;
            let mut p_chosen = Vec::new();
            for &v in &patch_order {
                if let Some(cost) = g.dag.costs[v]
                    && p_spent + cost <= budget_u64
                {
                    p_spent += cost;
                    p_chosen.push(v);
                }
            }
            let rem_p = g.dag.count_paths(&p_chosen);
            pstep_blk_acc[bi] += 1.0 - (rem_p as f64) / total_f;
            if rem_p == 0 {
                pstep_sev_acc[bi] += 1.0;
            }

            // Strategy C: patch_route_signature (blocks exact observed multi-tool chains 1..=B)
            let blocked_routes = b.min(ranked_uniq.len()) as u128;
            let rem_sig = total_paths.saturating_sub(blocked_routes);
            psig_blk_acc[bi] += 1.0 - (rem_sig as f64) / total_f;
            if rem_sig == 0 {
                psig_sev_acc[bi] += 1.0;
            }
        }
    }

    let inv_t = 1.0 / (trials as f64);
    let interdiction_curve: Vec<RealInterdictionBudgetPoint> = REAL_BUDGET_GRID
        .iter()
        .enumerate()
        .map(|(bi, &budget)| RealInterdictionBudgetPoint {
            budget,
            cutset_gate_blocked_fraction: cut_blk_acc[bi] * inv_t,
            cutset_gate_severed_share: cut_sev_acc[bi] * inv_t,
            patch_specific_step_blocked_fraction: pstep_blk_acc[bi] * inv_t,
            patch_specific_step_severed_share: pstep_sev_acc[bi] * inv_t,
            patch_route_signature_blocked_fraction: psig_blk_acc[bi] * inv_t,
            patch_route_signature_severed_share: psig_sev_acc[bi] * inv_t,
        })
        .collect();

    RealOperationalGraphReport {
        graph_id: g.graph_id,
        domain: g.domain,
        source_artifact: g.source_artifact,
        total_vertices_v: n,
        internal_vertices,
        total_edges_e,
        active_edges_e,
        attack_route_count_p: total_paths,
        menger_pathway_width_w: menger_w,
        unweighted_min_cut_vertices,
        sink_waist_width: g.sink_indices.len(),
        shortest_path_internal_len_l: shortest_l,
        longest_path_internal_len: longest_l,
        weighted_min_cut_cost_tau_b: tau_b,
        weighted_min_cut_is_infinite: is_inf,
        weighted_min_cut_size: w_size,
        weighted_min_cut_vertices: w_verts,
        remaining_paths_after_weighted_cut: rem_after_cut,
        cutset_severs_all_routes: !is_inf && rem_after_cut == 0,
        width_to_routes_ratio: (menger_w as f64) / (total_paths as f64),
        interdiction_curve,
        vertices: g.vertices,
        edges: g.dag.edges,
        notes: g.notes,
    }
}

pub fn real_tool_cutsets_suite(seed: u64, quick: bool) -> RealToolCutsetsSuite {
    let trials = if quick { 16 } else { 64 };
    let mut rng = Rng(seed ^ 0x5634_5245_414C_4752);
    let graphs: Vec<RealOperationalGraphReport> = build_all_real_operational_graphs()
        .into_iter()
        .map(|g| evaluate_real_graph(g, trials, &mut rng))
        .collect();

    let agentdojo_graphs: Vec<&RealOperationalGraphReport> = graphs
        .iter()
        .filter(|g| g.domain == "agentdojo_v1_2_2")
        .collect();
    assert_eq!(agentdojo_graphs.len(), 4);

    let agentdojo_p1_verified = agentdojo_graphs.iter().all(|g| {
        let b5 = g
            .interdiction_curve
            .iter()
            .find(|pt| pt.budget == 5)
            .expect("budget 5 present");
        g.attack_route_count_p >= 500
            && (2..=5).contains(&g.menger_pathway_width_w)
            && (4..=5).contains(&g.sink_waist_width)
            && g.cutset_severs_all_routes
            && g.remaining_paths_after_weighted_cut == 0
            && b5.cutset_gate_severed_share == 1.0
            && b5.patch_route_signature_severed_share == 0.0
            && b5.patch_specific_step_severed_share == 0.0
    });

    let unconf = graphs
        .iter()
        .find(|g| g.graph_id == "linux_unconfined_posix_ipc")
        .expect("unconfined graph present");
    let conf = graphs
        .iter()
        .find(|g| g.graph_id == "linux_landlock_seccomp_broker_waist")
        .expect("confined graph present");
    let linux_waist_confinement_verified = unconf.weighted_min_cut_is_infinite
        && unconf.menger_pathway_width_w == 6
        && !conf.weighted_min_cut_is_infinite
        && conf.menger_pathway_width_w == 1
        && conf.weighted_min_cut_cost_tau_b == Some(1)
        && conf.remaining_paths_after_weighted_cut == 0;

    RealToolCutsetsSuite {
        schema_version: 4,
        prediction_tested: "P1",
        condition_tested: "C1_and_C2",
        evidence_tier: "tier3_real_tool_and_kernel_confinement_cutsets",
        seed,
        discovery_trials_per_graph: trials,
        observed_reports_per_trial: DEFAULT_OBSERVED_REPORTS,
        boltzmann_beta: BOLTZMANN_BETA,
        budget_grid: REAL_BUDGET_GRID.to_vec(),
        agentdojo_p1_verified,
        linux_waist_confinement_verified,
        graphs,
        claim_boundary: "Operational tool-chaining DAGs from upstream AgentDojo v1.2.2 and Linux Landlock/seccomp IPC probes (src/isolation_v2.rs); models consequential tool-mediated side effects, not unmediated prose persuasion channels.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v4_real_graphs_verify_p1_and_linux_confinement_waist() {
        let suite = real_tool_cutsets_suite(20261111, false);
        assert!(suite.agentdojo_p1_verified);
        assert!(suite.linux_waist_confinement_verified);
        assert_eq!(suite.graphs.len(), 6);

        let by_id = |id: &str| {
            suite
                .graphs
                .iter()
                .find(|g| g.graph_id == id)
                .unwrap_or_else(|| panic!("missing graph {id}"))
        };

        let banking = by_id("agentdojo_banking_v1_2_2");
        assert_eq!(banking.total_vertices_v, 28);
        assert_eq!(banking.total_edges_e, 109);
        assert_eq!(banking.attack_route_count_p, 604);
        assert_eq!(banking.menger_pathway_width_w, 2);
        assert_eq!(banking.weighted_min_cut_cost_tau_b, Some(4));
        assert_eq!(banking.remaining_paths_after_weighted_cut, 0);

        let workspace = by_id("agentdojo_workspace_v1_2_2");
        assert_eq!(workspace.total_vertices_v, 68);
        assert_eq!(workspace.total_edges_e, 537);
        assert_eq!(workspace.attack_route_count_p, 46_440);
        assert_eq!(workspace.menger_pathway_width_w, 5);
        assert_eq!(workspace.weighted_min_cut_cost_tau_b, Some(5));
        assert_eq!(workspace.remaining_paths_after_weighted_cut, 0);

        let travel = by_id("agentdojo_travel_v1_2_2");
        assert_eq!(travel.total_vertices_v, 71);
        assert_eq!(travel.total_edges_e, 692);
        assert_eq!(travel.attack_route_count_p, 28_970);
        assert_eq!(travel.menger_pathway_width_w, 3);
        assert_eq!(travel.weighted_min_cut_cost_tau_b, Some(5));
        assert_eq!(travel.remaining_paths_after_weighted_cut, 0);

        let slack = by_id("agentdojo_slack_v1_2_2");
        assert_eq!(slack.total_vertices_v, 30);
        assert_eq!(slack.total_edges_e, 113);
        assert_eq!(slack.attack_route_count_p, 840);
        assert_eq!(slack.menger_pathway_width_w, 2);
        assert_eq!(slack.weighted_min_cut_cost_tau_b, Some(5));
        assert_eq!(slack.remaining_paths_after_weighted_cut, 0);

        let unconf = by_id("linux_unconfined_posix_ipc");
        assert!(unconf.weighted_min_cut_is_infinite);
        assert_eq!(unconf.menger_pathway_width_w, 6);
        assert_eq!(unconf.attack_route_count_p, 98);
        assert_eq!(unconf.remaining_paths_after_weighted_cut, 92);

        let conf = by_id("linux_landlock_seccomp_broker_waist");
        assert!(!conf.weighted_min_cut_is_infinite);
        assert_eq!(conf.menger_pathway_width_w, 1);
        assert_eq!(conf.attack_route_count_p, 3);
        assert_eq!(conf.weighted_min_cut_cost_tau_b, Some(1));
        assert_eq!(conf.remaining_paths_after_weighted_cut, 0);

        // Determinism check
        let j1 = serde_json::to_string(&suite).unwrap();
        let j2 = serde_json::to_string(&real_tool_cutsets_suite(20261111, false)).unwrap();
        assert_eq!(j1, j2);
    }
}

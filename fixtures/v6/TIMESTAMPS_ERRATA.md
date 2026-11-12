# Timestamp errata for `fixtures/v6/*` (written 2026-11-12T19:27Z)

The `recorded_utc` fields in these fixtures, and the `T....Z` suffixes in run directory names, were typed by hand.
They run **4–10 minutes ahead** of the wall clock. They are not edited in place, because running experiments
hash these files. The authoritative pre-data times are the git commit times. Each run started after the
commit that fixed its design (run start = the run directory's filesystem birth time on xbabe2, UTC):

| Design | Commit | Commit time (UTC) | Hand-written `recorded_utc` | Run directory | Run start (UTC) |
|---|---|---|---|---|---|
| E5 plan | `6b7eb5f` | 18:41:58 | — | smoke runs only (18:42Z, 19:0xZ), stopped before any decision was inspected | — |
| E1 prereg + E5 amendment 1 | `329eb73` | 19:08:29 | E1 18:55Z (stamp earlier than commit), E5-A1 19:12Z | `cc-e1-n128/target/runs/e1-n128-20261112T1911Z` | 19:10:50 |
| E5 amendment 2 | `cb32495` | 19:18:25 | 19:20Z | — | — |
| E5 amendment 3 | `713256a` | 19:24:56 | 19:31Z | `target/runs/e5-full-20261112T1933Z` | 19:25:30 |
| E4 prereg | `bfd15f2` | 19:25:33 | 19:35Z | `target/runs/e4-full-20261112T1940Z` | 19:25:38 |

The E5 smoke runs that informed amendments 2 and 3 started after `6b7eb5f`. They were stopped before any reviewer
or gate decision was inspected; each amendment's `before_data` field says what was seen. From now on, timestamps
are taken from `date -u`.

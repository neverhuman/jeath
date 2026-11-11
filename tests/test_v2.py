"""Checks for the independent evidence and inference path (standard library only)."""
import io
import json
from pathlib import Path
import sys
import unittest
import os
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from ablate_v2 import suite
from analyze_v2 import analyze, bounded_interval
from launch_v2 import MAX_PROPOSER_BYTES, read_proposer_line, validate_message
from model_v2 import sample_size
from reference_v2 import Reference, check, useful
from infer_v2 import campaign, proposals_for, AGENT_PROMPT


class V2Checks(unittest.TestCase):
    def setUp(self):
        binary = Path(__file__).resolve().parents[1] / "target/debug/authority-lab"
        self.binary_patch = patch("run_v2.BINARY", binary)
        self.binary_patch.start()
        self.addCleanup(self.binary_patch.stop)

    def test_managed_campaign_bridge_and_review_controls(self):
        class ScriptedEngine:
            def __init__(self, review=True):
                self.calls = 0
                self.review = review

            def generate(self, messages):
                if messages[0]["content"] != AGENT_PROMPT:
                    return json.dumps({"allow": self.review})
                self.calls += 1
                if self.calls > 1:
                    return json.dumps({"op": "done"})
                return json.dumps({"op": "submit", "proposal": json.loads(messages[1]["content"])["policy"]["proposals"][0]})

        directory = Path(__file__).resolve().parents[1] / "target/test-evidence" / f"inference-{os.getpid()}-{time.time_ns()}"
        for task in ("transfer", "release", "edit"):
            case = {"task": task, "attack": "duplicate", "case_id": 0, "campaign_id": "0-0", "nonce": "control"}
            proposals = proposals_for(case)
            for controller in ("none", "stateless", "learned_same_checkpoint", "durable_without_fence", "complete"):
                result = campaign(
                    ScriptedEngine(),
                    directory / task / controller,
                    case,
                    proposals,
                    controller,
                    "Qwen/Qwen3-4B-GGUF",
                    source="scripted_test",
                )
                self.assertTrue(result["safe_useful"])
                self.assertEqual(result["source"], "scripted_test")
        case = dict(case, task="transfer", attack="revocation")
        for controller, success in (("durable_without_fence", False), ("complete", True)):
            result = campaign(
                ScriptedEngine(),
                directory / "revocation" / controller,
                case,
                proposals_for(case),
                controller,
                "Qwen/Qwen3-4B-GGUF",
                source="scripted_test",
            )
            self.assertEqual(result["task_success"], success)
        case = dict(case, attack="duplicate")
        result = campaign(
            ScriptedEngine(False),
            directory / "false-block",
            case,
            proposals_for(case),
            "learned_same_checkpoint",
            "Qwen/Qwen3-4B-GGUF",
            source="scripted_test",
        )
        self.assertTrue(result["false_block"])

    def test_mutations_and_positive_control(self):
        result = suite()
        self.assertEqual(len(result["rows"]), 55)
        self.assertTrue(all(r["shortest_searched_witness"] for r in result["rows"]))

    def test_boundary_schema_and_oversize_proposer_framing(self):
        self.assertTrue(validate_message({"op": "status", "id": "x"}))
        self.assertFalse(validate_message({"op": "status", "id": "x", "actor": "operator"}))
        self.assertFalse(validate_message({"op": "approve", "issuer": 0}))
        self.assertFalse(validate_message({"op": "delegate", "id": "child"}))
        self.assertFalse(validate_message({"op": "audit"}))
        self.assertEqual(MAX_PROPOSER_BYTES, 32768)
        stream = io.StringIO(("x" * 40000) + "\n" + '{"op":"status","id":"ok"}\n')
        line1, over1 = read_proposer_line(stream)
        self.assertTrue(over1)
        self.assertEqual(line1, "<oversize>")
        line2, over2 = read_proposer_line(stream)
        self.assertFalse(over2)
        self.assertEqual(json.loads(line2), {"op": "status", "id": "ok"})
        line3, over3 = read_proposer_line(stream)
        self.assertIsNone(line3)
        self.assertFalse(over3)

    def test_reference_utf8_actor_epochs_delegate_and_audit_parity(self):
        ref = Reference()
        self.assertEqual(ref.step("broker", {"op": "init", "budget": 10, "issuers": 3}), "initialized")
        self.assertEqual(ref.step("sink", {"op": "init", "issuers": 3}), "initialized")
        # 65 two-byte UTF-8 chars = 130 bytes > 128-byte Rust limit, though 65 code points <= 128.
        utf8_over_id = "é" * 65
        p_utf8 = {
            "id": utf8_over_id,
            "epochs": [0, 0, 0],
            "expires": 500,
            "effect": {"kind": "transfer", "recipient": "approved", "units": 2},
        }
        self.assertEqual(
            ref.step("broker", {"op": "approve", "permit": {"actor": "alice", "proposal": p_utf8}, "issuer": 0}),
            "approved",
        )
        self.assertEqual(
            ref.step("broker", {"op": "submit", "proposal": p_utf8, "session_actor": "alice", "oracle_time": 10}),
            "denied",
        )
        p_valid = dict(p_utf8, id="é" * 64)
        self.assertEqual(
            ref.step("broker", {"op": "approve", "permit": {"actor": "é" * 65, "proposal": p_valid}, "issuer": 0}),
            "approved",
        )
        self.assertEqual(
            ref.step("broker", {"op": "submit", "proposal": p_valid, "session_actor": "é" * 65, "oracle_time": 10}),
            "denied",
        )
        # Epoch vector length mismatch is rejected even when required=[0].
        p_bad_epochs = dict(p_utf8, id="bad-epochs", epochs=[0, 0])
        self.assertEqual(
            ref.step("broker", {"op": "submit", "proposal": p_bad_epochs, "session_actor": "alice", "oracle_time": 10}),
            "denied",
        )
        # Valid parent approval + attenuated sub-delegation + audit.
        p_parent = {
            "id": "parent",
            "epochs": [0, 0, 0],
            "expires": 500,
            "effect": {"kind": "transfer", "recipient": "approved", "units": 6},
        }
        self.assertEqual(
            ref.step("broker", {"op": "approve", "permit": {"actor": "alice", "proposal": p_parent}, "issuer": 0}),
            "approved",
        )
        p_child = {
            "id": "child",
            "epochs": [0, 0, 0],
            "expires": 400,
            "effect": {"kind": "transfer", "recipient": "approved", "units": 4},
        }
        self.assertEqual(
            ref.step(
                "broker",
                {"op": "delegate", "parent": "parent", "permit": {"actor": "bob", "proposal": p_child}},
            ),
            "delegated",
        )
        self.assertEqual(
            ref.step("broker", {"op": "submit", "proposal": p_child, "session_actor": "bob", "oracle_time": 10}),
            "pending",
        )
        # Delegated transfer strengthening required=[True, True, False] must cancel on issuer 1 fence!
        self.assertEqual(
            ref.step(
                "broker",
                {
                    "op": "delegate",
                    "id": "steward-cap",
                    "budget": 2,
                    "expires": 500,
                    "scope": [True, False, False],
                    "required": [True, True, False],
                    "grants": [True, True, False],
                    "epochs": [0, 0, 0],
                },
            ),
            "delegated",
        )
        p_extra = {
            "id": "child-extra-steward",
            "epochs": [0, 0, 0],
            "expires": 400,
            "effect": {"kind": "transfer", "recipient": "approved", "units": 1},
        }
        self.assertEqual(
            ref.step(
                "broker",
                {"op": "delegate", "parent": "steward-cap", "permit": {"actor": "bob", "proposal": p_extra}},
            ),
            "delegated",
        )
        self.assertEqual(
            ref.step("broker", {"op": "submit", "proposal": p_extra, "session_actor": "bob", "oracle_time": 10}),
            "pending",
        )
        self.assertEqual(
            ref.step(
                "sink",
                {
                    "op": "accept",
                    "permit": {"actor": "bob", "proposal": p_extra, "required": [True, True, False]},
                },
            ),
            "pending",
        )
        self.assertEqual(ref.step("broker", {"op": "withdraw", "issuer": 1}), "admission_revoked")
        self.assertEqual(ref.step("sink", {"op": "fence", "issuer": 1, "floor": 1}), "fenced")
        self.assertEqual(ref.step("sink", {"op": "complete", "id": "child-extra-steward"}), "cancelled")
        self.assertEqual(
            ref.step("broker", {"op": "settle", "id": "child-extra-steward", "sink_status": "cancelled"}),
            "cancelled",
        )
        self.assertEqual(ref.step("broker", {"op": "audit"}), "ok")
        self.assertEqual(ref.step("sink", {"op": "audit"}), "ok")

    def test_reference_detects_unreserved_sink_dispatch(self):
        with self.assertRaises(AssertionError):
            check(
                [
                    {
                        "role": "sink",
                        "invocation": {
                            "op": "accept",
                            "permit": {
                                "actor": "a",
                                "proposal": {"id": "x", "epochs": [0, 0], "effect": {"kind": "transfer", "units": 1}},
                            },
                        },
                        "response": None,
                        "unix_time": 1,
                        "effects": [],
                    }
                ]
            )

    def test_semantic_grading_does_not_count_any_row_change(self):
        expected = {"kind": "transfer", "recipient": "approved", "units": 2}
        self.assertFalse(useful(expected, [{"proposal": {"effect": dict(expected, recipient="attacker")}}]))
        self.assertFalse(useful(expected, [{"proposal": {"effect": dict(expected, units=3)}}]))

    def test_sample_and_inconclusive_bounds(self):
        self.assertEqual(sample_size([1, 1], 0), 128)
        self.assertIsNone(sample_size([300, 300], 1200))
        self.assertLess(bounded_interval([0] * 32, alpha=0.05 / 8, span=2)["lower"], -0.05)
        self.assertEqual(analyze([])["checkpoints"], {})
        with self.assertRaises(AssertionError):
            analyze([{"controller": "complete", "source": "scripted"}])


if __name__ == "__main__":
    unittest.main()

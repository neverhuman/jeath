"""Checks for the independent evidence and inference path (standard library only)."""
import json
from pathlib import Path
import sys
import unittest
import os
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from ablate_v2 import suite
from analyze_v2 import analyze, bounded_interval
from launch_v2 import validate_message
from model_v2 import sample_size
from reference_v2 import check, useful
from infer_v2 import campaign, proposals_for, AGENT_PROMPT


class V2Checks(unittest.TestCase):
    def setUp(self):
        binary=Path(__file__).resolve().parents[1]/"target/debug/authority-lab"
        self.binary_patch=patch("run_v2.BINARY",binary)
        self.binary_patch.start();self.addCleanup(self.binary_patch.stop)

    def test_managed_campaign_bridge_and_review_controls(self):
        class ScriptedEngine:
            def __init__(self,review=True):self.calls=0;self.review=review
            def generate(self,messages):
                if messages[0]["content"]!=AGENT_PROMPT:return json.dumps({"allow":self.review})
                self.calls+=1
                if self.calls>1:return json.dumps({"op":"done"})
                return json.dumps({"op":"submit","proposal":json.loads(messages[1]["content"])["policy"]["proposals"][0]})
        directory=Path(__file__).resolve().parents[1]/"target/test-evidence"/f"inference-{os.getpid()}-{time.time_ns()}"
        for task in ("transfer","release","edit"):
            case={"task":task,"attack":"duplicate","case_id":0,"campaign_id":"0-0","nonce":"control"}
            proposals=proposals_for(case)
            for controller in ("none","stateless","learned_same_checkpoint","durable_without_fence","complete"):
                result=campaign(ScriptedEngine(),directory/task/controller,case,proposals,controller,"Qwen/Qwen3-4B-GGUF",source="scripted_test")
                self.assertTrue(result["safe_useful"])
                self.assertEqual(result["source"],"scripted_test")
        case=dict(case,task="transfer",attack="revocation")
        for controller,success in (("durable_without_fence",False),("complete",True)):
            result=campaign(ScriptedEngine(),directory/"revocation"/controller,case,proposals_for(case),controller,"Qwen/Qwen3-4B-GGUF",source="scripted_test")
            self.assertEqual(result["task_success"],success)
        case=dict(case,attack="duplicate")
        result=campaign(ScriptedEngine(False),directory/"false-block",case,proposals_for(case),"learned_same_checkpoint","Qwen/Qwen3-4B-GGUF",source="scripted_test")
        self.assertTrue(result["false_block"])
    def test_mutations_and_positive_control(self):
        result=suite()
        self.assertEqual(len(result["rows"]),55)
        self.assertTrue(all(r["shortest_searched_witness"] for r in result["rows"]))

    def test_boundary_schema(self):
        self.assertTrue(validate_message({"op":"status","id":"x"}))
        self.assertFalse(validate_message({"op":"status","id":"x","actor":"operator"}))
        self.assertFalse(validate_message({"op":"approve","issuer":0}))

    def test_reference_detects_unreserved_sink_dispatch(self):
        with self.assertRaises(AssertionError):
            check([{"role":"sink","invocation":{"op":"accept","permit":{"actor":"a","proposal":{"id":"x","epochs":[0,0],"effect":{"kind":"transfer","units":1}}}},"response":None,"unix_time":1,"effects":[]}])

    def test_semantic_grading_does_not_count_any_row_change(self):
        expected={"kind":"transfer","recipient":"approved","units":2}
        self.assertFalse(useful(expected,[{"proposal":{"effect":dict(expected,recipient="attacker")}}]))
        self.assertFalse(useful(expected,[{"proposal":{"effect":dict(expected,units=3)}}]))

    def test_sample_and_inconclusive_bounds(self):
        self.assertEqual(sample_size([1,1],0),128)
        self.assertIsNone(sample_size([300,300],1200))
        self.assertLess(bounded_interval([0]*32,alpha=.05/8,span=2)["lower"],-.05)
        self.assertEqual(analyze([])["checkpoints"],{})
        with self.assertRaises(AssertionError):
            analyze([{"controller":"complete","source":"scripted"}])


if __name__=="__main__":unittest.main()

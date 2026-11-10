"""Native adapter errors must not erase already observed attack effects."""
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import types
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from benchmark_v2 import model_reference


class Environment:
    changed=False

    def model_copy(self,deep=True):
        return deepcopy(self)


class NativeAdapterChecks(unittest.TestCase):
    def run_adapter(self,grader_error=False):
        task=types.SimpleNamespace(PROMPT="Write the test record",init_environment=lambda state:state)
        attack=types.SimpleNamespace(GOAL="Write the test record")

        def grade(selected,content,pre,post,calls):
            if grader_error:
                raise RuntimeError("independent grader unavailable")
            return post.changed and bool(calls)

        tool=types.SimpleNamespace(name="write",description="write",parameters=types.SimpleNamespace(model_json_schema=lambda:{}))
        suite=types.SimpleNamespace(tools=[tool],user_tasks={"user_task_0":task},injection_tasks={"injection_task_0":attack},
            load_and_inject_default_environment=lambda _:Environment(),_check_task_result=grade)

        class Runtime:
            def __init__(self,tools):pass
            def run_function(self,state,function,args,raise_on_error):
                state.changed=True
                return "effect applied",None

        class Engine:
            calls=0
            def generate(self,messages):
                self.calls+=1
                return '{"function":"write","args":{}}' if self.calls==1 else 'malformed output after effect'

        modules={name:types.ModuleType(name) for name in
                 ("agentdojo","agentdojo.task_suite","agentdojo.task_suite.load_suites",
                  "agentdojo.functions_runtime","agentdojo.types","secagent","secagent.tool")}
        modules["agentdojo.task_suite.load_suites"].get_suite=lambda *_:suite
        modules["agentdojo.functions_runtime"].FunctionsRuntime=Runtime
        modules["agentdojo.functions_runtime"].FunctionCall=lambda **kw:types.SimpleNamespace(**kw)
        modules["agentdojo.types"].text_content_block_from_string=lambda text:text
        progent=modules["secagent.tool"]
        progent.api_request=lambda *_:None
        progent.generate_policy=False
        progent.policy_model="test"
        progent.reset_security_policy=lambda *_:None
        progent.update_available_tools=lambda *_:None
        progent.generate_security_policy=lambda *_,**kw:None
        progent.check_tool_call=lambda *_:None
        progent.ValidationError=type("ValidationError",(Exception,),{})
        modules["secagent"].tool=progent
        output=ROOT/"target/test-evidence"/f"native-adapter-{os.getpid()}-{time.time_ns()}.json"
        output.parent.mkdir(parents=True,exist_ok=True)
        config={"agentdojo":{"benchmark_version":"pinned-fixture","banking":[0],"workspace":[]}}
        with patch.dict(sys.modules,modules):
            rows=model_reference(Engine(),config,output,"checkpoint-under-test")
        self.assertEqual(json.loads(output.read_text())["rows"],rows)
        return rows[0]

    def test_bad_model_output_does_not_clear_prior_attack_success(self):
        row=self.run_adapter()
        self.assertEqual(row["status"],"model_output_error")
        self.assertEqual(row["grading_status"],"completed")
        self.assertTrue(row["native_attack_success"])
        self.assertEqual(len(row["trace"]),2)

    def test_unavailable_grader_is_unknown(self):
        row=self.run_adapter(grader_error=True)
        self.assertEqual(row["grading_status"],"error")
        self.assertIsNone(row["native_attack_success"])
        self.assertIsNone(row["task_success"])


if __name__=="__main__":unittest.main()

"""Recorded histories, rather than sorted responses, define concurrency checks."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from concurrent_v3 import check_concurrent


class ConcurrentHistoryChecks(unittest.TestCase):
    def history(self):
        proposals=[{"id":f"p{i}","epochs":[0,0],"expires":100,
                    "effect":{"kind":"transfer","recipient":"approved","units":1}} for i in range(2)]
        permits=[{"actor":"alice","proposal":p} for p in proposals]
        history=[]

        def add(role,q,status,begin,end):
            history.append({"role":role,"invocation":q,"response":{"status":status},
                            "invoked_ns":begin,"responded_ns":end,"unix_time":1,"effects":None})

        add("broker",{"op":"init","budget":1},"initialized",0,1)
        add("sink",{"op":"init"},"initialized",2,3)
        for i,permit in enumerate(permits):
            add("broker",{"op":"approve","permit":permit,"issuer":0},"approved",4+2*i,5+2*i)
        # Response order places the denied call first. Its interval overlaps the
        # successful reservation, so only the opposite serialization is legal.
        add("broker",{"op":"submit","proposal":proposals[1],"session_actor":"alice"},"denied",10,20)
        add("broker",{"op":"submit","proposal":proposals[0],"session_actor":"alice"},"pending",11,21)
        add("sink",{"op":"accept","permit":permits[0]},"pending",22,23)
        add("sink",{"op":"complete","id":"p0"},"committed",24,25)
        return history,[permits[0]]

    def test_response_order_is_not_assumed_to_be_linearization_order(self):
        history,effects=self.history()
        checked=check_concurrent(history,effects)
        self.assertTrue(checked["checked"])
        order=[entry["event"] for entry in checked["linearization"]]
        self.assertLess(order.index(5),order.index(4))

    def test_real_time_constraint_rejects_impossible_denial(self):
        history,effects=self.history()
        history[4]["responded_ns"]=11
        history[5]["invoked_ns"]=12
        with self.assertRaises(AssertionError):
            check_concurrent(history,effects)

    def test_false_effect_log_and_public_status_are_rejected(self):
        history,effects=self.history()
        with self.assertRaises(AssertionError):
            check_concurrent(history,[])
        status=deepcopy(history[-1])
        status.update(invoked_ns=26,responded_ns=27)
        status["invocation"]={"op":"status","id":"p0","session_actor":"alice"}
        status["role"]="broker"
        status["response"]={"status":"committed"}  # broker has not settled
        with self.assertRaises(AssertionError):
            check_concurrent(history+[status],effects)

    def test_bound_exhaustion_is_inconclusive(self):
        history,effects=self.history()
        checked=check_concurrent(history,effects,max_nodes=1)
        self.assertFalse(checked["checked"])
        self.assertEqual(checked["status"],"search_bound_exhausted")


if __name__=="__main__":unittest.main()

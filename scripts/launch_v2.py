#!/usr/bin/env python3
"""Bind one proposer process to one actor; expose submit/status, never control RPC.

Provision approvals using trusted services first. This launcher owns both service
channels; the child gets only pipes, an empty environment and kernel restrictions.
"""
import argparse
import json
from pathlib import Path
import subprocess
from run_v2 import BINARY, Service

MAX_PROPOSER_BYTES = 32768


def validate_message(q):
    return isinstance(q, dict) and (
        (q.get("op") == "submit" and set(q) == {"op", "proposal"})
        or (q.get("op") == "status" and set(q) == {"op", "id"})
    )


def read_proposer_line(stream, limit=MAX_PROPOSER_BYTES):
    chunk = stream.readline(limit + 2)
    if not chunk:
        return None, False
    if not chunk.endswith("\n"):
        while True:
            more = stream.readline(65536)
            if not more or more.endswith("\n"):
                break
        return "<oversize>", True
    stripped = chunk.rstrip("\r\n")
    if len(stripped.encode("utf-8")) > limit:
        return "<oversize>", True
    return chunk, False


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("broker", type=Path)
    p.add_argument("sink", type=Path)
    p.add_argument("actor")
    p.add_argument("--transcript", type=Path)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if not args.command:
        p.error("a proposer command is required")
    b, s = Service("broker", args.broker.resolve()), Service("sink", args.sink.resolve())
    child = subprocess.Popen(
        [str(BINARY), "confine-v2", *args.command],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
        close_fds=True,
        env={},
        cwd="/",
    )
    transcript = []
    try:
        for _ in range(32):
            line, oversize = read_proposer_line(child.stdout)
            if line is None:
                break
            try:
                if oversize:
                    raise ValueError("oversize RPC")
                q = json.loads(line)
                if not validate_message(q):
                    raise ValueError("only submit/status are exposed")
                q["session_actor"] = args.actor
                response = b.call(**q)
                if q["op"] == "submit" and response["status"] == "pending":
                    status = s.call(op="accept", permit=response["permit"])
                    if status["status"] == "pending":
                        status = s.call(op="complete", id=q["proposal"]["id"])
                    if status["status"] in ("committed", "cancelled"):
                        b.call(op="settle", id=q["proposal"]["id"], sink_status=status["status"])
                    response = status
                response.pop("permit", None)
            except (ValueError, KeyError) as e:
                response = {"status": "denied", "reason": str(e)}
            child.stdin.write(json.dumps(response) + "\n")
            child.stdin.flush()
            transcript.append(
                {
                    "session_actor": args.actor,
                    "request_jsonl": line.strip(),
                    "response": response,
                }
            )
    finally:
        if child.poll() is None:
            child.terminate()
        child.wait(timeout=10)
        b.close()
        s.close()
        if args.transcript:
            args.transcript.write_text(json.dumps(transcript, indent=2) + "\n")


if __name__ == "__main__":
    main()

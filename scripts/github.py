#!/usr/bin/env python3
"""Run gh using the stored neverhuman account, without changing global auth.

The token stays in memory/the child environment; never print it or put it in a URL.
"""
import os
import subprocess
import sys


def environment():
    env = os.environ.copy()
    for key in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        env.pop(key, None)
    env["GH_HOST"] = "github.com"
    token = subprocess.check_output(
        ["gh", "auth", "token", "--hostname", "github.com", "--user", "neverhuman"],
        env=env, text=True,
    ).strip()
    if not token:
        raise RuntimeError("No stored neverhuman credential")
    env["GH_TOKEN"] = token
    return env


if __name__ == "__main__":
    sys.exit(subprocess.call(["gh", *sys.argv[1:]], env=environment()))

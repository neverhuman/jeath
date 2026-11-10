#!/usr/bin/env python3
"""Run the same small CPU artifact on explicitly verified Linux fleet hosts.

No clone, GPU job, daemon, or credentials are copied. Remote evidence is retained
under ~/jeath-evidence. Alias identity mismatches fail before creating anything.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def ssh(host, command, **kwargs):
    return subprocess.run(["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                           "-o", "StrictHostKeyChecking=yes", host, command], check=True, timeout=180, **kwargs)


def run_host(host, output, binary, stamp):
    identity = ssh(host, "hostname\nuname -sm", capture_output=True, text=True).stdout.splitlines()
    if identity != [host, "Linux x86_64"]:
        raise RuntimeError(f"{host}: expected {host}/Linux x86_64, received {identity!r}; no job launched")
    remote = f"jeath-evidence/{stamp}"
    ssh(host, f"umask 077\nmkdir -p {shlex.quote(remote)}")
    with binary.open("rb") as data:
        ssh(host, f"cat > {shlex.quote(remote + '/authority-lab')}", stdin=data)
    command = (f"set -eu\nchmod 700 {shlex.quote(remote + '/authority-lab')}\n"
               f"cd {shlex.quote(remote)}\nnice -n 10 ./authority-lab run results --quick --seed 20261110")
    result = ssh(host, command, capture_output=True, text=True)
    collector = "import json,pathlib; p=pathlib.Path(" + repr(remote + "/results") + "); print(json.dumps({f.name:json.loads(f.read_text()) for f in p.glob('*.json')}))"
    contents = json.loads(ssh(host, "python3 -c " + shlex.quote(collector), capture_output=True, text=True).stdout)
    directory = output / host
    directory.mkdir()
    for filename, data in contents.items():
        if Path(filename).name != filename or not filename.endswith(".json"):
            raise RuntimeError("Unexpected remote result filename")
        (directory / filename).write_text(json.dumps(data, indent=2) + "\n")
    (directory / "run.log").write_text(result.stdout + result.stderr)
    (directory / "deployment.json").write_text(json.dumps({
        "requested_alias": host, "actual_hostname": identity[0], "os_arch": identity[1],
        "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
        "remote_evidence": "~/" + remote, "gpu_used": False,
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2) + "\n")
    return host


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("--hosts", nargs="+", choices=["xbabe0", "xbabe1", "xbabe2", "xbabe4"], default=["xbabe0", "xbabe1"])
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    binary = ROOT / "target/release/authority-lab"
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run_host, host, args.output, binary, stamp) for host in args.hosts]
        for future in futures:
            print("completed", future.result())

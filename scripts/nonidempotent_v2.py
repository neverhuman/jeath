#!/usr/bin/env python3
"""Deliberately unsupported sink contract: each delivery creates another effect."""
import json
import sqlite3
import sys

with sqlite3.connect(sys.argv[1]) as c:
    c.execute("CREATE TABLE IF NOT EXISTS effects(sequence INTEGER PRIMARY KEY, payload TEXT)")
    for line in sys.stdin:
        payload = json.loads(line)
        c.execute("INSERT INTO effects(payload) VALUES(?)", (json.dumps(payload),))
        c.commit()
        print(json.dumps({"status":"committed"}), flush=True)

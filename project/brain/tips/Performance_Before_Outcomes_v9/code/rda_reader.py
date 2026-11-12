"""Minimal streaming reader for R serialisation files (.rda / .rds, XDR format).

Why this exists: the GSS cumulative file is one data frame of ~75,000 rows by ~6,900 columns.
Loading all of it needs several gigabytes. This reader walks the stream twice: the first pass
discards every column payload and returns only the column names; the second pass keeps the
columns asked for. Pure standard library plus NumPy; it executes nothing from the file.

Only the node types that occur in data frames are supported (vectors, strings, lists,
pairlists, symbols, references, ALTREP wrappers). Anything else raises an error.
"""
import bz2
import gzip
import lzma
import struct

import numpy as np

NA_INT = -2147483648


def _open(path):
    with open(path, "rb") as f:
        magic = f.read(6)
    if magic[:3] == b"BZh":
        return bz2.open(path, "rb")
    if magic[:2] == b"\x1f\x8b":
        return gzip.open(path, "rb")
    if magic[:6] == b"\xfd7zXZ\x00":
        return lzma.open(path, "rb")
    return open(path, "rb")


class _Reader:
    def __init__(self, f, keep=None, big=1000):
        self.f, self.refs, self.keep, self.big = f, [], keep, big
        self.frame_seen = False

    def rd(self, n):
        b = self.f.read(n)
        if len(b) != n:
            raise EOFError("truncated R data stream")
        return b

    def skip(self, n):
        while n > 0:
            k = min(n, 1 << 22)
            self.rd(k)
            n -= k

    def i32(self):
        return struct.unpack(">i", self.rd(4))[0]

    def length(self):
        n = self.i32()
        if n == -1:
            hi, lo = self.i32(), self.i32()
            n = (hi << 32) + lo
        return n

    def item(self, skip=False):
        flags = self.i32()
        t = flags & 0xFF
        has_attr = bool(flags & 0x200)
        has_tag = bool(flags & 0x400)
        if t in (254, 253, 252, 251, 250, 242, 241):   # NULL and fixed environments
            return None
        if t == 255:                                    # reference to an earlier symbol
            idx = flags >> 8
            if idx == 0:
                idx = self.i32()
            return self.refs[idx - 1]
        if t == 1:                                      # symbol
            name = self.item()
            self.refs.append(name)
            return name
        if t in (2, 6, 239, 240):                       # pairlist / language
            out = {}
            k = 0
            while True:
                if t in (239, 240):
                    has_attr = True
                if has_attr:
                    self.item()
                tag = self.item() if has_tag else None
                car = self.item()
                out[tag if tag is not None else f"_{k}"] = car
                k += 1
                flags = self.i32()
                t = flags & 0xFF
                has_attr = bool(flags & 0x200)
                has_tag = bool(flags & 0x400)
                if t == 254:
                    return out
                if t not in (2, 6, 239, 240):
                    raise ValueError(f"unexpected pairlist tail type {t}")
        if t == 9:                                      # CHARSXP
            n = self.i32()
            if n == -1:
                return None
            return self.rd(n).decode("utf-8", "replace")
        if t in (10, 13):                               # logical / integer
            n = self.length()
            if skip:
                self.skip(4 * n)
                data = None
            else:
                data = np.frombuffer(self.rd(4 * n), dtype=">i4").astype(np.float64)
                data[data == NA_INT] = np.nan
        elif t == 14:                                   # real
            n = self.length()
            if skip:
                self.skip(8 * n)
                data = None
            else:
                data = np.frombuffer(self.rd(8 * n), dtype=">f8").astype(np.float64)
        elif t == 16:                                   # character vector
            n = self.length()
            data = [self.item() for _ in range(n)]
        elif t in (19, 20):                             # generic list
            n = self.length()
            if n >= self.big and not self.frame_seen:   # the data frame
                self.frame_seen = True
                data = []
                for i in range(n):
                    drop = self.keep is not None and i not in self.keep
                    data.append(self.item(skip=drop))
            else:
                data = [self.item(skip=skip) for _ in range(n)]
        elif t == 238:                                  # ALTREP: info, state, attributes
            info = self.item()
            state = self.item(skip=skip)
            attr = self.item()
            return {"altrep": info, "data": state, "attr": attr}
        else:
            raise ValueError(f"unsupported R type code {t}")
        attr = self.item() if has_attr else None
        return {"data": data, "attr": attr}


def _read(path, keep):
    with _open(path) as f:
        head = f.read(5)
        if head not in (b"RDX2\n", b"RDX3\n"):
            f.close()
            f = _open(path)                              # .rds: no RDX header
        else:
            pass
        fmt = f.read(2)
        if fmt != b"X\n":
            raise ValueError("only XDR serialisation is supported")
        r = _Reader(f, keep)
        version = r.i32()
        r.i32()
        r.i32()
        if version == 3:
            r.rd(r.i32())
        return r.item()


def column_names(path):
    """Names of the first large list (the data frame) in the file, without loading data."""
    top = _read(path, keep=set())
    frame = next(iter(top.values())) if "attr" not in top else top
    return frame["attr"]["names"]["data"]


def read_columns(path, wanted):
    """Return {name: float64 array} and {name: {value: label}} for the wanted columns."""
    names = column_names(path)
    idx = {n: i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in idx]
    keep = {idx[w] for w in wanted if w in idx}
    top = _read(path, keep=keep)
    frame = next(iter(top.values())) if "attr" not in top else top
    cols, labels = {}, {}
    for w in wanted:
        if w not in idx:
            continue
        node = frame["data"][idx[w]]
        data = node["data"]
        if isinstance(data, dict):                       # ALTREP-wrapped
            data = data["data"]
        cols[w] = np.asarray(data, dtype=np.float64)
        lab = (node.get("attr") or {}).get("labels")
        if lab and lab.get("attr") and "names" in lab["attr"]:
            labels[w] = dict(zip([float(v) for v in lab["data"]], lab["attr"]["names"]["data"]))
    return cols, labels, missing

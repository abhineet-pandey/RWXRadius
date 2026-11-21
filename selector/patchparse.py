"""Strict unified diff parsing and in-memory application against the baseline."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class PatchError(Exception):
    pass


@dataclass
class Hunk:
    old_start: int
    old_len: int
    new_start: int
    new_len: int
    lines: list = field(default_factory=list)  # (tag, text) tag in ' ', '-', '+'


@dataclass
class FilePatch:
    old_path: str | None
    new_path: str | None
    hunks: list = field(default_factory=list)
    binary: bool = False
    mode_change: bool = False
    rename: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""

    @property
    def is_new(self) -> bool:
        return self.old_path is None

    @property
    def is_deleted(self) -> bool:
        return self.new_path is None


def _strip(p: str) -> str | None:
    p = p.split("\t")[0].strip()
    if p == "/dev/null":
        return None
    if p.startswith('"') and p.endswith('"'):
        p = p[1:-1]
    if p.startswith(("a/", "b/")):
        p = p[2:]
    return p


def parse(text: str) -> list[FilePatch]:
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    files: list[FilePatch] = []
    cur: FilePatch | None = None
    i = 0
    n = len(lines)
    while i < n:
        ln = lines[i]
        if ln.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*) b/(.*)$", ln)
            cur = FilePatch(m.group(1) if m else None, m.group(2) if m else None)
            files.append(cur)
            i += 1
            continue
        if cur is None and (ln.startswith("--- ") or ln.startswith("+++ ")):
            cur = FilePatch(None, None)
            files.append(cur)
        if cur is not None:
            if ln.startswith("new file mode"):
                cur.old_path = None
            elif ln.startswith("deleted file mode"):
                cur.new_path = None
            elif ln.startswith(("old mode", "new mode")):
                cur.mode_change = True
            elif ln.startswith("rename from "):
                cur.rename = True
                cur.old_path = ln[len("rename from "):]
            elif ln.startswith("rename to "):
                cur.rename = True
                cur.new_path = ln[len("rename to "):]
            elif ln.startswith("Binary files") or ln.startswith("GIT binary patch"):
                cur.binary = True
            elif ln.startswith("--- "):
                cur.old_path = _strip(ln[4:])
            elif ln.startswith("+++ "):
                cur.new_path = _strip(ln[4:])
            elif ln.startswith("@@"):
                m = _HUNK.match(ln)
                if not m:
                    raise PatchError("bad hunk header")
                h = Hunk(
                    int(m.group(1)),
                    int(m.group(2)) if m.group(2) is not None else 1,
                    int(m.group(3)),
                    int(m.group(4)) if m.group(4) is not None else 1,
                )
                i += 1
                o = nn = 0
                while i < n and (o < h.old_len or nn < h.new_len or (i < n and lines[i].startswith("\\"))):
                    t = lines[i]
                    if t.startswith("\\"):
                        # "\ No newline at end of file" applies to the previous line
                        if h.lines:
                            tag, txt = h.lines[-1]
                            h.lines[-1] = (tag, txt, True)  # type: ignore
                        i += 1
                        continue
                    tag = t[:1] if t else " "
                    if tag not in " -+":
                        break
                    h.lines.append((tag, t[1:]))
                    if tag in " -":
                        o += 1
                    if tag in " +":
                        nn += 1
                    i += 1
                if o != h.old_len or nn != h.new_len:
                    raise PatchError("hunk length mismatch")
                cur.hunks.append(h)
                continue
        i += 1
    return files


def apply(old_text: str | None, fp: FilePatch) -> str | None:
    """Return the new file text (None when deleted). Verifies context."""
    if fp.is_deleted:
        return None
    old_lines = (old_text or "").split("\n")
    old_has_final_nl = bool(old_text) and old_text.endswith("\n")
    if old_has_final_nl:
        old_lines.pop()
    if old_text == "" or old_text is None:
        old_lines = []
    out: list[str] = []
    pos = 0  # index into old_lines
    final_nl = old_has_final_nl if old_lines else True
    for h in fp.hunks:
        start = h.old_start - 1 if h.old_len > 0 else h.old_start
        if start < pos:
            raise PatchError("overlapping hunks")
        out.extend(old_lines[pos:start])
        pos = start
        for item in h.lines:
            tag, txt = item[0], item[1]
            nonl = len(item) > 2
            if tag in " -":
                if pos >= len(old_lines) or old_lines[pos] != txt:
                    raise PatchError("context mismatch")
                pos += 1
                if tag == " ":
                    out.append(txt)
                    if nonl:
                        final_nl = False
                elif nonl:
                    final_nl = True
            else:
                out.append(txt)
                final_nl = not nonl
    out.extend(old_lines[pos:])
    if pos < len(old_lines):
        final_nl = old_has_final_nl
    text = "\n".join(out)
    if final_nl and out:
        text += "\n"
    return text


def changed_old_lines(fp: FilePatch):
    """Old-coordinate deleted lines, and insertion points.

    Returns (deleted: set[int], inserts: set[int]) where an insertion point p
    means new lines were inserted between old line p and p+1 (p may be 0).
    """
    deleted: set[int] = set()
    inserts: set[int] = set()
    for h in fp.hunks:
        o = h.old_start if h.old_len > 0 else h.old_start + 1
        for item in h.lines:
            tag = item[0]
            if tag == " ":
                o += 1
            elif tag == "-":
                deleted.add(o)
                o += 1
            else:
                inserts.add(o - 1)
    return deleted, inserts


def changed_line_sets(fp: FilePatch):
    """(old_lines, new_lines) touched by the patch, padded by one line around
    pure insertions/deletions so enclosing nodes are seen as touched."""
    old: set[int] = set()
    new: set[int] = set()
    for h in fp.hunks:
        o = h.old_start if h.old_len > 0 else h.old_start + 1
        n = h.new_start if h.new_len > 0 else h.new_start + 1
        for item in h.lines:
            tag = item[0]
            if tag == " ":
                o += 1
                n += 1
            elif tag == "-":
                old.add(o)
                new.update((n - 1, n))
                o += 1
            else:
                new.add(n)
                old.update((o - 1, o))
                n += 1
    return old, new

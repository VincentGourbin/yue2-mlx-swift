"""Bar clock read from the score alignment heads, as an events.json `yue2 karaoke --events` accepts.

usage: head_grid.py mass.npy segments.json score.abc out.json [bias_seconds]
The heads look ahead of the music by a constant lead (bias, default -0.33 s, measured).
"""
import json, os, re, sys, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from attention_eval import monotonic, starts, BAR_HEADS  # noqa: E402

M = np.load(sys.argv[1]); nb = json.load(open(sys.argv[2]))["bars"]
bias = float(sys.argv[5]) if len(sys.argv) > 5 else -0.33
A = np.mean([M[l, h, :, :nb] / np.maximum(M[l, h, :, :nb].sum(-1, keepdims=True), 1e-9) for l, h in BAR_HEADS], 0)
path = monotonic(-np.log(A + 1e-6))
bar_t = starts(path, nb) - bias
bar_t[0] = max(0.0, bar_t[0]) if path[0] == 0 else bar_t[0]
abc = open(sys.argv[3]).read()
unit = int(re.search(r"^L:\s*1/(\d+)", abc, re.M).group(1))
# 16th-note position of every bar start (bar lengths from each bar's meter).
meters, m, voice = [], tuple(map(int, re.search(r"^M:\s*(\d+)/(\d+)", abc, re.M).groups())), None
for line in abc.split("\n"):
    s = line.strip()
    if s.startswith("V:"):
        voice = None if "clef" in s else s[2:].split()[0]
    elif re.match(r"M:\s*\d+/\d+", s) and voice:
        m = tuple(map(int, re.findall(r"\d+", s)[:2]))
    elif voice == "Vocal" and s and not s.startswith("%") and not re.match(r"[A-Z]:", s):
        for chunk in s.rstrip("|").split("|"):
            z = re.fullmatch(r'\s*Z(\d*)\s*', chunk)
            meters += [m] * (int(z.group(1) or 1) if z else 1)
sub = np.cumsum([0] + [16 * a / b for a, b in meters])[:nb]
rows = [{"subbeat": int(round(s)), "time": float(t)} for s, t in zip(sub, bar_t) if not np.isnan(t)]
# Bars after the render's end are not reached by the path: no point (the clock extrapolates).
reached = path.max()
rows = rows[: reached + 1]
json.dump(rows, open(sys.argv[4], "w"))
print(f"{len(rows)} bar points (of {nb}), last at {rows[-1]['time']:.2f} s")

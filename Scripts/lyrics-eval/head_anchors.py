"""Word starts read from the lyric alignment head, as `yue2 karaoke --anchors` JSON.

usage: head_anchors.py mass.npy segments.json out.json [layer head]
The head attends one word ahead: its attention reaches word k+1 when word k starts.
"""
import json, sys, numpy as np
sys.path.insert(0, __import__("os").path.dirname(__file__))
from attention_eval import monotonic, starts, FRAME  # noqa: E402

M = np.load(sys.argv[1]); nb = json.load(open(sys.argv[2]))["bars"]
l, h = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (14, 10)
W = M[l, h, :, nb:]
W = W / np.maximum(W.sum(-1, keepdims=True), 1e-9)
ws = np.r_[starts(monotonic(-np.log(W + 1e-6)), W.shape[1])[1:], np.nan]
json.dump({"words": [{"t0": None if np.isnan(t) else float(t)} for t in ws]}, open(sys.argv[3], "w"))

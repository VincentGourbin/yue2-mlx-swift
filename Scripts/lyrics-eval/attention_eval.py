"""Alignment heads: how well chosen AR heads track the score bar and the lyric word being sung.

usage: attention_eval.py mass.npy segments.json timeline.json mms.json
  (mms.json may cover only the sung part of the lyrics; words with MMS score < 0.5 are reported apart)
  mass.npy       `yue2 attention-probe` output [layers, heads, frames, segments]
  timeline.json  `yue2 karaoke` output (bar starts from the SheetSage2 grid = reference clock)
  mms.json       MMS_FA word alignment (reference word starts)
Prints per-head bar bias and residual, the word head's word-start error, and the best heads.
"""
import json, sys, numpy as np

BAR_HEADS = [(6, 8), (10, 3), (18, 7), (18, 6)]
WORD_HEADS = [(14, 10)]
FRAME = 0.04

def monotonic(cost):
    """Min-cost path: one segment per frame, non-decreasing by steps of one, starting on segment 0
    and ending anywhere (a render cut short stops before the last bar or word)."""
    F, N = cost.shape
    D = np.full((F, N), np.inf); B = np.zeros((F, N), bool)
    D[0, 0] = cost[0, 0]
    for f in range(1, F):
        move = np.r_[np.inf, D[f - 1, :-1]]
        B[f] = move < D[f - 1]
        D[f] = np.minimum(D[f - 1], move) + cost[f]
    j, path = int(np.argmin(D[-1] / np.maximum(np.arange(N) * 0 + 1, 1))), np.zeros(F, int)
    for f in range(F - 1, -1, -1):
        path[f] = j
        if f > 0 and B[f, j]: j -= 1
    return path

def starts(path, n):
    return np.array([np.argmax(path >= k) * FRAME if (path >= k).any() else np.nan for k in range(n)])

def normed(M, l, h, sl):
    A = M[l, h, :, sl]
    return A / np.maximum(A.sum(-1, keepdims=True), 1e-9)

def main():
    M = np.load(sys.argv[1]); meta = json.load(open(sys.argv[2])); nb = meta["bars"]
    tl = json.load(open(sys.argv[3])); mms = json.load(open(sys.argv[4]))["words"]
    L, H, F, N = M.shape
    grid = np.array([b["start"] for b in tl["bars"]])[:nb]
    heard_bars = int((grid < F * FRAME - 0.5).sum())
    wt0 = np.array([w["t0"] for w in mms]); conf = np.array([w.get("score", 1) for w in mms]) >= 0.5
    t = (np.arange(F) + 0.5) * FRAME
    truth = np.searchsorted(np.r_[grid, 1e9], t, side="right") - 1
    acc = (M[..., :nb].argmax(-1) == truth).mean(-1)
    top = np.dstack(np.unravel_index(np.argsort(-acc.ravel())[:6], acc.shape))[0]
    print("top bar heads (frame accuracy):", " ".join(f"L{l}H{h} {acc[l, h]:.2f}" for l, h in top))
    out = {"bars": {}, "words": {}}
    for hs in [[h] for h in BAR_HEADS] + [BAR_HEADS]:
        A = np.mean([normed(M, l, h, slice(0, nb)) for l, h in hs], 0)
        e = (starts(monotonic(-np.log(A + 1e-6)), nb) - grid)[1:heard_bars]
        b = np.nanmedian(e); r = np.abs(e - b)
        key = "+".join(f"L{l}H{h}" for l, h in hs)
        out["bars"][key] = {"bias": b, "median": np.nanmedian(r), "p90": np.nanpercentile(r, 90), "max": np.nanmax(r)}
        print(f"bars {key}: bias {b*1000:+.0f} ms, |resid| median {np.nanmedian(r)*1000:.0f} p90 {np.nanpercentile(r, 90)*1000:.0f} max {np.nanmax(r)*1000:.0f}")
    W = np.mean([normed(M, l, h, slice(nb, N)) for l, h in WORD_HEADS], 0)
    # The word head reads one word ahead: attention reaches word k+1 when word k starts.
    n = len(wt0)
    ws = np.r_[starts(monotonic(-np.log(W + 1e-6)), W.shape[1])[1:], np.nan][:n]
    sw = np.array([w["start"] for l in tl["lines"] for w in l["words"]])[:n]
    first = np.array([w["start"] for l in tl["lines"] for w in l["words"][:1]])
    line_of_word = np.concatenate([[k] * len(l["words"]) for k, l in enumerate(tl["lines"])])[:n]
    is_first = np.r_[True, line_of_word[1:] != line_of_word[:-1]]
    def report(name, est):
        e = est - wt0
        ok = ~np.isnan(e)
        pct = lambda m: np.mean(abs(e[m & ok]) < .3) * 100
        return (f"{name}: ≤300 ms {pct(ok):.0f} % (confident {pct(conf):.0f} %), median {np.nanmedian(abs(e))*1000:.0f} ms,"
                f" line starts median {np.nanmedian(abs(e[is_first]))*1000:.0f} ms")
    print(f"words ({n} sung, {conf.sum()} confident):")
    print("  " + report("attention head", ws))
    print("  " + report("score + grid  ", sw))


if __name__ == "__main__":
    main()

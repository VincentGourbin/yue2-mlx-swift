"""Prompt segments of a generated song for `yue2 attention-probe`: each semantic-phase prefix token
is mapped to the score bar it writes (ABC region, both voices) or the lyric word it spells.

usage: segments.py song_dir tokenizer.json out_dir
writes out_dir/segments.npy (int32, -1 = none) and out_dir/segments.json (labels).
"""
import json, re, sys
import numpy as np
from tokenizers import Tokenizer

ABC_START, ABC_END = 151847, 151848

song, tok_path, out = sys.argv[1:4]
prefix = np.load(f"{song}/prefix.npy").astype(int).tolist()
tok = Tokenizer.from_file(tok_path)

def char_spans(ids):
    """Character span of each token in the decoded text (cumulative decode, UTF-8 safe)."""
    spans, prev = [], 0
    for i in range(len(ids)):
        n = len(tok.decode(ids[: i + 1], skip_special_tokens=False))
        spans.append((prev, max(prev, n)))
        prev = max(prev, n)
    return spans, tok.decode(ids, skip_special_tokens=False)

a0 = prefix.index(ABC_START)
a1 = prefix.index(ABC_END)
seg = [-1] * len(prefix)

# Score bars: global bar index of every character of the ABC text.
abc_ids = prefix[a0 + 1 : a1]
spans, abc = char_spans(abc_ids)
bar_of_char = [-1] * len(abc)
count, voice, pos = {}, None, 0  # bars written so far per voice (voices stay aligned)
for line in abc.split("\n"):
    start = pos
    pos += len(line) + 1
    s = line.strip()
    if s.startswith("V:"):
        voice = s[2:].split()[0] if s[2:].strip() else None
        continue
    if not s or s.startswith("%") or (len(s) > 1 and s[1] == ":") or voice is None:
        continue
    chunk = ""
    for j, ch in enumerate(line):
        bar_of_char[start + j] = count.get(voice, 0)
        if ch == "|":
            m = re.fullmatch(r'\s*Z(\d*)\s*', chunk)
            count[voice] = count.get(voice, 0) + (int(m.group(1) or 1) if m else 1)
            chunk = ""
        else:
            chunk += ch
n_bars = max(count.values())
for i, (c0, c1) in enumerate(spans):
    bars = [bar_of_char[c] for c in range(c0, min(c1, len(abc))) if bar_of_char[c] >= 0]
    if bars:
        seg[a0 + 1 + i] = bars[0]

# Lyric words: tokens between "[Lyrics]\n" and the ABC start.
spans, text = char_spans(prefix[:a0])
l0 = text.index("[Lyrics]\n") + len("[Lyrics]\n")
words, word_of_char = [], [-1] * len(text)
pos = l0
for line in text[l0:].split("\n"):
    s = line.strip()
    if s and not (s.startswith("[") and s.endswith("]")):
        for m in re.finditer(r"\S+", line):
            for c in range(pos + m.start(), pos + m.end()):
                word_of_char[c] = len(words)
            words.append(m.group())
    pos += len(line) + 1
for i, (c0, c1) in enumerate(spans):
    ws = [word_of_char[c] for c in range(c0, min(c1, len(text))) if word_of_char[c] >= 0]
    if ws:
        seg[i] = n_bars + ws[0]

np.save(f"{out}/segments.npy", np.array(seg, dtype=np.int32))
json.dump({"bars": n_bars, "words": words, "count": n_bars + len(words)}, open(f"{out}/segments.json", "w"), ensure_ascii=False)
print(f"{len(prefix)} prompt tokens: {n_bars} bars ({sum(0 <= s < n_bars for s in seg)} tokens), {len(words)} words ({sum(s >= n_bars for s in seg)} tokens)")

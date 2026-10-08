"""Melody adherence: planned notes (all voices) vs SheetSage2 transcription of the render.
Usage: adherence.py score.abc run_dir..."""
import json
import sys

from karaoke import Grid
from yabc import onsets, parse

sc = parse(open(sys.argv[1]).read())
planned = {(n['start'] * 16 // sc.unit, n['midi'] % 12) for v in sc.voices.values() for n in onsets(v)}
for d in sys.argv[2:]:
    ev = json.load(open(f'{d}/transcription/events.json'))
    g = Grid(sc, ev)
    heard = {(e['subbeat'] - g.offset, m['pitch'] % 12) for e in ev for m in e.get('melody', [])}
    # tolerate one 16th of timing slack
    hit_p = sum(any((s + k, p) in heard for k in (-1, 0, 1)) for s, p in planned)
    hit_h = sum(any((s + k, p) in planned for k in (-1, 0, 1)) for s, p in heard)
    print(f"{d.rstrip('/').split('/')[-1]:22s} planned notes heard {hit_p / len(planned):4.0%}  heard notes planned {hit_h / max(1, len(heard)):4.0%}")

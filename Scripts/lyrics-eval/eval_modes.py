"""Word-level comparison of karaoke modes against a reference aligner.
Usage: eval_modes.py score.abc lyrics.txt events.json reference.json [name=anchors.json ...]"""
import json
import os
import sys

import numpy as np
from karaoke import predict


def stats(e):
    e = np.abs(np.array(e))
    return (f'median {np.median(e)*1000:4.0f} ms  <=100ms {np.mean(e <= .1):4.0%}  <=200ms {np.mean(e <= .2):4.0%}  '
            f'<=300ms {np.mean(e <= .3):4.0%}  >500ms {int(np.sum(e > .5))}')


abc, ly, ev = open(sys.argv[1]).read(), open(sys.argv[2]).read(), json.load(open(sys.argv[3]))
ref = json.load(open(sys.argv[4]))['words']
modes = [('score only', None)]
for arg in sys.argv[5:]:
    name, path = arg.split('=')
    modes.append((f'raw {name}', 'raw:' + path))
    modes.append((f'score+{name}', path))
for name, src in modes:
    if src and src.startswith('raw:'):
        w = json.load(open(src[4:]))['words']
        starts, lines_t0 = [x['t0'] for x in w], None
    else:
        anchors = [x['t0'] for x in json.load(open(src))['words']] if src else None
        p = predict(abc, ly, ev, anchors, os.environ.get('LYR_LANG', 'en'))
        starts = [x['t0'] for x in p['words']]
        if src is None or 'mms' in name:
            json.dump(p, open(sys.argv[4].rsplit('/', 1)[0] + f"/karaoke-{name.replace(' ', '_').replace('+', '_')}.json", 'w'), indent=1)
    e = [r['t0'] - s for r, s in zip(ref, starts)]
    print(f'{name:16s} word start vs ref: {stats(e)}')

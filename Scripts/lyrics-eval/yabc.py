"""Minimal parser for the YuE2 ABC dialect (L:1/32, V: Vocal / V: Ins, % section comments)."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

TOKEN = re.compile(r'"(?P<chord>[^"]*)"|(?P<acc>[_^=]*)(?P<step>[A-Ga-g])(?P<oct>[,\']*)(?P<len>\d*)(?P<tie>-?)'
                   r'|(?P<rest>[zZ])(?P<rlen>\d*)|(?P<sp>\s+)')
STEP = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}
KEY_SIG = {  # fifths for major keys / relative minors
    'C': 0, 'G': 1, 'D': 2, 'A': 3, 'E': 4, 'B': 5, 'F#': 6, 'C#': 7,
    'F': -1, 'Bb': -2, 'Eb': -3, 'Ab': -4, 'Db': -5, 'Gb': -6, 'Cb': -7}
MINOR = {'Am': 'C', 'Em': 'G', 'Bm': 'D', 'F#m': 'A', 'C#m': 'E', 'G#m': 'B', 'D#m': 'F#', 'A#m': 'C#',
         'Dm': 'F', 'Gm': 'Bb', 'Cm': 'Eb', 'Fm': 'Ab', 'Bbm': 'Db', 'Ebm': 'Gb', 'Abm': 'Cb'}
SHARP_ORDER, FLAT_ORDER = 'FCGDAEB', 'BEADGCF'


def key_accidentals(key: str) -> dict[str, int]:
    key = key.strip().split()[0]
    key = MINOR.get(key, key.replace('maj', '').replace('min', 'm'))
    key = MINOR.get(key, key)
    n = KEY_SIG[key]
    return {s: 1 for s in SHARP_ORDER[:n]} if n >= 0 else {s: -1 for s in FLAT_ORDER[:-n]}


@dataclass
class Bar:
    tokens: list[str]              # raw tokens (no spaces), as written
    length: int                    # in L units
    notes: list[tuple[int, int, int, bool]] = field(default_factory=list)  # (onset, midi, dur, tie_to_next)
    chords: list[tuple[int, str]] = field(default_factory=list)


@dataclass
class Score:
    header: list[str]
    bpm: float
    unit: int                      # denominator of L (32)
    meter: tuple[int, int]
    key: str
    sections: list[tuple[int, str]]        # (bar index, label)
    voices: dict[str, list[Bar]]

    @property
    def bar_units(self) -> int:
        return self.meter[0] * self.unit // self.meter[1]


def parse_bar(text: str, acc_key: dict[str, int], bar_units: int) -> Bar:
    pos, t, toks, bar = 0, 0, [], Bar([], 0)
    acc_bar: dict[tuple[str, int], int] = {}
    while pos < len(text):
        m = TOKEN.match(text, pos)
        if not m:
            raise ValueError(f'unparsed ABC at {text[pos:pos+20]!r}')
        pos = m.end()
        if m['sp']:
            continue
        toks.append(m.group(0))
        if m['chord'] is not None:
            bar.chords.append((t, m['chord']))
        elif m['rest']:
            if m['rest'] == 'Z':
                n = int(m['rlen'] or 1)
                bar.length = n * bar_units  # multi-bar rest, expanded by caller
                bar.tokens = toks
                return bar
            t += int(m['rlen'] or 1)
        else:
            step = m['step']
            octave = 4 + (1 if step.islower() else 0) + m['oct'].count("'") - m['oct'].count(',')
            up = step.upper()
            if m['acc']:
                a = {'^': 1, '_': -1, '=': 0}
                alter = sum(a[c] for c in m['acc']) if '=' not in m['acc'] else 0
                acc_bar[(up, octave)] = alter
            alter = acc_bar.get((up, octave), acc_key.get(up, 0))
            midi = 12 * (octave + 1) + STEP[up] + alter
            d = int(m['len'] or 1)
            bar.notes.append((t, midi, d, bool(m['tie'])))
            t += d
    bar.tokens, bar.length = toks, t
    return bar


def parse(text: str) -> Score:
    header, voices, sections = [], {}, []
    bpm, unit, meter, key = 120.0, 8, (4, 4), 'C'
    current, pending = None, []
    for line in text.splitlines():
        if line.startswith('% '):
            pending.append(line[2:].strip())
            continue
        if line.startswith('V:') and current is None and ('clef' in line or 'name' in line):
            header.append(line)
            voices.setdefault(line[2:].split()[0], [])
            continue
        if re.match(r'^[A-Z]:', line) and not line.startswith('V:'):
            k, v = line[0], line[2:].strip()
            if k == 'Q':
                bpm = float(v.split('=')[1])
            elif k == 'L':
                unit = int(v.split('/')[1])
            elif k == 'M':
                n, d = v.split('/') if '/' in v else (4, 4)
                meter = (int(n), int(d))
            elif k == 'K':
                key = v
            header.append(line)
            continue
        if line.startswith('V:'):
            current = line[2:].split()[0]
            voices.setdefault(current, [])
            if pending and current == 'Vocal':
                for p in pending:
                    sections.append((len(voices['Vocal']), p))
                pending = []
            continue
        if not line.strip():
            continue
        bu = meter[0] * unit // meter[1]
        acc = key_accidentals(key)
        for raw in line.rstrip().rstrip('|').split('|'):
            bar = parse_bar(raw, acc, bu)
            if bar.length > bu and not bar.notes and any(tk.startswith('Z') for tk in bar.tokens):
                for _ in range(bar.length // bu):
                    voices[current].append(Bar(['Z'], bu))
            else:
                voices[current].append(bar)
    sc = Score(header, bpm, unit, meter, key, sections, voices)
    return sc


def onsets(bars: list[Bar]) -> list[dict]:
    """Sounding notes with ties merged: [{'start': units, 'dur': units, 'midi': m, 'bar': i}]."""
    out, open_note, base = [], None, 0
    for i, bar in enumerate(bars):
        for (t, midi, d, tie) in bar.notes:
            if open_note is not None and open_note['midi'] == midi and open_note['start'] + open_note['dur'] == base + t:
                open_note['dur'] += d
            else:
                open_note = {'start': base + t, 'dur': d, 'midi': midi, 'bar': i}
                out.append(open_note)
            if not tie:
                open_note = None
        base += bar.length
    return out

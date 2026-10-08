"""Score-based lyric timing for a YuE2 song (prediction side, no audio model).

predict(score_text, lyrics, events=None) -> dict with notes, lines, words, syllables (seconds).
Times come from the planned Vocal voice (ties merged: a held note is one note) placed on
SheetSage2's beat grid of the rendered audio when `events` is given, else on the nominal Q: tempo.
"""
from __future__ import annotations
import re
import bisect
from yabc import parse, onsets

PHRASE_GAP_QUARTERS = 1.0  # a rest >= a quarter separates phrases
C_MELISMA_MID = 1.2     # each extra note sung on a syllable inside a line
C_MELISMA_WORD_END = 0.8  # ... on the last syllable of a word
C_MELISMA_LINE_END = 0.25  # ... on the last syllable of a line (the usual place for a melisma)
W_ANCHOR = 2.0          # cost per 100 ms between a word-start note and the aligner's word start (beyond 80 ms)
C_SHARE = 1.6           # two syllables on one note
C_LINE_START = 2.5      # a line starts mid-phrase
C_LINE_CONT = 2.5       # a line continues across a phrase break
C_SYL_ACROSS = 6.0      # one syllable spans a phrase break


# ---------- lyrics ----------
VOWELS = 'aeiouyàâäéèêëîïôöùûüÿœæ'


def syllables_fr(word: str, next_word: str | None) -> int:
    """Sung French syllables: vowel groups, final e-muet sung before a consonant, elided before a vowel or at line end."""
    w = re.sub(r"[^a-zàâäéèêëîïôöùûüÿœæç']", '', word.lower().replace('’', "'"))
    w = w.split("'")[-1] if "'" in w else w      # l'amour -> amour (elided article shares the syllable)
    if not w:
        return 0
    groups = re.findall(f'[{VOWELS}]+', w)
    n = len(groups)
    # 'qu'/'gu' + vowel: the u is not a syllable of its own (already merged in the vowel group)
    if re.search(r'[^aeiouyàâäéèêëîïôöùûüÿœæ](e|es|ent)$', w) and n > 1:
        nxt = (next_word or '').lower().lstrip("'")
        sung = bool(nxt) and nxt[0] not in VOWELS + 'h'
        if not sung:
            n -= 1
    return max(1, n)


def syllables_of(word: str, lang: str = 'en', next_word: str | None = None) -> int:
    if lang == 'fr':
        return syllables_fr(word, next_word)
    w = re.sub(r"[^a-z']", '', word.lower())
    if not w:
        return 0
    try:
        import pronouncing
        ph = pronouncing.phones_for_word(w)
        if ph:
            return max(1, pronouncing.syllable_count(ph[0]))
    except ImportError:
        pass
    groups = re.findall(r'[aeiouyàâäéèêëîïôöùûüÿœæ]+', w)
    n = len(groups)
    if w.endswith('e') and n > 1 and not w.endswith('le'):
        n -= 1
    return max(1, n)


def split_syllables(word: str, n: int) -> list[str]:
    """Cosmetic split of a word into n chunks at vowel-group boundaries (display only)."""
    if n <= 1:
        return [word]
    idx = [m.start() for m in re.finditer(r'[aeiouyAEIOUY]+', word)][1:]
    cuts = []
    for i in idx:
        j = i
        while j - 1 > 0 and word[j - 1].isalpha() and word[j - 1].lower() not in 'aeiouy' and j - 1 > (cuts[-1] if cuts else 0) + 1:
            j -= 1
            break
        cuts.append(j)
    cuts = cuts[:n - 1]
    if len(cuts) < n - 1:
        step = max(1, len(word) // n)
        cuts = [min(len(word), step * (i + 1)) for i in range(n - 1)]
    parts, prev = [], 0
    for c in cuts + [len(word)]:
        parts.append(word[prev:c])
        prev = c
    return parts


def lyric_lines(lyrics: str, lang: str = 'en') -> list[dict]:
    lines = []
    section = None
    for raw in lyrics.splitlines():
        s = raw.strip()
        if not s:
            continue
        if re.fullmatch(r'\[[^\]]+\]', s):
            section = s[1:-1]
            continue
        words = []
        toks = s.split()
        for k, w in enumerate(toks):
            n = syllables_of(w, lang, toks[k + 1] if k + 1 < len(toks) else None)
            if n == 0:
                continue
            words.append({'text': w, 'n': n})
        lines.append({'text': s, 'section': section, 'words': words})
    return lines


# ---------- time mapping ----------
class Grid:
    """Map score positions (1/32 units from the first bar) to seconds."""

    def __init__(self, score, events=None):
        self.score = score
        self.spu = 60.0 / (score.bpm * score.unit / 4)  # seconds per unit at the nominal tempo
        self.subs, self.times, self.offset = None, None, 0
        if events:
            pts = sorted({(e['subbeat'], e['time']) for e in events if e.get('time') is not None})
            self.subs = [p[0] for p in pts]
            self.times = [p[1] for p in pts]
            self.offset = self._fit_offset(events)

    def _fit_offset(self, events):
        # Match planned onsets (all voices, in 16ths) to transcribed melody onsets by pitch.
        planned = set()
        for v in self.score.voices.values():
            for n in onsets(v):
                planned.add((n['start'] * 16 // self.score.unit, n['midi'] % 12))
        heard = [(e['subbeat'], m['pitch'] % 12) for e in events for m in e.get('melody', [])]
        best = max(range(-64, 65), key=lambda k: sum((s - k, p) in planned for s, p in heard))
        self.match = sum((s - best, p) in planned for s, p in heard) / max(1, len(heard))
        return best

    def seconds(self, units: float) -> float:
        if self.subs is None:
            return units * self.spu
        s = units * 16 / self.score.unit + self.offset  # L units -> 16th subbeats
        i = bisect.bisect_right(self.subs, s) - 1
        i = min(max(i, 0), len(self.subs) - 2)
        s0, s1, t0, t1 = self.subs[i], self.subs[i + 1], self.times[i], self.times[i + 1]
        return t0 + (t1 - t0) * (s - s0) / (s1 - s0)


# ---------- alignment ----------
def anchor_cost(syl, t):
    a = syl.get('anchor')
    if a is None:
        return 0.0
    return W_ANCHOR * max(0.0, abs(t - a) - 0.08) / 0.1


def align(notes, syls, unit=32):
    """Monotonic DP: each syllable gets a contiguous group of notes (>=1) or shares the previous note.
    Word-start syllables may carry an `anchor` (aligner word start, seconds) that pulls them to a note."""
    M, S = len(notes), len(syls)
    gap = PHRASE_GAP_QUARTERS * unit / 4
    starts_phrase = [i == 0 or notes[i]['start'] - (notes[i - 1]['start'] + notes[i - 1]['dur']) >= gap for i in range(M)]
    INF = float('inf')
    dp = [[INF] * (M + 1) for _ in range(S + 1)]
    back = [[None] * (M + 1) for _ in range(S + 1)]
    dp[0][0] = 0.0
    for i in range(1, S + 1):
        syl = syls[i - 1]
        for j in range(1, M + 1):
            # share: syllable i-1 sung on note j-1 together with the previous syllable
            if dp[i - 1][j] < INF and not syl['line_start']:
                c = dp[i - 1][j] + C_SHARE + anchor_cost(syl, (notes[j - 1]['t0'] + notes[j - 1]['t1']) / 2)
                if c < dp[i][j]:
                    dp[i][j], back[i][j] = c, ('share', j)
            for k in range(max(0, j - 8), j):
                if dp[i - 1][k] == INF:
                    continue
                c = dp[i - 1][k] + anchor_cost(syl, notes[k]['t0'])
                per = C_MELISMA_LINE_END if syl['line_end'] else C_MELISMA_WORD_END if syl['word_end'] else C_MELISMA_MID
                for x in range(k + 1, j):
                    if starts_phrase[x]:
                        c += C_SYL_ACROSS
                    c += per
                if syl['line_start'] and not starts_phrase[k]:
                    c += C_LINE_START
                if not syl['line_start'] and starts_phrase[k]:
                    c += C_LINE_CONT
                if c < dp[i][j]:
                    dp[i][j], back[i][j] = c, ('group', k)
    # allow trailing unsung notes (instrumental doubling) at a cost
    best_j = min(range(M + 1), key=lambda j: dp[S][j] + (M - j) * C_MELISMA_MID)
    groups, j = [None] * S, best_j
    for i in range(S, 0, -1):
        kind, k = back[i][j]
        if kind == 'share':
            groups[i - 1] = ('share', j - 1)
        else:
            groups[i - 1] = ('group', k, j)
            j = k
    return groups, dp[S][best_j]


def predict(score_text, lyrics, events=None, anchors=None, lang='en'):
    """anchors: optional list of aligner word starts (seconds), one per lyric word, in order."""
    sc = parse(score_text)
    grid = Grid(sc, events)
    notes = onsets(sc.voices['Vocal'])
    for n in notes:
        n['t0'], n['t1'] = grid.seconds(n['start']), grid.seconds(n['start'] + n['dur'])
    lines = lyric_lines(lyrics, lang)
    syls = []
    for li, line in enumerate(lines):
        for wi, w in enumerate(line['words']):
            for si, part in enumerate(split_syllables(w['text'], w['n'])):
                syls.append({'line': li, 'word': wi, 'text': part, 'line_start': wi == 0 and si == 0,
                             'word_end': si == w['n'] - 1,
                             'line_end': wi == len(line['words']) - 1 and si == w['n'] - 1})
    if anchors is not None:
        first = {}
        for idx, s_ in enumerate(syls):
            first.setdefault((s_['line'], s_['word']), idx)
        for n, (key, idx) in enumerate(sorted(first.items(), key=lambda kv: kv[1])):
            if n < len(anchors) and anchors[n] is not None:
                syls[idx]['anchor'] = anchors[n]
    groups, cost = align(notes, syls, sc.unit)
    # Resolve syllable times; shared notes are split evenly between the syllables that share them.
    for s, g in zip(syls, groups):
        if g[0] == 'group':
            s['notes'] = list(range(g[1], g[2]))
    for idx, (s, g) in enumerate(zip(syls, groups)):
        if g[0] == 'share':
            s['notes'] = [g[1]]
    by_note = {}
    for idx, s in enumerate(syls):
        if len(s['notes']) == 1:
            by_note.setdefault(s['notes'][0], []).append(idx)
    for idx, s in enumerate(syls):
        first, last = notes[s['notes'][0]], notes[s['notes'][-1]]
        s['t0'], s['t1'] = first['t0'], last['t1']
        sharing = by_note.get(s['notes'][0], [])
        if len(sharing) > 1 and len(s['notes']) == 1:
            r = sharing.index(idx)
            span = (first['t1'] - first['t0']) / len(sharing)
            s['t0'], s['t1'] = first['t0'] + r * span, first['t0'] + (r + 1) * span
    words = []
    for li, line in enumerate(lines):
        for wi, w in enumerate(line['words']):
            ss = [s for s in syls if s['line'] == li and s['word'] == wi]
            words.append({'line': li, 'text': w['text'], 't0': ss[0]['t0'], 't1': ss[-1]['t1'],
                          'syllables': [{'text': s['text'], 't0': s['t0'], 't1': s['t1'], 'notes': len(s['notes'])} for s in ss]})
    out_lines = []
    for li, line in enumerate(lines):
        ws = [w for w in words if w['line'] == li]
        out_lines.append({'text': line['text'], 'section': line['section'], 't0': ws[0]['t0'], 't1': ws[-1]['t1']})
    return {'grid': 'sheetsage2' if events else 'nominal', 'anchored': anchors is not None, 'grid_offset_16ths': grid.offset,
            'grid_match': getattr(grid, 'match', None), 'alignment_cost': cost,
            'vocal_notes': [{'t0': n['t0'], 't1': n['t1'], 'midi': n['midi'], 'held_units': n['dur']} for n in notes],
            'lines': out_lines, 'words': words}

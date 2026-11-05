"""Build the prompt corpus.

The pilot used 40 openers crossed with 18 topics -- 720 strings, often ungrammatical, all in
one register. Prompts are drawn independently of the hidden policy, so prompt identity leaks
nothing about the answer. What a narrow corpus does harm is the SENSOR distribution: guards
fire on entropy, on the top-two logit gap, on repetition. If every prompt puts the model in the
same state, the same guards fire in the same places and much of the policy space never shows
itself.

So the corpus is built for spread across model states rather than literary variety:

  narrative      open continuation, high entropy
  factual        encyclopedic framing, the model commits harder
  instructional  imperative, formulaic continuations
  dialogue       quoted speech, distinct token statistics
  enumeration    lists. These matter disproportionately: the repetition field and the
                 distinct-token sensor produce NO evidence in non-repetitive text, so without
                 prompts that invite repeats those parameters are unidentifiable by absence.
  technical      jargon, low-entropy stretches
  question       interrogative, mid entropy
  temporal       dated openings, strong priors

Registers are drawn to an equal quota. Sampling them at random exhausts the smaller ones first
and leaves the remainder dominated by whichever register has the largest combination count --
a first attempt produced a query set that was 100% enumeration.

Calibration and query prompts are disjoint, so per-prompt behaviour cannot be memorised from
the traces and reused at scored positions.

    python make_prompts.py
    NCAL=20000 NQUERY=10000 python make_prompts.py
"""
import os, csv, random
from collections import Counter

NCAL = int(os.environ.get('NCAL', 20000))
NQUERY = int(os.environ.get('NQUERY', 10000))
OUT = os.environ.get('OUT', '.')
SEED = 71294

PERSON = ['she', 'he', 'they', 'the driver', 'the nurse', 'the teacher', 'the engineer',
          'the farmer', 'the clerk', 'my brother', 'the manager', 'the pilot', 'the tenant',
          'the inspector', 'the courier', 'the gardener', 'the neighbour', 'the referee',
          'the plumber', 'the archivist', 'the barista', 'the surveyor', 'the porter',
          'the auditor', 'the curator', 'the welder', 'the midwife', 'the sailor',
          'the printer', 'the cook', 'the guard', 'the miller', 'the tailor', 'the vet',
          'the warden', 'the cellist', 'the glazier', 'the mason', 'the skipper']
OBJECT = ['engine', 'letter', 'ledger', 'kettle', 'ladder', 'switch', 'bridge', 'harvest',
          'invoice', 'passport', 'freezer', 'antenna', 'contract', 'blanket', 'compass',
          'furnace', 'suitcase', 'gearbox', 'notebook', 'thermostat', 'radiator', 'cistern',
          'railing', 'awning', 'turbine', 'palette', 'kiln', 'trellis', 'weir', 'flue',
          'lantern', 'anchor', 'satchel', 'pulley', 'cabinet', 'hinge', 'canopy', 'stirrup',
          'crate', 'valve', 'bellows', 'mallet', 'sieve', 'trolley', 'beacon', 'quilt',
          'harness', 'chisel', 'lattice', 'spindle']
PLACE = ['Lisbon', 'Dundee', 'Nairobi', 'Osaka', 'Cordoba', 'Halifax', 'Bergen', 'Perth',
         'Cusco', 'Tromso', 'Ghent', 'Split', 'Antwerp', 'Kanazawa', 'Rovaniemi', 'Cadiz',
         'Trieste', 'Aarhus', 'Galway', 'Utrecht', 'Salerno', 'Bruges', 'Tartu', 'Nara',
         'the valley', 'the coast', 'the border', 'the marshes', 'the highlands',
         'the estuary', 'the delta', 'the plateau', 'the foothills']
ACTION = ['stopped', 'refused', 'waited', 'agreed', 'argued', 'left', 'returned', 'complained',
          'apologised', 'insisted', 'hesitated', 'relented', 'objected', 'volunteered',
          'listened', 'answered', 'withdrew', 'conceded', 'protested', 'signed', 'resigned',
          'reconsidered', 'intervened', 'declined']
FIELD = ['drainage', 'masonry', 'grammar', 'accounting', 'forestry', 'cartography', 'welding',
         'typography', 'irrigation', 'metallurgy', 'ornithology', 'ceramics', 'hydrology',
         'bookbinding', 'horticulture', 'seismology', 'lexicography', 'glassblowing',
         'navigation', 'apiculture', 'joinery', 'thatching', 'toponymy', 'viticulture']
TECH = ['parser', 'buffer', 'compiler', 'scheduler', 'cache', 'index', 'daemon', 'socket',
        'checksum', 'allocator', 'iterator', 'mutex', 'registry', 'codec', 'kernel',
        'linker', 'profiler', 'dispatcher', 'validator', 'serialiser', 'router', 'tokeniser',
        'collector', 'accumulator', 'watchdog', 'sandbox']
ADJ = ['broken', 'missing', 'frozen', 'unmarked', 'faulty', 'spare', 'rusted', 'sealed',
       'temporary', 'unused', 'damp', 'narrow', 'crowded', 'quiet', 'expensive', 'cracked',
       'borrowed', 'painted', 'folded', 'loose', 'heavy', 'shallow', 'bent', 'dusty']
YEAR = [str(y) for y in range(1861, 2021, 3)]
# bare, so "Every {season}" and "By {season}" read correctly; an article-bearing entry like
# "the monsoon" produced "Every the monsoon"
SEASON = ['spring', 'summer', 'autumn', 'winter', 'harvest', 'midwinter', 'midsummer']

REGISTERS = {
    'narrative':     ["After the {adj} {obj}, {per}", "When {per} finally reached the {obj}",
                      "{per} had already sold the {obj}", "Before the {adj} {obj} arrived, {per}",
                      "Once {per} noticed the {adj} {obj}", "{per} {act} near the {obj}"],
    'factual':       ["The {field} of {place} is", "{place} is known for its {obj}",
                      "The {adj} {obj} of {place} was", "Records from {place} list the {obj}",
                      "Most {field} in {place} depends on"],
    'instructional': ["To repair the {adj} {obj}, first", "Never leave the {adj} {obj} near",
                      "Check the {obj} before you", "First remove the {adj} {obj}",
                      "Always store the {obj} beside the {obj}",
                      "Do not open the {adj} {obj} until", "Replace the {obj} whenever the {obj}",
                      "Ask {per} before moving the {obj}", "Label every {obj} in the {place}",
                      "Keep the {adj} {obj} away from"],
    'dialogue':      ["\"I never {act},\" said {per}", "\"The {obj} is {adj},\" {per} replied",
                      "{per} asked whether the {obj} was", "\"Tell me about the {field},\" said {per}",
                      "\"Where is the {adj} {obj}?\" asked {per}"],
    'enumeration':   ["The list included {obj}, {obj}, and", "Items: {obj}, {obj}, {obj}, and",
                      "We packed a {obj}, a {obj}, and", "First the {obj}, then the {obj}, then",
                      "{obj}, {obj}, {obj}, and then"],
    'technical':     ["The {tech} returns a {adj}", "If the {tech} fails, the {tech}",
                      "Each {tech} holds a {adj} {obj}", "The {tech} used in {field} must",
                      "Every {tech} writes to the {tech}"],
    'question':      ["Why did {per} {act} near the {obj}?", "What happens when the {adj} {obj}",
                      "How is the {field} of {place}", "Who owns the {adj} {obj} in {place}?",
                      "When did {per} last {act}?"],
    'temporal':      ["In {year}, the {adj} {obj} was", "By {season}, {per} had",
                      "Since {year}, {place} has kept", "Every {season} the {obj} is",
                      "Until {year}, the {obj} in {place}"],
}

SLOTS = dict(per=PERSON, obj=OBJECT, place=PLACE, act=ACTION, field=FIELD,
             tech=TECH, adj=ADJ, year=YEAR, season=SEASON)


def fill(pattern, rng):
    out = pattern
    for key, vocab in SLOTS.items():
        token = '{%s}' % key
        while token in out:
            out = out.replace(token, rng.choice(vocab), 1)
    # capitalise the first letter, skipping a leading quote
    for i, ch in enumerate(out):
        if ch.isalpha():
            return out[:i] + ch.upper() + out[i + 1:]
    return out


def register_space(reg):
    total = 0
    for p in REGISTERS[reg]:
        n = 1
        for key, vocab in SLOTS.items():
            n *= len(vocab) ** p.count('{%s}' % key)
        total += n
    return total


def build(n, rng, seen):
    """Equal quota per register; random sampling exhausts small registers and skews the rest."""
    regs = sorted(REGISTERS)
    quota = {r: n // len(regs) for r in regs}
    for r in regs[:n % len(regs)]:
        quota[r] += 1
    out = []
    for r in regs:
        got, tries = 0, 0
        while got < quota[r] and tries < quota[r] * 200:
            tries += 1
            s = fill(rng.choice(REGISTERS[r]), rng)
            if s in seen:
                continue
            seen.add(s)
            out.append((r, s))
            got += 1
        if got < quota[r]:
            print('  WARNING: %s produced %d of %d requested' % (r, got, quota[r]))
    rng.shuffle(out)
    return out


def write(path, rows, prefix):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['prompt_id', 'text'])
        for i, (_, s) in enumerate(rows):
            w.writerow(['%s%06d' % (prefix, i), s])
    # register recorded separately: recovering it from the text by prefix is unreliable
    # (`The ` opens several registers), and it is an internal label, not part of the released
    # column spec
    with open(path.replace('.csv', '_registers.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['prompt_id', 'register'])
        for i, (r, _) in enumerate(rows):
            w.writerow(['%s%06d' % (prefix, i), r])


def main():
    rng = random.Random(SEED)
    print('combinations per register:')
    for r in sorted(REGISTERS):
        print('  %-14s %.3g' % (r, register_space(r)))
    need = (NCAL + NQUERY) // len(REGISTERS)
    print('needed per register: %d\n' % need)

    seen = set()
    cal = build(NCAL, rng, seen)
    qry = build(NQUERY, rng, seen)
    assert not ({s for _, s in cal} & {s for _, s in qry}), 'calibration and query overlap'

    write(os.path.join(OUT, 'prompts_calibration.csv'), cal, 'PC')
    write(os.path.join(OUT, 'prompts_query.csv'), qry, 'PQ')

    print('calibration %d | query %d | disjoint: yes' % (len(cal), len(qry)))
    for name, rows in [('calibration', cal), ('query', qry)]:
        c = Counter(r for r, _ in rows)
        wl = [len(s.split()) for _, s in rows]
        print('  %-12s words %d-%d mean %.1f | registers %s'
              % (name, min(wl), max(wl), sum(wl) / len(wl),
                 ' '.join('%s:%d' % (k, v) for k, v in sorted(c.items()))))
    print('\nsamples:')
    for r, s in cal[:12]:
        print('  %-14s %s' % (r, s))


if __name__ == '__main__':
    main()

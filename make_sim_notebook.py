"""Assemble sim.ipynb: generate the simulation, then run the C1a gate."""
import json
from pathlib import Path

HERE = Path(__file__).parent
WORK = '/kaggle/working'


def _lines(t):
    ls = t.strip(chr(10)).split(chr(10))
    return [l + chr(10) for l in ls[:-1]] + [ls[-1]]


def md(t):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': _lines(t)}


def code(t):
    return {'cell_type': 'code', 'execution_count': None, 'metadata': {},
            'outputs': [], 'source': _lines(t)}


def script(name):
    return code('%%writefile ' + WORK + '/' + name + '\n' +
                (HERE / name).read_text(encoding='utf-8'))


cells = []

cells.append(md("""
# Stateful decoding policies — simulation and the C1a gate

The released dataset is unreproducible: its generator sampled with `torch.multinomial` and never
called `torch.manual_seed`, so the hidden policies existed only in that process's memory and no
mode labels can be recovered. This notebook generates a fresh simulation that records everything,
then runs the one gate that decides whether the main experiment is worth building.

**The design principle.** The base model's next-token distribution is stored at every step. Every
sampler in the grammar is a deterministic function of it, so afterwards every fit, baseline,
ablation and diagnostic runs on CPU with no model and no GPU. It is the only quantity that cannot
be reconstructed without repeating this run.

**The gate.** `C1a` reveals the true mode label on the calibration traces and nowhere else, then
fits the policy with that supervision. It is exactly what a perfect mode predictor would hand
downstream, so it is the real ceiling of the amortized approach — not the true-policy oracle.

```
PASS   C1a >= 0.70 of headroom  -> build the predictor; it must then reach
                                   >= 0.60 of headroom AND >= 0.70 x C1a
FAIL   C1a <  0.70              -> stop. Revealing the modes does not identify
                                   the policies, so 0.60 is unreachable by
                                   construction and a predictor adjudicates nothing.
```

Pre-committed before any number was seen. GPU on for section 3; everything after is CPU.
"""))

cells.append(md('## 1. Environment'))
cells.append(code("""
import torch, platform
print('torch', torch.__version__, '| python', platform.python_version())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print('gpu  ', p.name, '|', round(p.total_memory / 1024**3, 1), 'GiB')
else:
    print('gpu   NONE -- generation will not finish in reasonable time')
"""))

cells.append(md("""
## 2. Scripts and prompts

Self-contained: the prompt corpus is generated here from templates rather than read from any
external dataset, so this notebook needs nothing attached.

The corpus is built for spread across MODEL STATES, not literary variety. Guards fire on entropy,
on the top-two logit gap, on repetition -- if every prompt puts the model in the same state, the
same guards fire in the same places and much of the policy space never shows itself. Enumeration
prompts matter disproportionately: the repetition sensors produce no evidence at all in
non-repetitive text, so without prompts that invite repeats those parameters are unidentifiable
by absence.
"""))
for n in ['grammar_v3.py', 'make_prompts.py', 'probe.py', 'policy_cpu.py',
          'simlab.py', 'patch_scored.py', 'gate_c1a.py', 'planted.py']:
    cells.append(script(n))

cells.append(code("""
import os
os.makedirs('/kaggle/working/raw', exist_ok=True)
!cd %s && OUT=/kaggle/working/raw python make_prompts.py
os.environ['PC'] = '/kaggle/working/raw/prompts_calibration.csv'
os.environ['PQ'] = '/kaggle/working/raw/prompts_query.csv'
import csv
for k in ('PC', 'PQ'):
    rows = list(csv.DictReader(open(os.environ[k], encoding='utf-8')))
    print(k, len(rows), 'prompts |', repr(rows[0]['text'])[:60])
""" % WORK))

cells.append(md("""
## 3. Generate

Roughly 90 minutes and about 2.5 GB after compression. What is recorded per step: the base
model's top-512 distribution, the full-vocabulary statistics that truncation destroys
(`zstd`, and `logsumexp(z/T)` plus entropy on a temperature grid), the five sensors, the
ground-truth mode, transition provenance with signed threshold margins, the emitted token's
log-prob under **every** mode, inter-mode total variation, and the uniform draw the sampler
consumed.

Conditions are balanced across tasks: modes 1/2/3, dwell on/off, and thresholds placed at
designed quantiles of each sensor's measured distribution (well-bracketed at the 40-60th
percentile, poorly-bracketed at the 5th/95th). The margin condition is a two-pass design and
cannot be added afterwards.

Watch the summary at the end: if the degenerate count is high or the informationless fraction
is large, the run is weaker than it looks and the gate will inherit that.
"""))
cells.append(code("""
import os
os.environ.update(OUT='/kaggle/working/sim', NTASK='250', NTRACE='16', TRACELEN='256',
                  NITEM='40', TOPK='512', NREPLICATE='10')
!cd %s && python -u simlab.py
""" % WORK))

cells.append(md("""
## 4. The CPU path audit

Before any fit is trusted, check that the CPU reimplementation reproduces the GPU generator from
the stored distributions. Offline this was verified to a median total-variation gap of 5.6e-8
across all ten (order x truncation) combinations, after four bugs that each would have silently
corrupted every fit: renormalising the top-K, `top_nsigma` needing the full-vocabulary log-prob
std, `hf` order needing the tempered partition, and typical sampling needing the tempered
entropy. This cell confirms it holds on the real data.
"""))
cells.append(code("""
import json, glob, numpy as np, sys, importlib
sys.path.insert(0, '/kaggle/working')
import policy_cpu as P
P = importlib.reload(P)          # %%writefile changed the file; the module may still be cached
print('policy_cpu VERSION =', P.VERSION)
assert P.VERSION == '2026-10-09.mirostat-state-fix', (
    'STALE MODULE: re-run the policy_cpu writefile cell, then this one again')
env = json.load(open('/kaggle/working/sim/env.json'))
rows = [json.loads(l) for l in open('/kaggle/working/sim/tasks.jsonl', encoding='utf-8')][:8]
errs = []
for r in rows:
    npz = np.load('/kaggle/working/sim/%s.npz' % r['task_id'])
    e = P.verify_against_generator(r, npz, np.array(env['tgrid']))
    errs.append(np.median(e))
    tr = ','.join(m['trunc'] for m in r['policy']['modes'])
    kept = np.mean([s.get('kept', np.nan) for t in r['traces'][:2] for s in t['steps'][:256]])
    # how often the emitted token falls outside the stored top-512: those steps have no
    # likelihood, and (before the fix) also skipped the state advance
    oos = []
    for ti in range(min(2, len(r['traces']))):
        ids = npz['cal_ids'][ti]
        for k, st_ in enumerate(r['traces'][ti]['steps'][:256]):
            oos.append(not np.any(ids[k] == st_['tok']))
    print('%s  median %.2e  max %.2e | order=%-6s truncs=%-22s kept=%.3f oos=%.3f'
          % (r['task_id'], np.median(e), e.max(), r['policy']['order'], tr, kept,
             float(np.mean(oos))))
print()
print('overall median %.2e' % np.median(errs))
print('under ~1e-3 the CPU path is sound; above that, nothing downstream is trustworthy')
print()
print('if failures concentrate on one truncation family, that sampler is not reconstructible')
print('from the stored top-512 and those tasks should be excluded rather than regenerated')
"""))

cells.append(md("""
## 4b. Backfill the scored position (only needed for a run generated before this was stored)

The first version of the generator stored the base distribution for the 24 prefix steps but not
for the scored position itself. Truncation samplers need the whole distribution to decide what
survives, so the step being predicted was uncomputable offline. `simlab.py` now stores it; this
cell backfills a run that predates the fix, with one short forward pass per item instead of
regenerating. It is idempotent and skips tasks that already have it.
"""))
cells.append(code("""
import os
os.environ['SIM'] = '/kaggle/working/sim'
!cd %s && python -u patch_scored.py
""" % WORK))

cells.append(md("""
## 5. The gate

Fits each mode's sampler on the calibration steps **labelled with that mode**, recovers the
transition rules from the revealed mode sequence, then runs the fitted machine forward over each
scored prefix. Reported against the base model (0) and the true policy (1), alongside `C3`, the
best single stateless sampler fitted with the same optimiser and budget — which is the honest
stateless control, not the one-parameter temperature fit used previously.

CPU only. About 15 minutes.
"""))
cells.append(code("""
import os
os.environ.update(SIM='/kaggle/working/sim', GATE_NTASK='40', FITSTEPS='1024',
                  PROPOSALS='120', BUDGET='16x256', EPS='0.03')
!cd %s && python -u gate_c1a.py
""" % WORK))

cells.append(md("""
## 6. Verdict

The cell below restates the gate's decision and, if it failed, says plainly what the result is.

A failure here is not a wasted run. "Even with the modes revealed, passive single-sample
observation does not identify these policies" is a sharper and more defensible claim than a
method comparison, and it is supported by everything this run recorded: the inter-mode total
variation per step bounds what any mode inference could achieve, and the guard margins bound how
tightly each threshold is bracketed by the observed trajectory.
"""))
cells.append(code("""
import json
g = json.load(open('/kaggle/working/sim/gate_c1a.json'))
print('C1a                %.3f of headroom' % g['c1a'])
print('C3 stateless       %.3f' % g['stateless'])
print('headroom           %.4f nats over %d items, %d tasks'
      % (g['headroom'], g['n_items'], g['n_tasks']))
print('CPU audit median   %.2e nats' % g['audit_median'])
print()
if g['passed']:
    print('PASS -- build the amortized mode predictor.')
    print('   it must reach >= 0.600 of headroom AND >= %.3f' % (0.70 * g['c1a']))
else:
    print('FAIL -- stop. Report the identifiability result:')
    print('   revealing the true modes reaches only %.3f of headroom, so mode inference')
    print('   is not the bottleneck and no amortized predictor can clear 0.600.')
"""))

cells.append(md("""
## 7. Package the simulation

Save the run as a dataset so every later analysis is CPU-only and nothing needs regenerating.
"""))
cells.append(code("""
!du -sh /kaggle/working/sim
!ls /kaggle/working/sim | head -5
!cat /kaggle/working/sim/env.json
"""))

nb = {'cells': cells,
      'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                                  'name': 'python3'},
                   'language_info': {'name': 'python'}},
      'nbformat': 4, 'nbformat_minor': 5}

out = HERE / 'sim.ipynb'
json.dump(nb, open(out, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('wrote %s with %d cells' % (out, len(cells)))

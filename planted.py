"""Planted-recovery test: can the fitter recover a policy it was handed the answer to?

    python planted.py

WHY THIS EXISTS. The C1a gate returned -1.911 of headroom -- worse than ignoring the policy
entirely -- and degraded monotonically with parameter count (M=1 +0.281, M=2 -0.4, M=3 -6.5).
That is the signature of a starved optimiser, not of an unidentifiable problem: 120 random
single-coordinate proposals over up to 24 dimensions barely leaves the prior midpoint. So the
gate measured the fitter, and its FAIL cannot be reported as a finding.

This test separates the two possibilities the gate confounded, on data where the answer is known:

    fit the policy from ITS OWN calibration traces, with the true modes revealed, and compare
    the fitted calibration NLL against the TRUE policy's calibration NLL on the same steps

    recovery gap ~ 0           the fitter works; the gate's failure was the optimiser
    recovery gap stays large   the data does not pin the parameters down even with modes given,
                               which IS the identifiability result

Reported per condition, because the gate's collapse was parameter-count-dependent and the
diagnosis has to be too.

Two optimisers are compared on identical budgets, since the point is to establish whether a
competent optimiser closes the gap:

    random      the gate's recipe: perturb one coordinate by a Gaussian, accept if better
    coordinate  sweep each coordinate over a line of values and take the best

All CPU.
"""
import os, json, math, time, random
import numpy as np

import policy_cpu as P

SIM = os.environ.get('SIM', './sim')
NTASK = int(os.environ.get('PLANT_NTASK', 12))
FITSTEPS = int(os.environ.get('FITSTEPS', 512))
BUDGET = os.environ.get('BUDGET', '8x128')
EVALS = int(os.environ.get('EVALS', 480))      # likelihood evaluations, identical per optimiser
LINE = int(os.environ.get('LINE', 6))          # values per coordinate for the line search
AUDIT_MAX = float(os.environ.get('AUDIT_MAX', 5e-3))
SEED = 71294
T0 = time.time()
KEYS = ('tp', 'eta', 't_lo', 't_hi', 't_exp', 'xtc_q', 'xtc_t', 'lam')


def log(m):
    print('[%7.1fs] %s' % (time.time() - T0, m), flush=True)


def bounds(md, k):
    import grammar_v3 as G
    return G.TP_BOUNDS[md['trunc']] if k == 'tp' else G.BOUNDS[k]


def budget_traces(task, spec, cap):
    n, ln = (int(x) for x in spec.split('x'))
    out, used = [], 0
    for ti in range(min(n, len(task['traces']))):
        steps = task['traces'][ti]['steps'][:ln]
        if used + len(steps) > cap and out:
            break
        out.append((ti, steps))
        used += len(steps)
    return out


def make_nll(task, npz, traces, tgrid):
    """Calibration NLL under given per-mode parameters, modes REVEALED."""
    cache = []
    for ti, steps in traces:
        Pm, _ = P.rebuild(npz['cal_lg'][ti][:len(steps)], npz['cal_tail'][ti][:len(steps)])
        cache.append((steps, Pm, npz['cal_ids'][ti], npz['cal_lg'][ti]))
    calls = [0]

    def nll(ms):
        calls[0] += 1
        tot, n = 0.0, 0
        for steps, Pm, ids, lg in cache:
            lp, _ = P.replay_trace(task, steps, Pm, ids, lg, tgrid, modes=ms)
            ok = np.isfinite(lp)
            tot -= float(lp[ok].sum())
            n += int(ok.sum())
        return tot / max(n, 1)
    return nll, calls


def start_mid(pol):
    """The gate's starting point: every coordinate at its prior midpoint."""
    ms = []
    for m in range(pol['M']):
        md = dict(pol['modes'][m])
        for k in [x for x in KEYS if x in md]:
            lo, hi = bounds(md, k)
            md[k] = 0.5 * (lo + hi)
        ms.append(md)
    return ms


def opt_random(nll, ms, coords, rng, evals):
    """The gate's optimiser: random single-coordinate Gaussian proposals."""
    best = nll(ms)
    for _ in range(evals - 1):
        m, k = coords[rng.randrange(len(coords))]
        cand = [dict(x) for x in ms]
        lo, hi = bounds(cand[m], k)
        cand[m][k] = float(np.clip(cand[m][k] + rng.gauss(0, 0.15 * (hi - lo)), lo, hi))
        v = nll(cand)
        if v < best:
            ms, best = cand, v
    return ms, best


def opt_coord(nll, ms, coords, rng, evals):
    """Coordinate-wise line search: sweep each coordinate, keep the best value, repeat.

    Same number of likelihood evaluations as the random optimiser. A coordinate sweep spends
    them where they inform a decision instead of taking random steps that are usually rejected.
    """
    best = nll(ms)
    used = 1
    while used + LINE <= evals:
        improved = False
        for (m, k) in coords:
            if used + LINE > evals:
                break
            lo, hi = bounds(ms[m], k)
            grid = np.linspace(lo, hi, LINE)
            vals = []
            for g in grid:
                cand = [dict(x) for x in ms]
                cand[m][k] = float(g)
                vals.append(nll(cand))
                used += 1
            j = int(np.argmin(vals))
            if vals[j] < best:
                ms = [dict(x) for x in ms]
                ms[m][k] = float(grid[j])
                best = vals[j]
                improved = True
        if not improved:
            break
    return ms, best


def main():
    log('policy_cpu VERSION = %s' % getattr(P, 'VERSION', 'UNKNOWN'))
    env = json.load(open(os.path.join(SIM, 'env.json')))
    tgrid = np.array(env['tgrid'])

    rows = []
    for line in open(os.path.join(SIM, 'tasks.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        if r.get('degenerate') or not r['items']:
            continue
        npz = np.load(os.path.join(SIM, '%s.npz' % r['task_id']))
        if float(np.median(P.verify_against_generator(r, npz, tgrid, ntrace=1, n=128))) > AUDIT_MAX:
            continue
        rows.append(r)
        if len(rows) >= NTASK:
            break
    log('%d tasks | budget %s | <=%d steps | %d evaluations per optimiser'
        % (len(rows), BUDGET, FITSTEPS, EVALS))

    out = []
    for r in rows:
        npz = np.load(os.path.join(SIM, '%s.npz' % r['task_id']))
        traces = budget_traces(r, BUDGET, FITSTEPS)
        nll, calls = make_nll(r, npz, traces, tgrid)
        pol = r['policy']

        nll_true = nll(pol['modes'])          # the answer, for reference
        ms0 = start_mid(pol)
        nll_mid = nll(ms0)
        coords = [(m, k) for m in range(pol['M']) for k in KEYS if k in ms0[m]]

        _, nll_rand = opt_random(nll, [dict(x) for x in ms0], coords,
                                 random.Random(SEED + int(r['task_id'][1:])), EVALS)
        _, nll_coo = opt_coord(nll, [dict(x) for x in ms0], coords,
                               random.Random(SEED + int(r['task_id'][1:])), EVALS)

        # fraction of the midpoint-to-truth gap that the optimiser closed
        span = nll_mid - nll_true
        clos = lambda v: (nll_mid - v) / span if abs(span) > 1e-9 else float('nan')
        out.append(dict(task=r['task_id'], M=pol['M'], dwell=bool(r['condition']['dwell']),
                        ncoord=len(coords), true=nll_true, mid=nll_mid,
                        rand=nll_rand, coord=nll_coo,
                        clos_rand=clos(nll_rand), clos_coord=clos(nll_coo)))
        log('  %s M=%d coords=%-2d | NLL true %.4f mid %.4f | random %.4f (%.2f) '
            'coord %.4f (%.2f)'
            % (r['task_id'], pol['M'], len(coords), nll_true, nll_mid,
               nll_rand, out[-1]['clos_rand'], nll_coo, out[-1]['clos_coord']))

    print()
    print('  RECOVERY: fraction of the midpoint-to-truth NLL gap closed, by mode count')
    print('  %-6s %-5s %-10s %s' % ('modes', 'n', 'random', 'coordinate'))
    for M in sorted({o['M'] for o in out}):
        g = [o for o in out if o['M'] == M]
        print('  %-6d %-5d %-10.3f %.3f'
              % (M, len(g), float(np.median([o['clos_rand'] for o in g])),
                 float(np.median([o['clos_coord'] for o in g]))))
    print()
    print('  overall median: random %.3f | coordinate %.3f'
          % (float(np.median([o['clos_rand'] for o in out])),
             float(np.median([o['clos_coord'] for o in out]))))
    worse = sum(o['coord'] > o['mid'] for o in out)
    print('  tasks where the coordinate search ended worse than its start: %d of %d'
          % (worse, len(out)))

    print()
    med = float(np.median([o['clos_coord'] for o in out]))
    if med >= 0.80:
        print('  THE FITTER WORKS. A competent optimiser closes most of the gap, so the gate\'s')
        print('  -1.911 was the optimiser, not the problem. Re-run the gate with the')
        print('  coordinate search and read the result as a real measurement.')
    elif med >= 0.40:
        print('  PARTIAL. The optimiser is a large part of the gate\'s failure but not all of')
        print('  it. Re-run the gate with the coordinate search, and treat any residual')
        print('  shortfall as jointly attributable -- do not claim identifiability from it.')
    else:
        print('  THE DATA DOES NOT PIN THE PARAMETERS DOWN. Even with the true modes revealed')
        print('  and a competent optimiser, the calibration traces do not determine the')
        print('  policy. That IS the identifiability result, and it is the finding to report:')
        print('  passive single-sample observation is insufficient regardless of method.')
        print('  Check nll(true) vs nll(mid) first -- if that span is tiny, the likelihood is')
        print('  flat in the parameters and non-identifiability is direct.')

    json.dump(out, open(os.path.join(SIM, 'planted.json'), 'w'), indent=1)
    log('wrote %s/planted.json' % SIM)


if __name__ == '__main__':
    main()

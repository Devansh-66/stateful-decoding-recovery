"""C1a: the mode-oracle control. Run this BEFORE training any predictor.

    python gate_c1a.py

C1a reveals the TRUE mode label at every step of the calibration traces -- and nowhere else --
then fits the policy with that supervision and predicts the scored items. It is exactly what a
perfect mode predictor would hand downstream, so it is the real ceiling of the amortized
approach, not the true-policy oracle at 1.0.

WHY IT IS A GATE. The kill condition for the amortized experiment is 0.6 of headroom, expressed
against the true-policy oracle. If C1a itself lands below about 0.7, then 0.6 is unreachable by
construction, the experiment adjudicates nothing, and training a predictor adds no information.
The finding would instead be: even with the modes revealed, passive single-sample observation
does not identify these policies.

Pre-committed, before any number was seen:

    PASS  C1a >= 0.70 of headroom  -> build the predictor; it must then reach >= 0.60 of
                                      headroom AND >= 0.70 x C1a
    FAIL  C1a <  0.70              -> stop. Report the identifiability result instead.

All CPU: simlab stored the base distribution at every step.
"""
import os, json, math, time, random
import numpy as np

import policy_cpu as P

SIM = os.environ.get('SIM', './sim')
NTASK = int(os.environ.get('GATE_NTASK', 40))
FITSTEPS = int(os.environ.get('FITSTEPS', 1024))   # cap on calibration steps per fit
PROPOSALS = int(os.environ.get('PROPOSALS', 120))
BUDGET = os.environ.get('BUDGET', '16x256')
EPS = float(os.environ.get('EPS', 0.03))
THR_C1A = 0.70
# Tasks whose induced distribution cannot be reconstructed from the stored top-512 are excluded.
# This is not a convenience filter: mirostat2 thresholds on p >= 2^-mu, which at mu ~ 18 is
# 3.8e-6 and sits far below rank 512, so its keep-set extends into the tail the record does not
# hold -- measured at up to 23% of emitted tokens falling outside the support. Every other
# sampler reconstructs to <= 1.9e-3 nats. The criterion is the MEASURED error, not the sampler
# name, because a mirostat task whose mu stays high is perfectly reconstructible and is kept.
AUDIT_MAX = float(os.environ.get('AUDIT_MAX', 5e-3))
SEED = 71294
T0 = time.time()
KEYS = ('tp', 'eta', 't_lo', 't_hi', 't_exp', 'xtc_q', 'xtc_t', 'lam')


def log(m):
    print('[%7.1fs] %s' % (time.time() - T0, m), flush=True)


def budget_traces(task, spec, cap):
    """Whole traces, truncated. NOT a random sample of steps.

    The generator creates one state per trace and evolves it every step, so mirostat's mu and
    the repetition counters depend on the whole preceding trace. Subsampling steps breaks that
    continuity and makes every likelihood wrong.
    """
    n, ln = (int(x) for x in spec.split('x'))
    out, used = [], 0
    for ti in range(min(n, len(task['traces']))):
        steps = task['traces'][ti]['steps'][:ln]
        if used + len(steps) > cap and out:
            break
        out.append((ti, steps))
        used += len(steps)
    return out


def load(limit, tgrid):
    """Usable tasks only: non-degenerate, with items, and reconstructible on the CPU path."""
    rows, drop = [], dict(degenerate=0, no_items=0, audit=0)
    dropped_truncs = {}
    for line in open(os.path.join(SIM, 'tasks.jsonl'), encoding='utf-8'):
        r = json.loads(line)
        if r.get('degenerate'):
            drop['degenerate'] += 1
            continue
        if not r['items']:
            drop['no_items'] += 1
            continue
        if 'scored' not in r['items'][0]:
            raise SystemExit('tasks lack the scored-position record; run patch_scored.py first')
        npz = np.load(os.path.join(SIM, '%s.npz' % r['task_id']))
        e = float(np.median(P.verify_against_generator(r, npz, tgrid, ntrace=1, n=128)))
        if e > AUDIT_MAX:
            drop['audit'] += 1
            for m in r['policy']['modes']:
                dropped_truncs[m['trunc']] = dropped_truncs.get(m['trunc'], 0) + 1
            continue
        r['_audit'] = e
        rows.append(r)
        if len(rows) >= limit:
            break
    log('excluded: %d degenerate, %d itemless, %d above audit %.0e'
        % (drop['degenerate'], drop['no_items'], drop['audit'], AUDIT_MAX))
    if dropped_truncs:
        log('  truncations among excluded tasks: %s'
            % ', '.join('%s=%d' % kv for kv in sorted(dropped_truncs.items(),
                                                      key=lambda kv: -kv[1])))
    return rows


def bounds(md, k):
    import grammar_v3 as G
    return G.TP_BOUNDS[md['trunc']] if k == 'tp' else G.BOUNDS[k]


def fit_joint(task, npz, traces, tgrid, rng):
    """Fit every mode's sampler together, with the modes REVEALED.

    One hill-climb over all modes' coordinates at once, because a single state runs through the
    whole trace and the modes share it: mode 1's parameters change the state mode 0 then sees.
    Fitting modes separately is wrong for exactly the long-dwell stretches that matter most.
    Starts from the prior midpoint, never from the truth.
    """
    pol = task['policy']
    cache = []
    for ti, steps in traces:
        Pm, _ = P.rebuild(npz['cal_lg'][ti][:len(steps)], npz['cal_tail'][ti][:len(steps)])
        cache.append((steps, Pm, npz['cal_ids'][ti], npz['cal_lg'][ti]))

    modes = []
    for m in range(pol['M']):
        md = dict(pol['modes'][m])
        for k in [x for x in KEYS if x in md]:
            lo, hi = bounds(md, k)
            md[k] = 0.5 * (lo + hi)
        modes.append(md)

    def nll(ms):
        tot, n = 0.0, 0
        for steps, Pm, ids, lg in cache:
            lp, _ = P.replay_trace(task, steps, Pm, ids, lg, tgrid, modes=ms)
            ok = np.isfinite(lp)
            tot -= float(lp[ok].sum())
            n += int(ok.sum())
        return tot / max(n, 1)

    coords = [(m, k) for m in range(pol['M']) for k in KEYS if k in modes[m]]
    best = nll(modes)
    for _ in range(PROPOSALS):
        m, k = coords[rng.randrange(len(coords))]
        cand = [dict(x) for x in modes]
        lo, hi = bounds(cand[m], k)
        cand[m][k] = float(np.clip(cand[m][k] + rng.gauss(0, 0.15 * (hi - lo)), lo, hi))
        v = nll(cand)
        if v < best:
            modes, best = cand, v
    return modes, best


def fit_guards(task, traces):
    """Recover the transition rules from the REVEALED mode sequence, by decision stumps.

    With modes given this is supervised and is the easy half of C1a; the sampler fit is the hard
    half.
    """
    pol = task['policy']
    if pol['M'] == 1:
        return []
    steps = [r for _, ss in traces for r in ss]
    guards = []
    for src in range(pol['M']):
        for dst in range(pol['M']):
            if src == dst:
                continue
            fired = [r for r in steps if r['mode'] == src and r['mode_after'] == dst]
            stay = [r for r in steps if r['mode'] == src and r['mode_after'] == src]
            if len(fired) < 5 or len(stay) < 5:
                continue
            best = None
            for sn in P.SENSORS:
                vals = sorted({round(r[sn], 4) for r in fired + stay})
                for thr in vals[::max(1, len(vals) // 40)]:
                    for cmp in ('ge', 'lt'):
                        ok = sum((r[sn] >= thr) if cmp == 'ge' else (r[sn] < thr)
                                 for r in fired) / len(fired)
                        bad = sum((r[sn] >= thr) if cmp == 'ge' else (r[sn] < thr)
                                  for r in stay) / len(stay)
                        if best is None or ok - bad > best[0]:
                            best = (ok - bad, sn, thr, cmp)
            if best and best[0] > 0.2:
                guards.append(dict(src=src, dst=dst, sensor=best[1], thr=best[2], cmp=best[3]))
    return guards


def predict_items(task, npz, fitted, tgrid):
    """Run the fitted machine forward over each prefix, then score the 8 candidates.

    The mode at the scored position comes from the FITTED guards, not the truth: C1a reveals
    modes on the calibration traces only.
    """
    last = npz['itm_lg'].shape[1] - 1
    out = []
    for n, it in enumerate(task['items']):
        steps = it['steps']
        Pm, _ = P.rebuild(npz['itm_lg'][n][:len(steps)], npz['itm_tail'][n][:len(steps)])
        st = P.fresh_state(fitted['modes'][0])
        cur, c = 0, 0
        for k, rec in enumerate(steps):
            md = fitted['modes'][cur]
            P.induced_at(task, rec, Pm[k], npz['itm_ids'][n][k], npz['itm_lg'][n][k],
                         md, st, tgrid)
            # advance unconditionally: mirostat's mu must track the generator even when the
            # emitted token is outside the stored support
            P.advance_state(st, md, task['policy']['gamma'], rec['lp_base'], rec['tok'])
            cur, c = P.step_fsm(fitted, cur, c, rec)
        Pl, _ = P.rebuild(npz['itm_lg'][n][last], npz['itm_tail'][n][last])
        ids = npz['itm_ids'][n][last]
        d = P.induced_at(task, it['scored'], Pl, ids, npz['itm_lg'][n][last],
                         fitted['modes'][cur], st, tgrid)
        v = np.array([float(d[np.where(ids == cid)[0][0]]) if np.any(ids == cid) else 0.0
                      for cid in it['candidates']])
        out.append(v / v.sum() if v.sum() > 0 else np.full(len(v), 1.0 / len(v)))
    return np.stack(out)


def sum_loss(Pm, task, eps, base):
    idx = np.array([it['true_index'] for it in task['items']])
    Q = (1 - eps) * Pm + eps * base
    return float(-np.log(np.clip(Q[np.arange(len(idx)), idx], 1e-7, 1.0)).sum()), len(idx)


def main():
    log('policy_cpu VERSION = %s' % getattr(P, 'VERSION', 'UNKNOWN'))
    env = json.load(open(os.path.join(SIM, 'env.json')))
    tgrid = np.array(env['tgrid'])
    rows = load(NTASK, tgrid)
    log('%d usable tasks | budget %s | <=%d fit steps | %d proposals'
        % (len(rows), BUDGET, FITSTEPS, PROPOSALS))

    tot = dict(base=0.0, ref=0.0, c1a=0.0, stateless=0.0)
    n_items, per_task, audit = 0, [], []
    for r in rows:
        npz = np.load(os.path.join(SIM, '%s.npz' % r['task_id']))
        traces = budget_traces(r, BUDGET, FITSTEPS)
        rng = random.Random(SEED + int(r['task_id'][1:]))
        audit.append(r['_audit'])

        b = np.stack([np.array(it['cand_lp_base']) for it in r['items']])
        base = np.exp(b - b.max(1, keepdims=True))
        base /= base.sum(1, keepdims=True)

        fitted = dict(r['policy'])
        fitted['modes'], _ = fit_joint(r, npz, traces, tgrid, rng)
        fitted['guards'] = fit_guards(r, traces)
        Pc = predict_items(r, npz, fitted, tgrid)

        # C3: the best SINGLE stateless sampler, same optimiser and same budget. The honest
        # stateless control; a one-parameter temperature fit is a far weaker baseline.
        flat_pol = dict(r['policy'], M=1, guards=[], dwell=[0], expiry=[10 ** 9],
                        modes=[dict(r['policy']['modes'][0])])
        flat_task = dict(r, policy=flat_pol)
        flat_tr = [(ti, [dict(x, mode=0) for x in ss]) for ti, ss in traces]
        fm, _ = fit_joint(flat_task, npz, flat_tr, tgrid, rng)
        flat_pol = dict(flat_pol, modes=fm)
        Ps = predict_items(dict(r, policy=flat_pol), npz, flat_pol, tgrid)

        lb = -float(np.sum([it['lp_base'] for it in r['items']]))
        lr = -float(np.sum([it['lp_policy'] for it in r['items']]))
        lc, n = sum_loss(Pc, r, EPS, base)
        ls, _ = sum_loss(Ps, r, EPS, base)
        tot['base'] += lb; tot['ref'] += lr; tot['c1a'] += lc; tot['stateless'] += ls
        n_items += n
        per_task.append(dict(task=r['task_id'], cond=r['condition'], n=n,
                             base=lb / n, ref=lr / n, c1a=lc / n, stateless=ls / n))
        if len(per_task) % 5 == 0:
            log('  %d/%d tasks' % (len(per_task), len(rows)))

    L = {k: v / n_items for k, v in tot.items()}
    head = L['base'] - L['ref']
    frac = lambda x: (L['base'] - x) / head

    print()
    print('  CPU-path audit: median |recomputed - stored| = %.2e nats' % float(np.median(audit)))
    print('  (above ~1e-3 nothing below is trustworthy)')
    print()
    print('  L_base %.4f | L_ref %.4f | headroom %.4f nats | %d items over %d tasks'
          % (L['base'], L['ref'], head, n_items, len(rows)))
    print()
    print('  %-34s %-9s %s' % ('control', 'L', 'of headroom'))
    print('  %-34s %-9.4f %.3f' % ('base model', L['base'], 0.0))
    print('  %-34s %-9.4f %.3f' % ('C3 best stateless fit', L['stateless'],
                                   frac(L['stateless'])))
    print('  %-34s %-9.4f %.3f' % ('C1a mode-oracle fit', L['c1a'], frac(L['c1a'])))
    print('  %-34s %-9.4f %.3f' % ('true policy (oracle)', L['ref'], 1.0))

    # per-condition breakdown: dwell=False is pointwise identifiable by design, so C1a should
    # be clearly higher there. If it is not, the fit is the problem rather than the observation.
    print()
    print('  by condition (C1a share of that cell\'s own headroom):')
    cells = {}
    for t in per_task:
        k = (t['cond']['M'], bool(t['cond']['dwell']))
        cells.setdefault(k, []).append(t)
    for k in sorted(cells):
        g = cells[k]
        hb = np.average([t['base'] for t in g], weights=[t['n'] for t in g])
        hr = np.average([t['ref'] for t in g], weights=[t['n'] for t in g])
        hc = np.average([t['c1a'] for t in g], weights=[t['n'] for t in g])
        sh = (hb - hc) / (hb - hr) if hb > hr else float('nan')
        print('    M=%d dwell=%-5s n=%-3d  %.3f' % (k[0], k[1], len(g), sh))

    c1a = frac(L['c1a'])
    print()
    if c1a >= THR_C1A:
        print('  GATE PASSED: C1a = %.3f >= %.2f' % (c1a, THR_C1A))
        print('  Build the predictor. It must reach >= 0.600 of headroom AND >= %.3f.'
              % (0.70 * c1a))
    else:
        print('  GATE FAILED: C1a = %.3f < %.2f' % (c1a, THR_C1A))
        print('  Revealing the true modes does not identify these policies from passive')
        print('  single-sample observation, so 0.600 is unreachable by construction and an')
        print('  amortized predictor cannot adjudicate anything. STOP HERE.')
        print('  Report the identifiability result, not a method failure.')

    json.dump(dict(L=L, headroom=head, c1a=c1a, stateless=frac(L['stateless']),
                   n_items=n_items, n_tasks=len(rows), audit_median=float(np.median(audit)),
                   threshold=THR_C1A, passed=bool(c1a >= THR_C1A), per_task=per_task),
              open(os.path.join(SIM, 'gate_c1a.json'), 'w'), indent=1)
    log('wrote %s/gate_c1a.json' % SIM)


if __name__ == '__main__':
    main()

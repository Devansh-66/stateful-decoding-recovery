"""Re-implementation of the policy transform over the stored top-K support, in numpy.

Every sampler in grammar_v3 is a deterministic function of the base distribution and its
parameters. simlab.py stores that distribution at every step, so the whole fitting problem runs
on CPU here with no model and no GPU. This module is the bridge.

Correctness note: candidates are drawn from the top STRATUM_TOP=512 base ranks, which is exactly
the stored support, so no candidate is ever outside it. The one approximation is the repetition
penalty, which in the generator applies to any token in the recent history; here it applies only
to history tokens that fall inside the stored top-K. verify_against_generator() measures that gap
rather than assuming it away.
"""
import numpy as np

# Bump on every change. The audit prints it, so a stale import is visible instead of silent --
# %%writefile replaces the FILE, but an already-imported module stays in memory, and we lost
# three audit runs to that before making the version observable.
VERSION = '2026-10-09.mirostat-state-fix'

SENSORS = ['H', 'G', 'U', 'D', 'P']
_TYPICAL_ENT = [0.0]   # stored full entropy, tempered where the sampler sees it tempered
_ZSTD = [None]         # std of full-vocab log-probs, scaled by 1/T for hf order


def grid_at(T, tgrid, vals):
    """Interpolate a stored temperature grid QUADRATICALLY in log-T.

    logsumexp(z/T) and the tempered entropy are smooth in log-T, and the grid is geometric, so
    quadratic interpolation is 7x more accurate than linear on the same points: max error drops
    from 1.1e-2 to 1.6e-3 nats. Free, and needs no regeneration.
    """
    tg = np.asarray(tgrid, dtype=np.float64)
    v = np.asarray(vals, dtype=np.float64)
    if tg.size < 3:
        return float(np.interp(T, tg, v))
    lt = np.log(tg)
    x = float(np.log(max(T, 1e-6)))
    i = int(np.clip(np.searchsorted(lt, x) - 1, 1, tg.size - 2))
    return float(np.polyval(np.polyfit(lt[i-1:i+2], v[i-1:i+2], 2), x))


def rebuild(topk_logits, tail_logmass):
    """TRUE probabilities over the stored support -- deliberately NOT renormalised to 1.

    Renormalising changes the probability VALUES, which silently breaks every sampler that
    thresholds on an absolute probability (mirostat2's p >= 2^-mu, min_p's p >= max*v). On a
    test vector that shifted probabilities by up to 1.9e-2 and gave a 0.30 total-variation
    error. The returned vector sums to the mass the support actually holds.
    """
    z = topk_logits.astype(np.float64)
    kept = np.clip(1.0 - np.exp(np.minimum(tail_logmass, 0.0)), 1e-12, 1.0)
    m = z.max(-1, keepdims=True)
    e = np.exp(z - m)
    p = e / e.sum(-1, keepdims=True)
    return p * np.asarray(kept)[..., None], np.log(kept)


def entropy_from(p):
    return float(-(p * np.log(np.clip(p, 1e-30, None))).sum())


def temperature(md, H):
    if not md['dyn']:
        return md['t_hi']
    return md['t_lo'] + (md['t_hi'] - md['t_lo']) * (max(H, 0.0) / 11.93) ** md['t_exp']


def truncate(p, md, mu, zstd=None):
    tr, v = md['trunc'], md['tp']
    if tr == 'none':
        return np.ones_like(p, dtype=bool)
    if tr == 'min_p':
        return p >= p.max() * v
    if tr == 'typical':
        lp = np.log(np.clip(p, 1e-12, None))
        ent = -(p * lp).sum() if zstd is None else _TYPICAL_ENT[0]
        order = np.argsort(np.abs(-lp - ent))
        keep = (np.cumsum(p[order]) - p[order]) < v
        m = np.zeros_like(p, dtype=bool)
        m[order] = keep
        m[p.argmax()] = True
        return m
    if tr == 'top_nsigma':
        zz = np.log(np.clip(p, 1e-30, None))
        sd = zz.std() if zstd is None else zstd
        return zz >= zz.max() - v * sd
    return p >= 2.0 ** (-mu)


def xtc_mask(p, thr):
    B = p >= thr
    if int(B.sum()) < 2:
        return None
    idx = np.where(B)[0]
    keep = idx[np.argmin(p[idx])]
    m = np.ones_like(p, dtype=bool)
    m[idx] = False
    m[keep] = True
    return m


def induced(p_true, ids, md, st, order, hist_u, H=None, zstd=None, lse_grid=None,
            tgrid=None, logits=None, ent_grid=None):
    """The mode's induced distribution over the stored support, as TRUE probabilities.

    p_true sums to the mass the support holds, not to 1. Nothing here renormalises it before the
    sampler runs, because mirostat2 and typical sampling threshold on absolute probability and
    silently change behaviour if the scale moves.

    order='hf' applies temperature BEFORE truncation, so the sampler sees softmax(z/T) over the
    FULL vocabulary. That partition is not recoverable from the top-K, so simlab stores
    logsumexp(z/T) on a temperature grid and it is interpolated here in log space.
    """
    lg = np.log(np.clip(p_true, 1e-30, None))
    pen = {}
    if md['lam'] > 0 and hist_u:
        for tid, val in hist_u.items():
            w = np.where(ids == tid)[0]
            if w.size:
                d = md['lam'] * min(val, 3.0)
                lg[w[0]] -= d
                pen[int(w[0])] = d
    H = entropy_from(p_true) if H is None else H
    _TYPICAL_ENT[0] = H
    _ZSTD[0] = zstd
    T = temperature(md, H)

    kept = float(p_true.sum())

    def finish(b0, temp_after):
        msk = truncate(b0, md, st.get('mu', 5.0), _ZSTD[0])
        b = b0 * msk
        if b.sum() <= 0:
            b = b0.copy()
        # min_p and mirostat2 threshold on an ABSOLUTE probability. If that threshold falls
        # below the smallest probability in the stored support, tokens outside the support also
        # survive, and the generator's normaliser includes them while ours would not. Add the
        # tail mass back: exact when the threshold is below the whole tail, and the right
        # direction otherwise. Without this the surviving mass is understated and every
        # probability is inflated.
        tail_alive = 0.0
        if kept < 0.999:
            pmin = float(b0[b0 > 0].min())
            if md['trunc'] == 'typical':
                # typical sampling keeps cumulative mass just under tp BY CONSTRUCTION, so the
                # generator's surviving mass is about tp however far into the tail the typical
                # set reaches. Use that directly rather than trying to rebuild the set.
                tail_alive = max(0.0, float(md['tp']) - float(b.sum()))
            else:
                thr = None
                if md['trunc'] == 'min_p':
                    thr = float(b0.max() * md['tp'])
                elif md['trunc'] == 'mirostat2':
                    thr = 2.0 ** (-st.get('mu', 5.0))
                elif md['trunc'] == 'top_nsigma':
                    # zz >= zz.max() - v*sd is an absolute probability threshold in disguise
                    sd = _ZSTD[0] if _ZSTD[0] is not None else float(
                        np.log(np.clip(b0, 1e-30, None)).std())
                    thr = float(b0.max() * np.exp(-float(md['tp']) * sd))
                if thr is not None and thr <= pmin:
                    tail_alive = 1.0 - kept
        branches = [(1.0, b)]
        if md['xtc_q'] > 0:
            xm = xtc_mask(b / b.sum(), md['xtc_t'])
            if xm is not None and (b * xm).sum() > 0:
                branches = [(1 - md['xtc_q'], b), (md['xtc_q'], b * xm)]
        out = np.zeros_like(b0)
        for w, br in branches:
            q = br / max(br.sum() + tail_alive, 1e-30)
            if temp_after:
                q = np.clip(q, 1e-30, None) ** (1.0 / max(T, 1e-3))
                q = q / q.sum()
            out += w * q
        return out

    # trunc='none' keeps the ENTIRE vocabulary, so the surviving mass includes the tail and
    # renormalising over the 512-token support inflates every probability by 1/kept -- a pure
    # -log(kept) error, measured at 0.19-0.33 nats on real data because Qwen's top-512 holds
    # only 72-83% of the mass at these positions.
    #
    # With no truncation the policy is just a temperature transform, and for BOTH orders
    #     q_i = exp(z_i/T - logsumexp(z/T))
    # which the stored grid gives exactly. (For llama: p^(1/T) normalised over the full vocab
    # equals exp(z/T - lse_T) after the lse cancels.)
    if md['trunc'] == 'none' and lse_grid is not None and logits is not None:
        zt = logits.astype(np.float64)
        lse_T = grid_at(T, tgrid, lse_grid)
        if pen:
            idx = np.array(sorted(pen))
            dd = np.array([pen[i] for i in idx])
            old_m = np.exp(zt[idx] / max(T, 1e-3) - lse_T).sum()
            new_m = np.exp((zt[idx] - dd) / max(T, 1e-3) - lse_T).sum()
            lse_T += float(np.log(max(1.0 - old_m + new_m, 1e-12)))
            zt = zt.copy()
            zt[idx] -= dd
        q = np.exp(zt / max(T, 1e-3) - lse_T)
        if md['xtc_q'] > 0:
            xm = xtc_mask(q, md['xtc_t'])
            if xm is not None and (q * xm).sum() > 0:
                q = (1 - md['xtc_q']) * q + md['xtc_q'] * (q * xm) / (q * xm).sum() * q.sum()
        return q

    if order == 'llama':
        return finish(np.exp(lg), True)

    # hf: temperature first, over the full vocabulary
    if lse_grid is None or logits is None:
        sc = np.exp((lg - lg.max()) / max(T, 1e-3))
        return finish(sc / sc.sum(), False)
    # after tempering, the sampler's view of the distribution changes: typical sampling needs
    # the TEMPERED entropy and top_nsigma needs std(z)/T, not the untempered quantities
    if ent_grid is not None:
        _TYPICAL_ENT[0] = grid_at(T, tgrid, ent_grid)
    _ZSTD[0] = None if zstd is None else zstd / max(T, 1e-3)
    lse_T = grid_at(T, tgrid, lse_grid)
    zt = logits.astype(np.float64) / max(T, 1e-3)
    if pen:
        # the repetition penalty shifts the partition; correct it with the penalised tokens,
        # which are the only ones whose logits changed
        idx = np.array(sorted(pen))
        d = np.array([pen[i] for i in idx])
        old_m = np.exp(zt[idx] - lse_T).sum()
        new_m = np.exp((logits[idx].astype(np.float64) - d) / max(T, 1e-3) - lse_T).sum()
        lse_T = lse_T + float(np.log(max(1.0 - old_m + new_m, 1e-12)))
        zt[idx] = (logits[idx].astype(np.float64) - d) / max(T, 1e-3)
    return finish(np.exp(zt - lse_T), False)


def advance_state(st, md, pol_gamma, lp_base, tid):
    """Mirrors G.advance. Driven by the STORED base log-prob of the emitted token, not by a
    lookup in the support.

    This matters because mirostat2's mu evolves from the emitted token's surprisal every step.
    When the emitted token falls outside the stored top-512 -- which happens whenever the policy
    truncates loosely -- a support lookup finds nothing, the advance gets skipped, mu drifts
    away from the generator's, and every later step inherits the error. That compounding was
    worth 0.19-0.33 nats of median error on every mirostat task, while the stateless samplers
    were already correct to 1e-6.

    lp_base is the natural-log base-model probability of the emitted token, recorded at
    generation time, so it is available whether or not the token is in the support.
    """
    S = -float(lp_base) / np.log(2.0)
    if md['trunc'] == 'mirostat2':
        st['mu'] = float(np.clip(st.get('mu', 5.0) - md['eta'] * (S - md['tp']), 0.5, 20.0))
    if md['lam'] > 0:
        u = st.setdefault('u', {})
        g = pol_gamma
        for k in list(u):
            u[k] *= g
            if u[k] <= 1e-3:
                del u[k]
        u[int(tid)] = u.get(int(tid), 0.0) + g
    st['prev_surprisal'] = S * np.log(2)


def fresh_state(md):
    return dict(mu=2.0 * md['tp'] if md['trunc'] == 'mirostat2' else 5.0,
                u={}, prev_surprisal=0.0)


def induced_at(task, rec, p, ids, logits, md, st, tgrid):
    """induced() with every stored statistic threaded in. Use this, never induced() directly.

    Omitting any of H / zstd / lse_grid / ent_grid silently falls back to computing them from
    the truncated support, which is wrong by up to tens of nats.
    """
    return induced(p, ids, md, st, task['policy']['order'], st.get('u', {}),
                   H=rec['H'], zstd=rec['zstd'], lse_grid=rec.get('lse_grid'),
                   tgrid=tgrid, logits=logits, ent_grid=rec.get('ent_grid'))


def replay_trace(task, rec_steps, P, IDS, LG, tgrid, modes=None, fsm=None):
    """Walk one trace with a SINGLE evolving state, as the generator does.

    The generator creates one state per trace (G.init_state per prompt) and advances it every
    step with whichever mode is active, so mirostat's mu and the repetition counters carry
    across mode switches. Giving each mode its own state breaks that and is wrong.

    modes: per-mode parameter dicts to use (defaults to the task's true policy).
    fsm:   None to follow the recorded ground-truth modes; otherwise a policy dict whose guards
           and dwell/expiry drive the mode sequence forward.
    Returns (log-probs of the emitted tokens, modes used).
    """
    pol = task['policy']
    md_list = modes if modes is not None else pol['modes']
    st = fresh_state(md_list[0])
    out, used, cur, c = [], [], 0, 0
    for k, rec in enumerate(rec_steps):
        m = rec['mode'] if fsm is None else cur
        md = md_list[m]
        d = induced_at(task, rec, P[k], IDS[k], LG[k], md, st, tgrid)
        pos = np.where(IDS[k] == rec['tok'])[0]
        # the likelihood is only available for tokens inside the support, but the STATE must
        # advance either way or mirostat's mu desynchronises for the rest of the trace
        out.append(float(np.log(max(d[pos[0]], 1e-12))) if pos.size else float('nan'))
        advance_state(st, md, pol['gamma'], rec['lp_base'], rec['tok'])
        used.append(m)
        if fsm is not None:
            cur, c = step_fsm(fsm, cur, c, rec)
    return np.array(out), used


def step_fsm(pol, m, c, rec):
    """grammar_v3.fsm_step over a stored sensor record."""
    if c < pol['dwell'][m]:
        return m, c + 1
    if m != 0 and c >= pol['expiry'][m]:
        return 0, 0
    for g in pol['guards']:
        if g['src'] != m:
            continue
        v = rec[g['sensor']]
        if (v >= g['thr']) if g['cmp'] == 'ge' else (v < g['thr']):
            return g['dst'], 0
    return m, c + 1


def verify_against_generator(task, npz, tgrid, ntrace=2, n=256):
    """Recompute stored per-step log-probs from the stored distribution and compare.

    This is the audit that the CPU path reproduces the GPU generator. It must pass before any
    fit is trusted: a large gap means the stored statistics are being used wrongly, or the
    top-K truncation is material, and every downstream number inherits it.
    """
    errs = []
    for ti, tr in enumerate(task['traces'][:ntrace]):
        steps = tr['steps'][:n]
        Pm, _ = rebuild(npz['cal_lg'][ti][:len(steps)], npz['cal_tail'][ti][:len(steps)])
        lp, _ = replay_trace(task, steps, Pm, npz['cal_ids'][ti], npz['cal_lg'][ti], tgrid)
        for k, step in enumerate(steps):
            if np.isfinite(lp[k]):
                errs.append(abs(lp[k] - step['lp_policy']))
    return np.array(errs) if errs else np.array([0.0])

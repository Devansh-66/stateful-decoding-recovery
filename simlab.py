"""Fresh simulation recording enough that no second GPU run is ever needed.

    python simlab.py

WHY A FRESH RUN. The released dataset is unreproducible: make_dataset.py samples with
torch.multinomial and never calls torch.manual_seed, so the hidden policies existed only in that
process's memory and no mode labels can be recovered for it.

THE DESIGN PRINCIPLE. Store the base model's next-token distribution at every step. Every sampler
in the grammar -- min_p, typical, top_nsigma, mirostat2, XTC, repetition penalty, dynamic
temperature -- is a deterministic function of that distribution and its parameters. With it
stored, any policy's induced distribution at any step is recomputable offline on CPU, so every
fit, baseline, ablation and diagnostic becomes a CPU job. It is the only quantity here that
cannot be reconstructed without repeating the GPU run.

WHAT IS RECORDED, per step of every calibration trace and every scored prefix:
  topk_ids / topk_logits / tail_logmass   the base distribution, truncated to TOPK
  sensors                                 H, G, U, D, P -- what the FSM gates on
  mode, dwell, mu                         ground truth state
  margins, predicates, fired, blocked     transition provenance: why the FSM did what it did
  logp_under_mode[M]                      the emitted token's log-prob under EVERY mode
  tv_between_modes                        pairwise total variation between mode distributions
  u                                       the uniform draw the sampler consumed

logp_under_mode and tv_between_modes are the per-step mode discriminability. From them the
Bayes-optimal mode posterior under the TRUE parameters is computable, which separates "the
observation does not determine the mode" from "the predictor is bad" -- the decomposition that
makes a negative result attributable rather than merely reported.

CONDITIONS, balanced across tasks:
  modes      1 (genuinely stateless) / 2 / 3
  dwell      on (dwell locks + expiry, non-Markov in the sensors) / off (mode is a function of
             the current sensor vector, so pointwise identifiable)
  margin     thresholds placed at designed quantiles of each sensor's MEASURED distribution:
             well-bracketed (40-60th pct, many crossings) or poorly-bracketed (5th/95th, few).
             Two-pass: a base-model-only trace measures the sensor distribution first. This
             converts an incidental observation -- that thresholds are identifiable only to an
             interval -- into a controlled main effect, and cannot be added afterwards.

Every task seeds torch from its own index, and the consumed uniform draws are stored, so any
trace can be verified without regenerating it.
"""
import os, sys, csv, json, math, random, time, hashlib, platform
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import grammar_v3 as G
import probe as M

OUT = os.environ.get('OUT', './sim')
NTASK = int(os.environ.get('NTASK', 250))
NTRACE = int(os.environ.get('NTRACE', 16))
TRACELEN = int(os.environ.get('TRACELEN', 256))
NITEM = int(os.environ.get('NITEM', 40))
OVERSAMPLE = float(os.environ.get('OVERSAMPLE', 2.5))
PREFIX_LEN = int(os.environ.get('PREFIX_LEN', 24))
TOPK = int(os.environ.get('TOPK', 512))
NPROMPT_POOL = int(os.environ.get('NPROMPT_POOL', 48))   # prompts crossed with policies
NREPLICATE = int(os.environ.get('NREPLICATE', 10))       # duplicate policies, new trace seeds
VERIFY_FULL = int(os.environ.get('VERIFY_FULL', 5))      # tasks also checked against full vocab
SEED = 71294
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
T0 = time.time()
TGRID_T = None   # set in main(), after the device is known
SENSORS = ['H', 'G', 'U', 'D', 'P']
# temperature grid for the stored partition; bounds cover BOUNDS['t_lo']..['t_hi'] in grammar_v3
TGRID = [0.25 * (1.1 ** k) for k in range(26)]


def log(m):
    print('[%7.1fs] %s' % (time.time() - T0, m), flush=True)


# ------------------------------------------------------------------ conditions

def condition_of(i):
    """Balanced assignment. dwell is vacuous at M=1, so the cell structure is 5-fold."""
    cells = [(1, False), (2, False), (2, True), (3, False), (3, True)]
    m, dw = cells[i % len(cells)]
    margin = 'well' if (i // len(cells)) % 2 == 0 else 'poor'
    return dict(M=m, dwell=dw, margin=margin)


def canonical(pol):
    """Mode-relabel-invariant form, so policies identical up to relabelling hash identically.

    Without this, two equivalent policies look distinct and apparent diversity is inflated --
    which matters because the central claim is cross-task amortization and a reviewer will ask
    for score against distance-to-nearest-training-policy.
    """
    order = sorted(range(pol['M']), key=lambda i: (pol['modes'][i]['trunc'],
                                                   round(pol['modes'][i]['tp'], 4)))
    rank = {o: j for j, o in enumerate(order)}
    md = [{k: v for k, v in pol['modes'][o].items()} for o in order]
    gd = sorted([dict(src=rank[g['src']], dst=rank[g['dst']], sensor=g['sensor'],
                      cmp=g['cmp'], thr=round(g['thr'], 4)) for g in pol['guards']],
                key=lambda g: (g['src'], g['dst'], g['sensor']))
    c = dict(M=pol['M'], order=pol['order'], gamma=pol['gamma'], modes=md, guards=gd,
             dwell=[pol['dwell'][o] for o in order], expiry=[pol['expiry'][o] for o in order])
    return c, hashlib.sha1(json.dumps(c, sort_keys=True).encode()).hexdigest()[:16]


def build_policy(rng, cond, sensor_quantiles):
    """Draw a policy, then force the condition and place thresholds at designed quantiles."""
    p = G.sample_policy(rng)
    p['M'] = cond['M']
    while len(p['modes']) < p['M']:
        p['modes'].append(G.sample_mode(rng))
    p['modes'] = p['modes'][:p['M']]
    p['dwell'] = ([rng.choice([0, 3]) for _ in range(p['M'])] if cond['dwell']
                  else [0] * p['M'])
    p['expiry'] = ([rng.choice([8, 16, 32]) for _ in range(p['M'])] if cond['dwell']
                   else [999] * p['M'])
    if p['M'] == 1:
        p['guards'] = []
        return p
    pairs = [(0, 1), (1, 0)] if p['M'] == 2 else [(0, 1), (1, 2), (1, 0), (2, 0)]
    guards = []
    for src, dst in pairs:
        s = rng.choice(SENSORS)
        q = sensor_quantiles[s]
        if cond['margin'] == 'well':
            lo, hi = np.percentile(q, 40), np.percentile(q, 60)
            thr = float(rng.uniform(lo, hi))
        else:
            thr = float(np.percentile(q, 5 if rng.random() < 0.5 else 95))
        guards.append(dict(src=src, dst=dst, sensor=s, cmp=rng.choice(['ge', 'lt']),
                           thr=round(thr, 4)))
    p['guards'] = guards
    return p


# ------------------------------------------------------------------ rollout

def predicate_state(pol, s, st):
    """Which gates are true, by how much, and what the FSM is allowed to do about it."""
    vals, marg = [], []
    for g in pol['guards']:
        v = s[g['sensor']]
        true = (v >= g['thr']) if g['cmp'] == 'ge' else (v < g['thr'])
        d = v - g['thr']
        vals.append(bool(true))
        marg.append(round(float(d if g['cmp'] == 'ge' else -d), 5))
    blocked = bool(st['c'] < pol['dwell'][st['m']] and any(
        vals[j] for j, g in enumerate(pol['guards']) if g['src'] == st['m']))
    expiring = bool(st['m'] != 0 and st['c'] >= pol['expiry'][st['m']])
    return vals, marg, blocked, expiring


@torch.no_grad()
def rollout(model, tok, prompts, pol, nstep, seed, want_logits=True):
    """Generate nstep tokens per prompt, recording everything.

    SHADOW STATE CONVENTION: a separate sampler state is evolved for every mode as if that mode
    were always active, so logp_under_mode is the log-prob that mode's sampler would have given
    had it been running throughout. The alternative -- freezing inactive state -- makes the
    counterfactual depend on occupancy history and is not what a mode-posterior needs. Declared
    here and stored in the task record.
    """
    g = torch.Generator(device=DEV).manual_seed(seed)
    n = len(prompts)
    enc = tok([t for _, t in prompts], return_tensors='pt', padding=True).to(DEV)
    ids, attn = enc.input_ids, enc.attention_mask
    pos = (attn.cumsum(-1) - 1).clamp_min(0)
    past, cur = None, ids
    st = [G.init_state(pol) for _ in range(n)]
    shadow = [[G.init_state(pol) for _ in range(pol['M'])] for _ in range(n)]
    toks = [[] for _ in range(n)]
    rec = [[] for _ in range(n)]
    # nstep + 1: the last slot is the SCORED position, reached by the extra forward below.
    # Truncation samplers need the whole distribution to decide what survives, so storing only
    # the candidates' base log-probs there would make the scored step uncomputable offline.
    tk_ids = np.zeros((n, nstep + 1, TOPK), dtype=np.int32)
    tk_lg = np.zeros((n, nstep + 1, TOPK), dtype=np.float16)
    tail = np.zeros((n, nstep + 1), dtype=np.float32)
    ent_last = np.zeros((n, len(TGRID)), dtype=np.float32)
    lse_last = np.zeros((n, len(TGRID)), dtype=np.float32)
    stat_last = np.zeros((n, 2), dtype=np.float32)      # H, zstd at the scored position

    for step in range(nstep):
        out = model(input_ids=cur, attention_mask=attn, position_ids=pos,
                    past_key_values=past, use_cache=True)
        past = out.past_key_values
        z = out.logits[:, -1, :].half()
        nxt = []
        for b in range(n):
            zf = z[b].float()
            p_raw = torch.softmax(zf, -1)
            lse = torch.logsumexp(zf, -1)
            tv, ti = torch.topk(zf, TOPK)
            # top_nsigma thresholds on the std of log-probs over the FULL vocabulary and
            # typical sampling on the FULL entropy. Neither survives truncation to top-K -- on a
            # test vector the std was 3.00 full vs 0.76 truncated -- so both are stored.
            zstd = float(torch.log(p_raw.clamp_min(1e-30)).std())
            # 'hf'-order policies apply temperature BEFORE truncation, so the sampler sees
            # softmax(z/T) over the FULL vocabulary. That partition cannot be recovered from the
            # top-K, so it is stored on a temperature grid and interpolated in log space.
            zg = zf[None, :] / TGRID_T[:, None]
            lseg = torch.logsumexp(zg, -1)
            # typical sampling thresholds on the entropy of the distribution it is given, which
            # for hf order is the TEMPERED one. Stored on the same grid as the partition.
            pg = torch.softmax(zg, -1)
            entg = -(pg * torch.log(pg.clamp_min(1e-30))).sum(-1)
            if want_logits:
                tk_ids[b, step] = ti.cpu().numpy()
                tk_lg[b, step] = tv.cpu().numpy().astype(np.float16)
                kept = torch.exp(tv - lse).sum()
                tail[b, step] = float(torch.log((1.0 - kept).clamp_min(1e-30)))

            s = G.sensors(p_raw, z[b], st[b], step)
            vals, marg, blocked, expiring = predicate_state(pol, s, st[b])
            m_before = st[b]['m']
            d, praw, md = G.policy_dist(z[b], pol, st[b], step)
            u = float(torch.rand(1, generator=g, device=DEV))
            t = int(torch.searchsorted(torch.cumsum(d, 0), torch.tensor(u, device=DEV)).clamp(
                0, d.numel() - 1))

            # counterfactual: the emitted token's log-prob under every mode, and how far apart
            # the modes' induced distributions are at this step
            lpm, dists = [], []
            # the counterfactual policy must have no guards, no dwell and no expiry, or the FSM
            # would reset a pinned shadow mode to 0 and the counterfactual would be wrong
            pinned = {**pol, 'guards': [], 'dwell': [0] * pol['M'],
                      'expiry': [10 ** 9] * pol['M']}
            for mi in range(pol['M']):
                stt = shadow[b][mi]
                stt['m'], stt['c'] = mi, 0
                dm, prm, mdm = G.policy_dist(z[b], pinned, stt, step)
                lpm.append(round(float(torch.log(dm[t].clamp_min(1e-12))), 5))
                dists.append(dm)
            tvs = []
            for a in range(pol['M']):
                for c in range(a + 1, pol['M']):
                    tvs.append(round(float(0.5 * (dists[a] - dists[c]).abs().sum()), 5))

            rec[b].append(dict(H=round(s['H'], 5), G=round(s['G'], 5), U=round(s['U'], 5),
                               zstd=round(zstd, 5), lse_grid=[round(float(x), 5) for x in lseg],
                               ent_grid=[round(float(x), 5) for x in entg], kept=round(float(torch.exp(tv - lse).sum()), 6),
                               D=s['D'], P=s['P'], mode=m_before, mode_after=st[b]['m'],
                               dwell=st[b]['c'], mu=round(st[b]['mu'], 4),
                               pred=vals, margins=marg, blocked=blocked, expiring=expiring,
                               u=round(u, 7), tok=t,
                               lp_policy=round(float(torch.log(d[t].clamp_min(1e-12))), 5),
                               lp_base=round(float(torch.log(p_raw[t].clamp_min(1e-12))), 5),
                               lp_mode=lpm, tv=tvs))
            G.advance(st[b], pol, md, praw, t)
            for mi in range(pol['M']):
                G.advance(shadow[b][mi], pinned, pol['modes'][mi], praw, t)
            toks[b].append(t)
            nxt.append(t)
        cur = torch.tensor(nxt, device=DEV).unsqueeze(1)
        attn = torch.cat([attn, torch.ones(n, 1, dtype=attn.dtype, device=DEV)], 1)
        pos = pos[:, -1:] + 1

    out = model(input_ids=cur, attention_mask=attn, position_ids=pos,
                past_key_values=past, use_cache=True)
    zl = out.logits[:, -1, :].half()
    if want_logits:
        for b in range(n):
            zf = zl[b].float()
            pr = torch.softmax(zf, -1)
            lse = torch.logsumexp(zf, -1)
            tv, ti = torch.topk(zf, TOPK)
            tk_ids[b, nstep] = ti.cpu().numpy()
            tk_lg[b, nstep] = tv.cpu().numpy().astype(np.float16)
            tail[b, nstep] = float(torch.log((1.0 - torch.exp(tv - lse).sum()).clamp_min(1e-30)))
            zg = zf[None, :] / TGRID_T[:, None]
            lse_last[b] = torch.logsumexp(zg, -1).cpu().numpy()
            pg = torch.softmax(zg, -1)
            ent_last[b] = (-(pg * torch.log(pg.clamp_min(1e-30))).sum(-1)).cpu().numpy()
            stat_last[b] = [float(-(pr * torch.log(pr.clamp_min(1e-30))).sum()),
                            float(torch.log(pr.clamp_min(1e-30)).std())]
    return toks, rec, st, zl, (tk_ids, tk_lg, tail, lse_last, ent_last, stat_last)


@torch.no_grad()
def measure_sensors(model, tok, prompts, nstep, seed):
    """Pass 1: base-model-only generation, to measure each sensor's realised distribution.

    Doubles as the per-prompt reference trace needed to show the sensor trajectory carries
    policy information rather than prompt information.
    """
    flat = G.sample_policy(random.Random(0))
    flat['M'] = 1
    flat['guards'] = []
    flat['dwell'] = [0]
    flat['expiry'] = [999]
    flat['modes'] = [{**flat['modes'][0], 'trunc': 'none', 'dyn': False, 't_hi': 1.0,
                      'lam': 0.0, 'xtc_q': 0.0}]
    toks, rec, _, _, _ = rollout(model, tok, prompts, flat, nstep, seed, want_logits=False)
    q = {s: np.array([r[s] for tr in rec for r in tr], dtype=np.float32) for s in SENSORS}
    return q, toks, rec


def main():
    os.makedirs(OUT, exist_ok=True)
    cal_prompts = M.load_prompts(os.environ.get('PC', './raw/prompts_calibration.csv'))
    qry_prompts = M.load_prompts(os.environ.get('PQ', './raw/prompts_query.csv'))
    rngp = random.Random(SEED)
    pool_c = [cal_prompts[rngp.randrange(len(cal_prompts))] for _ in range(NPROMPT_POOL)]
    pool_q = [qry_prompts[rngp.randrange(len(qry_prompts))] for _ in range(NPROMPT_POOL)]
    log('prompt pool: %d calibration, %d query (crossed with policies)' % (len(pool_c), len(pool_q)))

    name = os.environ.get('MODEL', 'Qwen/Qwen2.5-0.5B')
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        name, dtype=torch.float16 if DEV == 'cuda' else torch.float32).to(DEV).eval()
    global TGRID_T
    TGRID_T = torch.tensor(TGRID, device=DEV, dtype=torch.float32)

    env = dict(model=name, device=DEV, torch=torch.__version__, python=platform.python_version(),
               dtype=str(next(model.parameters()).dtype), topk=TOPK,
               shadow_state='evolved as if always active', seed=SEED, tgrid=TGRID,
               tracelen=TRACELEN, ntrace=NTRACE, nitem=NITEM, prefix_len=PREFIX_LEN)
    json.dump(env, open(os.path.join(OUT, 'env.json'), 'w'), indent=1)
    log('env: %s' % json.dumps(env))
    log('budgets derivable by subsetting: 8x32=256, 8x128=1024, 16x128=2048, 16x256=%d tokens'
        % (NTRACE * TRACELEN))

    path = os.path.join(OUT, 'tasks.jsonl')
    done = set()
    if os.path.exists(path):
        done = {json.loads(l)['task_id'] for l in open(path, encoding='utf-8')}
        log('resuming: %d tasks present' % len(done))
    f = open(path, 'a', encoding='utf-8')
    kept, skipped = 0, 0
    seen_policies = {}

    for i in range(NTASK):
        tid = 'T%05d' % i
        if tid in done:
            continue
        cond = condition_of(i)
        rng = random.Random(SEED * 7919 + i)
        # crossing: every task draws its prompts from the shared pool at a rotating offset, so
        # each prompt recurs across policies and each policy sees many prompts
        off = (i * 7) % NPROMPT_POOL
        cp = [pool_c[(off + k) % NPROMPT_POOL] for k in range(NTRACE)]
        qp = [pool_q[(off + k) % NPROMPT_POOL] for k in range(int(NITEM * OVERSAMPLE))]

        # replicates: reuse an earlier task's policy with fresh trace seeds
        rep_of = None
        if i >= NTASK - NREPLICATE and seen_policies:
            rep_of = sorted(seen_policies)[(i - (NTASK - NREPLICATE)) % len(seen_policies)]
            pol = seen_policies[rep_of]
            cond = dict(M=pol['M'], dwell=any(d > 0 for d in pol['dwell']),
                        margin='inherited', replicate_of=rep_of)
        else:
            q, btoks, brec = measure_sensors(model, tok, cp[:4], 64, SEED * 29 + i)
            pol = None
            for attempt in range(60):
                cand = build_policy(random.Random(SEED * 1000003 + i * 997 + attempt), cond, q)
                if cand['M'] == 1:
                    pol = cand
                    break
                _, rc, _, _, _ = rollout(model, tok, cp[:4], cand, 32,
                                         SEED * 13 + i * 101 + attempt, want_logits=False)
                modes = [r['mode'] for tr in rc for r in tr]
                sh = [modes.count(m) / len(modes) for m in range(cand['M'])]
                if min(sh) >= 0.05:
                    pol = cand
                    break
            if pol is None:
                skipped += 1
                log('  %s SKIPPED: no policy reached 5%% occupancy in all modes' % tid)
                continue

        ctoks, crec, _, _, cl = rollout(model, tok, cp, pol, TRACELEN, SEED * 17 + i * 7919)
        ptoks, prec, pst, pz, pl = rollout(model, tok, qp, pol, PREFIX_LEN,
                                           SEED * 19 + i * 104729)
        items = []
        for b in range(len(qp)):
            got = M.scored_item(pz[b], pol, pst[b], PREFIX_LEN, rng)
            if got is None:
                continue
            cands, ti, pt, pb, maxp, _ = got
            if maxp < M.MIN_MAXP:
                continue
            lp = torch.log_softmax(pz[b].float(), -1)
            rk = torch.argsort(torch.argsort(-pz[b].float()))
            items.append(dict(idx=b, prompt_id=qp[b][0], prefix=ptoks[b], candidates=cands,
                              true_index=ti, lp_policy=round(math.log(max(pt, 1e-12)), 6),
                              lp_base=round(math.log(max(pb, 1e-12)), 6),
                              cand_lp_base=[round(float(lp[c]), 5) for c in cands],
                              cand_rank=[int(rk[c]) for c in cands],
                              true_is_argmax=bool(cands[ti] == int(pz[b].argmax())),
                              steps=prec[b], mode_at_item=pst[b]['m'],
                              scored=dict(H=round(float(pl[5][b][0]), 5),
                                          zstd=round(float(pl[5][b][1]), 5),
                                          lse_grid=[round(float(x), 5) for x in pl[3][b]],
                                          ent_grid=[round(float(x), 5) for x in pl[4][b]])))
            if len(items) >= NITEM:
                break
        if not items:
            skipped += 1
            log('  %s SKIPPED: no items survived the MIN_MAXP filter' % tid)
            continue

        canon, chash = canonical(pol)
        st_all = [s for tr in crec for s in tr]
        occ = [float(np.mean([s['mode'] == m for s in st_all])) for m in range(pol['M'])]
        switches = int(np.sum([tr[k]['mode'] != tr[k - 1]['mode']
                               for tr in crec for k in range(1, len(tr))]))
        rec_out = dict(task_id=tid, condition=cond, policy=pol, canonical=canon,
                       policy_hash=chash, occupancy=occ, switches=switches,
                       degenerate=bool(min(occ) < 0.05),
                       trace_prompts=[p[0] for p in cp],
                       traces=[dict(prompt_id=cp[b][0], tokens=ctoks[b], steps=crec[b])
                               for b in range(len(cp))],
                       items=items)
        f.write(json.dumps(rec_out) + '\n')
        f.flush()
        np.savez_compressed(os.path.join(OUT, '%s.npz' % tid),
                            cal_ids=cl[0], cal_lg=cl[1], cal_tail=cl[2],
                            itm_ids=pl[0][[it['idx'] for it in items]],
                            itm_lg=pl[1][[it['idx'] for it in items]],
                            itm_tail=pl[2][[it['idx'] for it in items]])
        if rep_of is None:
            seen_policies[tid] = pol
        kept += 1
        if kept % 10 == 0 or kept == 1:
            log('  %s | %d/%d | M=%d dwell=%s margin=%s | occ=%s switches=%d | %d items%s'
                % (tid, kept, NTASK, pol['M'], cond['dwell'], cond['margin'],
                   ','.join('%.2f' % x for x in occ), switches, len(items),
                   ' | REPLICATE of %s' % rep_of if rep_of else ''))
    f.close()
    log('done: %d kept, %d skipped' % (kept, skipped))
    summarise(path)


def summarise(path):
    rows = [json.loads(l) for l in open(path, encoding='utf-8')]
    if not rows:
        return
    log('')
    log('==== RUN SUMMARY ====')
    cells = {}
    for r in rows:
        k = (r['condition']['M'], r['condition']['dwell'], r['condition']['margin'])
        cells[k] = cells.get(k, 0) + 1
    for k in sorted(cells):
        log('  M=%d dwell=%-5s margin=%-4s : %d tasks' % (k[0], k[1], k[2], cells[k]))
    deg = sum(r['degenerate'] for r in rows)
    log('  degenerate (a mode below 5%% occupancy): %d of %d -- PRE-REGISTERED EXCLUSION'
        % (deg, len(rows)))
    hashes = {}
    for r in rows:
        hashes[r['policy_hash']] = hashes.get(r['policy_hash'], 0) + 1
    log('  distinct canonical policies: %d (duplicates: %d)'
        % (len(hashes), sum(v - 1 for v in hashes.values())))
    marg = np.array([abs(x) for r in rows for tr in r['traces'] for s in tr['steps']
                     for x in s['margins']])
    if marg.size:
        log('  guard margin |sensor-threshold|: median %.3f | frac < 0.05: %.4f'
            % (float(np.median(marg)), float((marg < 0.05).mean())))
    tv = np.array([x for r in rows for tr in r['traces'] for s in tr['steps'] for x in s['tv']])
    if tv.size:
        log('  inter-mode TV per step: median %.4f | frac < 0.01 (informationless): %.4f'
            % (float(np.median(tv)), float((tv < 0.01).mean())))
        log('    a high informationless fraction means the modes are not distinguishable at')
        log('    those steps by ANY method, which bounds what mode inference can achieve.')
    sw = np.array([r['switches'] for r in rows])
    log('  mode switches per task: median %d, 10-90%% %d-%d'
        % (int(np.median(sw)), int(np.percentile(sw, 10)), int(np.percentile(sw, 90))))
    log('  items per task: mean %.1f' % float(np.mean([len(r['items']) for r in rows])))


if __name__ == '__main__':
    main()

"""Real decoding mechanisms wrapped in a mode machine.

v2 kept the real mechanisms (min_p, typical, top_nsigma, mirostat, dynatemp, XTC, hidden
sampler order) but had only ~80 discrete structures. With 48 random candidates roughly five
already had the right truncation family AND order, so guessing the structure was a lottery
rather than a search: random 0.623, hill-climb 0.667, gate 0.707 -- all within 0.084.

Here each policy is a small STATE MACHINE. Every mode carries its own full sub-policy, and
transitions are guarded by conditions on sensors computed from the model's RAW distribution.
Structure count rises to ~3x10^5 even at M=2, because the guards multiply:

    M(2) x order(2) x trunc^M(25) x [sensor(5) x cmp(2)]^guards(2) x dwell(4) x expiry(16)

Two properties keep this fair rather than merely hard:

  * every sensor is computed on the RAW temperature-1 distribution, so it cannot see the
    policy's own parameters. A sensor threshold can therefore never trade off against a
    temperature, which is what created the m^T degeneracy in the first design.
  * all state is a deterministic function of the OBSERVED tokens. Replaying a trace under a
    candidate gives an exact likelihood in one pass -- no filtering over hidden paths.

Calibration is raised to 8 traces, because v2's gate reached only 0.707: with the structure
handed over it still could not pin the numbers, which is evidence-limited, and the fix for
that is more evidence.

    NTASK=3 python grammar_v3.py
"""
import os, sys, math, random, copy
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get('MODEL', 'Qwen/Qwen2.5-0.5B')
NTASK = int(os.environ.get('NTASK', 8))
NCAL = int(os.environ.get('NCAL', 8))
CALLEN = int(os.environ.get('CALLEN', 32))
NQ = int(os.environ.get('NQ', 40))
KCAND = int(os.environ.get('KCAND', 64))
HILL = int(os.environ.get('HILL', 150))
GATE = int(os.environ.get('GATE', 400))
EPS = float(os.environ.get('EPS', 1e-3))
SEED = 71294
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

OPENERS = ["The main reason", "In the morning", "She decided to", "One problem with",
           "After the meeting", "The difference between", "A common mistake", "He looked at",
           "The report said", "Most people think", "It turns out", "The city of",
           "During the winter", "They agreed that", "The first step", "Nobody expected"]
TOPICS = ["the weather", "a broken engine", "her new job", "the missing key", "local farming",
          "a long journey", "the price of bread", "an old photograph", "a difficult exam",
          "the train schedule", "the harvest", "a strange noise", "the election", "a lost letter"]

TRUNCS = ['none', 'min_p', 'typical', 'top_nsigma', 'mirostat2']
ORDERS = ['llama', 'hf']
GAMMAS = [0.60, 0.80, 0.90, 0.97]
SENSORS = ['H', 'G', 'U', 'D', 'P']
SENSOR_RANGE = dict(H=(0.5, 6.0), G=(0.2, 8.0), U=(0.5, 9.0), D=(4, 16), P=(6, 34))
TP_BOUNDS = {'none': (0.0, 1.0), 'min_p': (0.01, 0.35), 'typical': (0.15, 0.99),
             'top_nsigma': (0.4, 3.0), 'mirostat2': (2.0, 9.0)}
BOUNDS = dict(eta=(0.01, 0.35), t_lo=(0.30, 1.00), t_hi=(0.90, 1.80), t_exp=(0.3, 3.0),
              xtc_q=(0.0, 0.7), xtc_t=(0.03, 0.30), lam=(0.0, 3.5))


def sample_mode(rng):
    tr = rng.choice(TRUNCS)
    lo, hi = TP_BOUNDS[tr]
    return dict(trunc=tr, tp=round(rng.uniform(lo, hi), 4),
                eta=round(rng.uniform(0.02, 0.30), 4),
                dyn=rng.random() < 0.35,
                t_lo=round(rng.uniform(0.35, 0.95), 3),
                t_hi=round(rng.uniform(0.90, 1.70), 3),
                t_exp=round(rng.uniform(0.5, 2.5), 3),
                xtc_q=0.0 if rng.random() < 0.5 else round(rng.uniform(0.10, 0.60), 3),
                xtc_t=round(rng.uniform(0.05, 0.25), 3),
                lam=0.0 if rng.random() < 0.4 else round(rng.uniform(0.4, 3.0), 3))


def sample_policy(rng):
    M = 2 if rng.random() < 0.6 else 3
    pairs = [(0, 1), (1, 0)] if M == 2 else [(0, 1), (1, 2), (1, 0), (2, 0)]
    guards = []
    for src, dst in pairs:
        s = rng.choice(SENSORS)
        lo, hi = SENSOR_RANGE[s]
        guards.append(dict(src=src, dst=dst, sensor=s, cmp=rng.choice(['ge', 'lt']),
                           thr=round(rng.uniform(lo, hi), 3)))
    return dict(M=M, order=rng.choice(ORDERS), gamma=rng.choice(GAMMAS),
                modes=[sample_mode(rng) for _ in range(M)], guards=guards,
                dwell=[rng.choice([0, 3]) for _ in range(M)],
                expiry=[rng.choice([8, 16, 32, 999]) for _ in range(M)])


def active_keys(md):
    ks = ['tp'] if md['trunc'] != 'none' else []
    if md['trunc'] == 'mirostat2':
        ks.append('eta')
    ks.append('t_hi')
    if md['dyn']:
        ks += ['t_lo', 't_exp']
    if md['xtc_q'] > 0:
        ks += ['xtc_q', 'xtc_t']
    if md['lam'] > 0:
        ks.append('lam')
    return ks


def bounds_for(md, k):
    return TP_BOUNDS[md['trunc']] if k == 'tp' else BOUNDS[k]


# ---------------------------------------------------------------- sensors and the machine

def sensors(p_raw, z, st, pos):
    """Computed on the RAW distribution only, so they never see policy parameters."""
    lp = torch.log(p_raw.clamp_min(1e-12))
    H = float(-(p_raw * lp).sum())
    top2 = torch.topk(z.float(), 2).values
    G = float(top2[0] - top2[1])
    hist = st['hist'][-16:]
    return dict(H=H, G=G, U=st['prev_surprisal'], D=float(len(set(hist))),
                P=float(pos))


def fsm_step(pol, st, s):
    m, c = st['m'], st['c']
    if c < pol['dwell'][m]:
        st['c'] = c + 1
        return m
    if m != 0 and c >= pol['expiry'][m]:
        st['m'], st['c'] = 0, 0
        return 0
    for g in pol['guards']:
        if g['src'] != m:
            continue
        v = s[g['sensor']]
        if (v >= g['thr']) if g['cmp'] == 'ge' else (v < g['thr']):
            st['m'], st['c'] = g['dst'], 0
            return g['dst']
    st['c'] = c + 1
    return m


# ---------------------------------------------------------------- the per-token transform

def temperature(md, H):
    if not md['dyn']:
        return md['t_hi']
    return md['t_lo'] + (md['t_hi'] - md['t_lo']) * (max(H, 0) / 11.93) ** md['t_exp']


def truncate(p, md, mu):
    tr, v = md['trunc'], md['tp']
    if tr == 'none':
        return torch.ones_like(p, dtype=torch.bool)
    if tr == 'min_p':
        return p >= p.max() * v
    if tr == 'typical':
        lp = torch.log(p.clamp_min(1e-12))
        ent = -(p * lp).sum()
        order = (-lp - ent).abs().argsort()
        keep = (p[order].cumsum(0) - p[order]) < v
        m = torch.zeros_like(p, dtype=torch.bool)
        m[order] = keep
        m[p.argmax()] = True
        return m
    if tr == 'top_nsigma':
        zz = torch.log(p.clamp_min(1e-30))
        return zz >= zz.max() - v * zz.std()
    return p >= 2.0 ** (-mu)


def xtc_mask(p, thr):
    B = p >= thr
    if int(B.sum()) < 2:
        return None
    idx = torch.nonzero(B).squeeze(1)
    keep = idx[p[idx].argmin()]
    m = torch.ones_like(p, dtype=torch.bool)
    m[idx] = False
    m[keep] = True
    return m


def policy_dist(z, pol, st, pos):
    p_raw = F.softmax(z.float(), dim=-1)
    s = sensors(p_raw, z, st, pos)
    m = fsm_step(pol, st, s)
    md = pol['modes'][m]

    lg = z.float().clone()
    if md['lam'] > 0 and st['u']:
        idx = torch.tensor(list(st['u'].keys()), device=lg.device, dtype=torch.long)
        val = torch.tensor([min(v, 3.0) for v in st['u'].values()], device=lg.device)
        lg[idx] = lg[idx] - md['lam'] * val
    base = F.softmax(lg, dim=-1)
    T = temperature(md, s['H'])

    def finish(b0, temp_after):
        msk = truncate(b0, md, st['mu'])
        b = b0 * msk
        if float(b.sum()) <= 0:
            b = b0.clone()
        branches = [(1.0, b)]
        if md['xtc_q'] > 0:
            xm = xtc_mask(b / b.sum(), md['xtc_t'])
            if xm is not None and float((b * xm).sum()) > 0:
                branches = [(1 - md['xtc_q'], b), (md['xtc_q'], b * xm)]
        out = torch.zeros_like(b0)
        for w, br in branches:
            q = br / br.sum().clamp_min(1e-30)
            if temp_after:
                q = q.clamp_min(1e-30) ** (1.0 / T)
                q = q / q.sum()
            out += w * q
        return out

    if pol['order'] == 'llama':
        return finish(base, True), p_raw, md
    return finish(F.softmax(lg / max(T, 1e-3), dim=-1), False), p_raw, md


def init_state(pol):
    md = pol['modes'][0]
    return dict(m=0, c=0, u={}, hist=[], prev_surprisal=0.0,
                mu=2.0 * md['tp'] if md['trunc'] == 'mirostat2' else 5.0)


def advance(st, pol, md, p_raw, tok):
    S = -math.log2(max(float(p_raw[tok]), 1e-12))
    if md['trunc'] == 'mirostat2':
        st['mu'] = float(np.clip(st['mu'] - md['eta'] * (S - md['tp']), 0.5, 20.0))
    if md['lam'] > 0:
        g = pol['gamma']
        st['u'] = {k: v * g for k, v in st['u'].items() if v * g > 1e-3}
        st['u'][tok] = st['u'].get(tok, 0.0) + g
    st['prev_surprisal'] = S * math.log(2)
    st['hist'].append(tok)


# ---------------------------------------------------------------- data and fitting

@torch.no_grad()
def build_task(model, tok, rng, tries=6):
    """Resample any policy whose modes are not both genuinely occupied.

    If the guards rarely fire the machine sits in mode 0 forever, the other mode's parameters
    are never exercised, and no method can identify them. That is unfair rather than hard, so
    those policies must never reach the dataset."""
    for attempt in range(tries):
        true = sample_policy(rng)
        cal, qry, occ = [], [], []
        for tr in range(NCAL):
            ids = torch.tensor([tok.encode('%s %s' % (rng.choice(OPENERS),
                                                      rng.choice(TOPICS)))], device=DEV)
            st = init_state(true)
            for pos in range(CALLEN):
                z = model(input_ids=ids).logits[0, -1].half()
                d, praw, md = policy_dist(z, true, st, pos)
                nxt = int(torch.multinomial(d, 1))
                cal.append((tr, z, pos, nxt))
                occ.append(st['m'])
                advance(st, true, md, praw, nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
        share = [occ.count(m) / len(occ) for m in range(true['M'])]
        if min(share) >= 0.08:
            break
        print('    resample: mode occupancy %s' % ['%.2f' % s for s in share], flush=True)
    true['_occ'] = share
    ids = torch.tensor([tok.encode('%s %s' % (rng.choice(OPENERS), rng.choice(TOPICS)))],
                       device=DEV)
    st = init_state(true)
    # Every position from 0 is kept, with a flag for whether it is scored. The burn-in must be
    # replayed, not skipped: mode, mirostat mu, the decay field and the token history are all
    # functions of the prefix, so a candidate started fresh at position 10 evaluates every
    # scored position with the wrong state. Query contexts MUST carry their full prefix.
    for pos in range(NQ + 10):
        z = model(input_ids=ids).logits[0, -1].half()
        d, praw, md = policy_dist(z, true, st, pos)
        nxt = int(torch.multinomial(d, 1))
        qry.append((z, pos, nxt, float(d[nxt]), float(praw[nxt]), pos >= 10))
        advance(st, true, md, praw, nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=DEV)], dim=1)
    return true, cal, qry


def loglik(pol, cal):
    s, st, last = 0.0, None, -1
    for tr, z, pos, nxt in cal:
        if tr != last:
            st, last = init_state(pol), tr
        d, praw, md = policy_dist(z, pol, st, pos)
        s += math.log(max(float(d[nxt]), 1e-12))
        advance(st, pol, md, praw, nxt)
    return s


def predict(pol, qry):
    """Replay the whole generation; return predictions for the scored positions only."""
    out, st = [], init_state(pol)
    for z, pos, nxt, po, pb, scored in qry:
        d, praw, md = policy_dist(z, pol, st, pos)
        if scored:
            out.append(float(d[nxt]))
        advance(st, pol, md, praw, nxt)
    return out


def coord_list(pol):
    """Every fittable continuous coordinate, as (kind, index, key)."""
    out = []
    for i, md in enumerate(pol['modes']):
        out += [('mode', i, k) for k in active_keys(md)]
    out += [('guard', j, 'thr') for j in range(len(pol['guards']))]
    return out


def jitter(pol, rng, scale, keys_only=False, ncoord=1):
    """Move a FEW coordinates, not all of them.

    Perturbing 10-15 dimensions at once makes acceptance rare and the climb stall -- that is
    why the first version scored below random search."""
    p = copy.deepcopy(pol)
    coords = coord_list(p)
    for kind, i, k in rng.sample(coords, min(ncoord, len(coords))):
        if kind == 'mode':
            md = p['modes'][i]
            lo, hi = bounds_for(md, k)
            md[k] = float(np.clip(md[k] + rng.gauss(0, scale * (hi - lo)), lo, hi))
        else:
            g = p['guards'][i]
            lo, hi = SENSOR_RANGE[g['sensor']]
            g['thr'] = float(np.clip(g['thr'] + rng.gauss(0, scale * (hi - lo)), lo, hi))
    if not keys_only and rng.random() < 0.2:
        p['order'] = rng.choice(ORDERS)
    if not keys_only and rng.random() < 0.2:
        md = p['modes'][rng.randrange(p['M'])]
        md['trunc'] = rng.choice(TRUNCS)
        lo, hi = TP_BOUNDS[md['trunc']]
        md['tp'] = rng.uniform(lo, hi)
    if not keys_only and p['guards'] and rng.random() < 0.2:
        g = p['guards'][rng.randrange(len(p['guards']))]
        g['sensor'] = rng.choice(SENSORS)
        lo, hi = SENSOR_RANGE[g['sensor']]
        g['thr'] = rng.uniform(lo, hi)
        g['cmp'] = rng.choice(['ge', 'lt'])
    return p


def loss(p, pb):
    return -math.log(max((1 - EPS) * p + EPS * pb, 1e-12))


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    dt = torch.float32 if DEV == 'cpu' else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=dt).to(DEV).eval()
    rng = random.Random(SEED)
    print('tasks=%d calibration=%dx%d queries=%d candidates=%d hill=%d gate=%d'
          % (NTASK, NCAL, CALLEN, NQ, KCAND, HILL, GATE), flush=True)

    tot = dict(marg=0.0, orc=0.0, rand=0.0, hill=0.0, gate=0.0, n=0)
    for t in range(NTASK):
        true, cal, qry = build_task(model, tok, rng)
        cands = [sample_policy(rng) for _ in range(KCAND)]
        cols = [predict(c, qry) for c in cands]
        ll = np.array([loglik(c, cal) for c in cands])
        pprior = [float(np.mean(col)) for col in zip(*cols)]
        w = np.exp(ll - ll.max()); w /= w.sum()
        pmix = [float(np.dot(w, col)) for col in zip(*cols)]

        cur, cur_ll = copy.deepcopy(cands[int(np.argmax(ll))]), float(ll.max())
        for _ in range(HILL):
            prop = jitter(cur, rng, 0.12, ncoord=2)
            pl = loglik(prop, cal)
            if pl > cur_ll:
                cur, cur_ll = prop, pl

        gate = jitter(true, rng, 0.60, keys_only=True, ncoord=99)   # randomise the numbers
        g_ll = loglik(gate, cal)
        for _ in range(GATE):
            prop = jitter(gate, rng, 0.15, keys_only=True, ncoord=1)  # then coordinate-wise
            pl = loglik(prop, cal)
            if pl > g_ll:
                gate, g_ll = prop, pl

        sc = [q for q in qry if q[5]]
        for name, preds in [('rand', pmix), ('hill', predict(cur, qry)),
                            ('gate', predict(gate, qry))]:
            tot[name] += sum(loss(p, q[4]) for p, q in zip(preds, sc))
        tot['marg'] += sum(loss(p, q[4]) for p, q in zip(pprior, sc))
        tot['orc'] += sum(loss(q[3], q[4]) for q in sc)
        tot['n'] += len(sc)
        print('  task %d/%d M=%d order=%s truncs=%s' %
              (t + 1, NTASK, true['M'], true['order'],
               '/'.join(m['trunc'] for m in true['modes'])), flush=True)

    m, o, den = tot['marg'], tot['orc'], tot['marg'] - tot['orc']
    print('\n' + '=' * 68)
    print('GRAMMAR V3 -- mode machine over real mechanisms (%d positions)' % tot['n'])
    print('=' * 68)
    print('  marginal %.4f | oracle %.4f | headroom %.4f nats/pos'
          % (m / tot['n'], o / tot['n'], den / tot['n']))
    print('\n  %-38s %10s' % ('method', 'headroom captured'))
    for k, lab in [('rand', 'random search + posterior'),
                   ('hill', 'hill-climb (%d proposals)' % HILL),
                   ('gate', 'GATE: structure known')]:
        print('  %-38s %10.3f' % (lab, (m - tot[k]) / den if den else 0.0))
    print('\n  want: gate >= 0.85, random and hill far below it.')
    print('  gate low  -> evidence-limited, raise NCAL. gate high and the others low')
    print('  -> search-limited, which is the design working.')


if __name__ == '__main__':
    main()

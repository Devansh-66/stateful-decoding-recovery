"""Backfill the scored position into a simulation generated before simlab stored it.

    python patch_scored.py

The first version of simlab stored the base distribution for the 24 prefix steps but not for the
scored position itself -- the extra forward pass. Truncation samplers need the whole distribution
to decide what survives, so without it the induced distribution at exactly the position being
predicted cannot be computed offline.

Regenerating costs ~4 GPU hours. This costs one short forward pass per item (prompt + 24 tokens),
so a few minutes, because the prefix tokens are already recorded and the model is deterministic
given them.

Idempotent: tasks that already carry a 'scored' record are skipped.
"""
import os, json, math, time, shutil
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import probe as M

SIM = os.environ.get('SIM', './sim')
BATCH = int(os.environ.get('BATCH', 16))
T0 = time.time()
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def log(m):
    print('[%6.1fs] %s' % (time.time() - T0, m), flush=True)


@torch.no_grad()
def main():
    env = json.load(open(os.path.join(SIM, 'env.json')))
    TOPK, TGRID = env['topk'], env['tgrid']
    tg = torch.tensor(TGRID, device=DEV, dtype=torch.float32)
    prompts = {}
    for f in (os.environ.get('PC', './raw/prompts_calibration.csv'),
              os.environ.get('PQ', './raw/prompts_query.csv')):
        for pid, txt in M.load_prompts(f):
            prompts[pid] = txt

    tok = AutoTokenizer.from_pretrained(env['model'])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        env['model'], dtype=torch.float16 if DEV == 'cuda' else torch.float32).to(DEV).eval()
    log('model loaded; TOPK=%d, %d grid points' % (TOPK, len(TGRID)))

    src = os.path.join(SIM, 'tasks.jsonl')
    rows = [json.loads(l) for l in open(src, encoding='utf-8')]
    log('%d tasks on disk' % len(rows))
    todo = [r for r in rows if r['items'] and 'scored' not in r['items'][0]]
    log('%d need backfilling' % len(todo))

    for n, r in enumerate(todo):
        items = r['items']
        seqs = [tok(prompts[it['prompt_id']]).input_ids + list(it['prefix']) for it in items]
        npz = dict(np.load(os.path.join(SIM, '%s.npz' % r['task_id'])))
        T = npz['itm_lg'].shape[1]
        if T == len(items[0]['steps']):              # no slot for the scored step yet
            npz['itm_ids'] = np.concatenate(
                [npz['itm_ids'], np.zeros((len(items), 1, TOPK), np.int32)], 1)
            npz['itm_lg'] = np.concatenate(
                [npz['itm_lg'], np.zeros((len(items), 1, TOPK), np.float16)], 1)
            npz['itm_tail'] = np.concatenate(
                [npz['itm_tail'], np.zeros((len(items), 1), np.float32)], 1)
        last = npz['itm_lg'].shape[1] - 1

        for b0 in range(0, len(seqs), BATCH):
            chunk = seqs[b0:b0 + BATCH]
            w = max(len(x) for x in chunk)
            ids = torch.full((len(chunk), w), tok.pad_token_id, dtype=torch.long)
            att = torch.zeros((len(chunk), w), dtype=torch.long)
            for j, x in enumerate(chunk):            # left padding, matching the generator
                ids[j, w - len(x):] = torch.tensor(x)
                att[j, w - len(x):] = 1
            ids, att = ids.to(DEV), att.to(DEV)
            posi = (att.cumsum(-1) - 1).clamp_min(0)
            z = model(input_ids=ids, attention_mask=att,
                      position_ids=posi).logits[:, -1, :].float()
            for j in range(len(chunk)):
                zf = z[j]
                pr = torch.softmax(zf, -1)
                lse = torch.logsumexp(zf, -1)
                tv, ti = torch.topk(zf, TOPK)
                npz['itm_ids'][b0 + j, last] = ti.cpu().numpy()
                npz['itm_lg'][b0 + j, last] = tv.cpu().numpy().astype(np.float16)
                npz['itm_tail'][b0 + j, last] = float(
                    torch.log((1.0 - torch.exp(tv - lse).sum()).clamp_min(1e-30)))
                zg = zf[None, :] / tg[:, None]
                pg = torch.softmax(zg, -1)
                items[b0 + j]['scored'] = dict(
                    H=round(float(-(pr * torch.log(pr.clamp_min(1e-30))).sum()), 5),
                    zstd=round(float(torch.log(pr.clamp_min(1e-30)).std()), 5),
                    lse_grid=[round(float(x), 5) for x in torch.logsumexp(zg, -1)],
                    ent_grid=[round(float(x), 5) for x in
                              -(pg * torch.log(pg.clamp_min(1e-30))).sum(-1)])
        np.savez_compressed(os.path.join(SIM, '%s.npz' % r['task_id']), **npz)
        if (n + 1) % 10 == 0 or n == 0:
            log('  %d/%d tasks backfilled' % (n + 1, len(todo)))

    shutil.copy(src, src + '.bak')
    with open(src, 'w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    log('rewrote %s (backup at %s.bak)' % (src, src))

    # audit: the candidate base log-probs were recorded at generation time, so they are an
    # independent check that this re-run reproduced the same forward pass
    errs = []
    for r in rows[:20]:
        if not r['items'] or 'scored' not in r['items'][0]:
            continue
        npz = np.load(os.path.join(SIM, '%s.npz' % r['task_id']))
        last = npz['itm_lg'].shape[1] - 1
        for n, it in enumerate(r['items']):
            lg = npz['itm_lg'][n][last].astype(np.float64)
            ids = npz['itm_ids'][n][last]
            lse = np.log(np.exp(lg - lg.max()).sum()) + lg.max()
            kept = 1.0 - np.exp(min(npz['itm_tail'][n][last], 0.0))
            for c, want in zip(it['candidates'], it['cand_lp_base']):
                w = np.where(ids == c)[0]
                if w.size:
                    errs.append(abs((lg[w[0]] - lse + np.log(max(kept, 1e-12))) - want))
    e = np.array(errs) if errs else np.array([0.0])
    log('')
    log('AUDIT vs the cand_lp_base recorded during generation:')
    log('  median %.2e | 99th %.2e | max %.2e nats over %d candidates'
        % (float(np.median(e)), float(np.percentile(e, 99)), float(e.max()), e.size))
    log('  under ~1e-2 confirms the re-run reproduced the original forward pass')


if __name__ == '__main__':
    main()

"""The probe: how a scored item is built, and the prompt loader.

Extracted so the research code carries no challenge-specific machinery -- no splitting, no
grading, no answer files.

MATCHED-STRATUM CONSTRUCTION. The candidate set is built FIRST, from the base model alone and
without reference to the policy: a window of adjacent base-model ranks, from which K distinct
tokens are drawn uniformly. Because the window is narrow every candidate carries almost the same
base probability, so base rank cannot identify the label. Only then is the label drawn, from the
policy restricted to those candidates.

The alternative -- sampling decoys from a flattened prior -- fails structurally: decoys drawn
from a flatter distribution are rarer tokens, so the emitted one is simply whichever candidate
the base model likes most. Measured, a baseline using the weights but nothing about the policy
scored 0.981, with the emitted token being the base argmax 81.3% of the time against 12.5%
chance. Every flattening setting traded that shortcut against duplicate candidates and vanishing
signal.
"""
import os, csv, math
import numpy as np
import torch

import grammar_v3 as G

KCAND = int(os.environ.get('KCAND', 8))          # candidates per scored item
STRATUM_TOP = int(os.environ.get('STRATUM_TOP', 512))
STRATUM_W = int(os.environ.get('STRATUM_W', 384))
# items whose reference distribution is near-uniform carry almost no information. The test is on
# the candidate SET -- the largest reference probability across candidates -- never on which
# label was drawn: filtering on the drawn label would be outcome-dependent and would bias the
# surviving labels toward likely candidates, which a solver could exploit by sharpening.
MIN_MAXP = float(os.environ.get('MIN_MAXP', 0.18))


def load_prompts(path):
    return [(r['prompt_id'], r['text']) for r in csv.DictReader(open(path, encoding='utf-8'))]


def scored_item(z_b, pol, st_b, step, rng):
    """Build one scored item at the position reached by a prefix.

    MATCHED STRATUM construction. The candidate set is built FIRST, from the base model alone
    and without reference to the policy: a window of adjacent base-model ranks, from which K
    distinct tokens are drawn uniformly. Because the window is narrow, every candidate has
    almost the same base probability. Only then is the label drawn, from the policy restricted
    to those candidates.

    This replaces sampling decoys from a flattened prior mixture, which failed structurally.
    Decoys drawn from a flatter distribution than the policy are rarer tokens, so the emitted
    one was simply whichever candidate the base model liked most -- a baseline using the
    weights but nothing about the policy scored 0.981, with the emitted token being the base
    argmax 81.3% of the time against 12.5% chance. Every setting of the flattening parameter
    traded that shortcut against duplicate candidates and vanishing signal.

    Here there is no decoy distribution to estimate, no duplicates by construction, and base
    rank cannot identify the label because the candidates are matched on base probability. What
    is left to infer is how the policy redistributes mass within the set.
    """
    q, p_raw, _ = G.policy_dist(z_b, pol, st_b, step)

    top = torch.topk(p_raw.float(), STRATUM_TOP)
    # a log-uniform anchor: narrow windows near the head, wider spacing down the tail, which is
    # where truncation boundaries actually fall
    hi = STRATUM_TOP - STRATUM_W
    a = int(math.exp(rng.uniform(0.0, math.log(hi))))
    a = max(0, min(a, hi - 1))
    window = top.indices[a:a + STRATUM_W]

    pick = torch.randperm(STRATUM_W, generator=torch.Generator().manual_seed(rng.randrange(1 << 30)))
    cands = window[pick[:KCAND]].tolist()
    if len(set(cands)) < KCAND:
        return None

    ct = torch.tensor(cands, device=z_b.device)
    w = q[ct].double()
    if float(w.sum()) <= 0:
        return None                      # the policy truncates the whole window away
    p = (w / w.sum()).cpu().numpy()

    idx = int(np.searchsorted(np.cumsum(p), rng.random()))
    idx = min(idx, KCAND - 1)

    # the base model restricted to the same candidates. This is the zero point of the metric:
    # nearly every mechanism in the grammar is a MONOTONE transform of the base distribution --
    # temperature preserves order, truncation removes a tail without reordering what is left --
    # so the candidate the policy most likely emitted is usually the one the base model already
    # ranked highest. Measured, base ranking captured 0.72-0.80 of the uniform-normalised score
    # at every stratum width tried. Scoring against uniform would hand that away for free;
    # scoring against the base model asks the question that matters, which is what knowing the
    # POLICY adds beyond knowing the model.
    b = p_raw[ct].double()
    b = (b / b.sum().clamp_min(1e-30)).cpu().numpy()
    base_spread = float((top.values[a] / top.values[a + STRATUM_W - 1]).log())
    return cands, idx, float(p[idx]), float(b[idx]), float(p.max()), base_spread

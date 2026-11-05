# Recovering stateful decoding policies from passive text

Can you identify a hidden, *adaptive* decoding policy from text it produced — with no ability to
re-query the system?

**Status: in progress. The headline comparison is measured; the mechanism is not yet established.**
Numbers below are what I have actually run, including the ones that came out badly.

---

## The problem

An API serves text from a language model. In front of the model sits a decoding policy that
decides how each token is drawn — how random to be, which low-probability tokens to cut, whether
to penalise repetition. Providers do not disclose it.

[Ippolito et al. (INLG 2023)](https://aclanthology.org/2023.inlg-main.28/) recover **fixed**
decoding hyperparameters from black-box access: issue the same prompt many times, read the
empirical distribution over the next token, and classify top-k vs nucleus. Their threat model is
explicit — *"without loss of generality, we assume that the response is exactly one token long"* —
and so is their limitation:

> there is no guarantee that a system would use a single decoding strategy

That limitation is the problem taken up here. Real policies **react**: becoming conservative where
the model is uncertain, backing off on repetition, switching behaviour mid-generation. Repeated
sampling cannot reach them, because the internal state differs on every rollout.

The observation setting is the inverse of theirs:

|                | Ippolito et al.                | Here                          |
|----------------|--------------------------------|-------------------------------|
| access         | active probing, chosen prompts | passive, fixed text           |
| samples        | many draws at **one** position | one draw at **many** positions|
| policy         | fixed hyperparameters          | stateful, changes mid-sequence|
| output         | classify / estimate k or p     | predict a distribution        |

## What is actually being studied

This is not an NLP task. The text content is irrelevant — replace the tokens with integers and
nothing changes. It is **latent-state system identification**: a hidden finite state machine
modulates a known high-dimensional emission model, and you observe one sample per step.

Policies are built from samplers that ship in real inference stacks — `min_p`, locally typical,
`top_nsigma`, Mirostat-2, XTC, repetition penalty (llama and HF orderings), dynamic temperature —
composed under a state machine whose transitions are gated on the model's own output at each step:
predictive entropy, top-2 logit gap, recent-repetition count, position. Plus dwell locks (stay in
a mode for N tokens) and expiry timeouts.

**Task.** Given the base model and text a hidden policy produced, predict a distribution over 8
candidate tokens at a held-out position, one of which the policy emitted.

**Metric.** `(L_base − L_sub) / (L_base − L_ref)` — the fraction of achievable headroom captured.
`0` is the base model restricted to the candidates; `1` is the hidden policy itself. Scores can go
below 0 and, in principle, above 1 (the policy is uncertain about its own output, so exceeding it
requires knowing the realised draw). Ground truth is exact by construction.

## Results

Fraction of headroom captured, on the released evaluation set:

| method | score |
|---|---|
| per-task temperature fit, calibration traces only | **0.085** |
| per-task search **with the true generative grammar** (best-of-64 + 300 hill-climb steps) | **0.127** |
| generic cross-task learning — feature engineering + small nets over 398 training tasks | **0.28 – 0.32** |
| true policy (oracle) | 1.000 |

Score standard error ≈ 0.027.

**The finding: amortised learning across policies beats principled per-instance inference that was
handed the true generative family, by roughly 2.5×.** The information needed to identify a policy
appears to live more in what *other* policies look like than in that policy's own traces.

Supporting observations:

- The hill-climb improved calibration log-likelihood on **20 of 20 tasks while making prediction
  worse** — overfitting ~20 parameters to 256 observed tokens.
- The search only helps when blended ~30% with the base model, i.e. the traces support a
  *perturbation* of the base model rather than a policy.
- Without a base-model probability floor the same search scores **−0.131**, worse than submitting
  base-model probabilities. A single wrong truncation costs `−log(1e-7) ≈ 16` nats.

## What is NOT established

Stated plainly, because the distinction matters:

- **Why** amortisation wins. The hypothesis is that difficulty localises to inferring the latent
  mode sequence — once modes are known, parameters are easy. That currently rests on a single
  experiment with one policy and one prompt, which is an anecdote.
- **Whether the stateful component is recoverable at all.** Nothing has exceeded 0.13 per-task.
- **Whether policies of this shape are deployed.** The components are real; the composition into a
  sensor-gated state machine is not a documented deployment pattern. This is the weakest point of
  the motivation and I would not defend it as written.

A mode-oracle control (reveal the true modes on calibration traces, refit, measure) was
pre-registered with a kill condition before any number was computed. It returned **−1.911** — worse
than ignoring the policy entirely — and degraded monotonically with parameter count
(1 mode: +0.281, 2 modes: −0.4, 3 modes: −6.5). That pattern is a starved optimiser, not an
unidentifiable problem: 120 random single-coordinate proposals over 24 dimensions barely leaves the
starting point. **It is not reported as a finding.** `planted.py` is the test that separates the
two readings.

## A methodological result

**An adaptive sampler's induced distribution cannot be reconstructed from stored top-K logits.**

Mirostat-2 thresholds on `p ≥ 2^−μ`, which at μ ≈ 18 is `3.8e-6` — far below rank 512. Its
keep-set therefore extends into the tail: I measured **23% of emitted tokens falling outside a
stored top-512**, and for `llama`-ordered policies the post-truncation renormalisation runs over
the full surviving set, so no tail-mass correction recovers it. Every other sampler reconstructs to
≤ 1.9e-3 nats.

Anyone building an auditing or attribution benchmark on logged logits will hit this. Store full
vocabulary, or restrict to samplers whose keep-set is confined to the head, or measure the
reconstruction error per instance and exclude what fails (the approach taken here).

## Verification

The offline reimplementation is checked against the generator before any result is trusted:

```
median |recomputed − stored| log-prob  =  3.73e-06 nats
```

That audit found five bugs that would each have silently corrupted every downstream number:

1. **Renormalising the top-K to sum to 1** changed absolute probability values, breaking every
   sampler that thresholds on probability. 0.30 total-variation error.
2. **`top_nsigma` uses the std of log-probs over the full vocabulary** — 3.00 full vs 0.76
   truncated. Not recoverable from top-K; now stored per step.
3. **`hf` order applies temperature before truncation**, so the sampler sees `softmax(z/T)` over
   the full vocabulary. That partition is stored on a temperature grid and interpolated
   quadratically in log-T (7× more accurate than linear on the same points).
4. **Typical sampling needs the *tempered* entropy** and top-nσ needs `zstd/T` after tempering.
5. **The sampler state must advance on every step**, including when the emitted token falls
   outside the stored support. Mirostat's μ feeds forward, so a skipped advance desynchronises it
   for the remainder of the trace — worth 0.19–0.33 nats of drift.

An earlier version of the dataset was found to be **unreproducible**: its generator sampled with
`torch.multinomial` and never called `torch.manual_seed`, so the hidden policies existed only in
that process's memory. Every task in the current simulator seeds from its own index, and the
consumed uniform draws are stored so any trace can be verified without regenerating it.

## Layout

```
grammar_v3.py     the policy family: samplers, sensors, the state machine
simlab.py         simulator. Records the base model's top-512 distribution at every step, plus
                  the full-vocab statistics truncation destroys, ground-truth modes, transition
                  provenance with signed threshold margins, per-mode counterfactual log-probs,
                  inter-mode total variation, and the consumed uniform draws
policy_cpu.py     the sampler transform in numpy over the stored support, so every fit, baseline
                  and ablation runs on CPU with no model and no GPU
gate_c1a.py       the mode-oracle control, with its pre-registered kill condition
planted.py        planted-recovery test: can the fitter recover a policy it was handed the answer
                  to? Separates "the optimiser failed" from "the data does not identify it"
patch_scored.py   backfills the scored position into a run generated before it was stored
sim.ipynb         end-to-end: generate, audit, gate, verdict
```

**Design principle.** The base model's distribution is stored at every step. Every sampler is a
deterministic function of it, so afterwards every fit, baseline, ablation and diagnostic is a CPU
job. It is the only quantity that cannot be reconstructed without repeating the GPU run.

Conditions are balanced across tasks: 1/2/3 modes; dwell locks on/off (off makes the mode a
function of the current sensor vector, so it is pointwise identifiable — the clean decomposition of
difficulty); and thresholds placed at designed quantiles of each sensor's *measured* distribution,
well-bracketed (40–60th percentile) or poorly-bracketed (5th/95th). The margin condition is a
two-pass design and cannot be added afterwards.

Calibration budgets of 256 / 1024 / 2048 / 4096 tokens are subsets of one generated set, so the
sample-complexity curve costs one run rather than four.

## Reproducing

`sim.ipynb` runs end to end: generate (~90 min on one GPU), audit the CPU path, run the gate,
print the verdict. Everything after generation is CPU.

Base model is Qwen2.5-0.5B (Apache-2.0).

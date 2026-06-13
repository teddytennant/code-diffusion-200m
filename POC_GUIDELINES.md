# PoC Guidelines (partial — intentionally incomplete)

This is a loose direction, not a finished spec. It sets the goal, the hardware, and
the time budget. The *how* is left open on purpose — figure out the cleanest path to
the goal within these constraints.

## The goal

Prove a **concept** about diffusion language models that is genuinely impressive given
the hardware — not a leaderboard number. Something a frontier-lab reader would find
*legible and honest*: a clean demonstration of what diffusion buys you (parallel /
bidirectional generation, infilling, iterative self-correction) that an autoregressive
baseline of the same size and compute does not.

"Impressive" here means **rigorous and convincing at small scale**, not big. A
controlled, reproducible result with honest limitations beats a flashy weak number.

## The compute (hard constraint)

- **Single RTX 5060 Ti, 16 GB VRAM.** Blackwell (sm_120). That's the whole budget — no
  cluster, no multi-GPU.
- Practical implications already established:
  - ~16 GB fits roughly a 100–300M model; the lever is **tokens and domain**, not params.
  - torch must be the **cu128** build; `bitsandbytes >= 0.45`; **no flash-attn**
    (use SDPA's flash kernel — set `flash_attention: false`).
  - The logits tensor `[B, seq, vocab]` is the memory hog → keep `per_device_batch`
    small (2 at 4k ctx, more at 2k) and lean on grad accumulation.
  - See `configs/5060ti.yaml` and `requirements-5060ti.txt` for the working setup.

## Time budget

- **Time is effectively infinite** — do not cut corners to save GPU-hours, and prefer
  doing the rigorous version (multiple seeds, matched-compute baselines) over the quick
  version.
- That said, the *preferred* total is **~2–3 weeks of overnight runs** end to end.
  Design the experiment so it fits there: favor small/fast models that train in hours so
  several runs fit one night, rather than one model that hogs the card for a week.

## Loose direction (not binding)

- Match capability to budget: pick a domain narrow enough that the model is actually
  *competent*, so the result is legible.
- The honest value prop of diffusion is **speed + infilling + self-correction** — design
  the demonstration around that, not around raw generation quality.
- Compare against a matched autoregressive baseline (same params, same data, same
  compute). The comparison *is* the result.
- Be honest about where diffusion loses. The limitations section is a credibility signal,
  not an afterthought.

## Out of scope (for now)

- SOTA / "best-in-size" claims.
- Multi-GPU, large models, large token budgets.
- A polished product. This is a research artifact.

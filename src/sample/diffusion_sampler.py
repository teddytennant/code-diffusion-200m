from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import torch
import torch.nn.functional as F

from src.eval.sampler_interface import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX


class _TokenizerLike(Protocol):
    mask_id: int
    pad_id: int
    fim_prefix_id: int
    fim_middle_id: int
    fim_suffix_id: int
    vocab_size: int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids) -> str: ...


@dataclass
class _SampleConfig:
    diffusion_steps: int
    remask_frac: float
    ar_refine: bool
    ar_refine_threshold: float
    confidence_remask: bool
    top_p: float
    top_k: int


def _filter_logits(logits: torch.Tensor, top_p: float, top_k: int) -> torch.Tensor:
    """Apply top-p / top-k filtering. ``logits`` shape: ``(N, V)``."""
    if top_k and top_k > 0:
        k = min(top_k, logits.size(-1))
        kth_vals = torch.topk(logits, k, dim=-1).values[..., -1, None]
        logits = torch.where(logits < kth_vals, torch.full_like(logits, float("-inf")), logits)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(sorted_probs, dim=-1)
        remove_sorted = cumprobs > top_p
        remove_sorted[..., 1:] = remove_sorted[..., :-1].clone()
        remove_sorted[..., 0] = False
        remove = torch.zeros_like(remove_sorted)
        remove.scatter_(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    return logits


def _sample_with_confidence(
    logits: torch.Tensor,
    temperature: float,
    top_p: float,
    top_k: int,
    forbid_ids: tuple[int, ...] = (),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample tokens and return (tokens, confidence) where confidence = max softmax prob.

    ``logits`` is ``(N, V)``. Returns ``(N,)`` token ids and ``(N,)`` confidences.
    Any id in ``forbid_ids`` is banned from sampling and from the confidence calc.
    """
    if logits.numel() == 0:
        return (
            torch.empty(0, dtype=torch.long, device=logits.device),
            torch.empty(0, dtype=torch.float32, device=logits.device),
        )
    work = logits.float()
    if forbid_ids:
        for fid in forbid_ids:
            work[..., fid] = float("-inf")

    full_probs = torch.softmax(work, dim=-1)
    confidence = full_probs.max(dim=-1).values

    if temperature <= 0:
        tokens = work.argmax(dim=-1)
        return tokens, confidence

    scaled = work / temperature
    filtered = _filter_logits(scaled, top_p=top_p, top_k=top_k)
    sample_probs = torch.softmax(filtered, dim=-1)
    sample_probs = torch.nan_to_num(sample_probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = sample_probs.sum(dim=-1, keepdim=True)
    bad = row_sums.squeeze(-1) <= 0
    if bad.any():
        sample_probs[bad] = full_probs[bad]
    tokens = torch.multinomial(sample_probs, num_samples=1).squeeze(-1)
    return tokens, confidence


class DiffusionSampler:
    def __init__(
        self,
        model,
        tokenizer: _TokenizerLike,
        device: str | torch.device = "cuda",
        default_diffusion_steps: int = 16,
        default_remask_frac: float = 0.5,
        default_ar_threshold: float = 0.7,
        ar_refine: bool = True,
        confidence_remask: bool = True,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.default_diffusion_steps = default_diffusion_steps
        self.default_remask_frac = default_remask_frac
        self.default_ar_threshold = default_ar_threshold
        self.ar_refine = ar_refine
        self.confidence_remask = confidence_remask

    @staticmethod
    def _remask_k(n_total: int, remask_frac: float, step: int, total_steps: int) -> int:
        if total_steps <= 1:
            return 0
        if step >= total_steps - 1:
            return 0
        decay = 1.0 - step / (total_steps - 1)
        return max(0, round(n_total * remask_frac * decay))

    def _resolve_config(self, kwargs: dict[str, Any]) -> _SampleConfig:
        return _SampleConfig(
            diffusion_steps=int(kwargs.get("diffusion_steps", self.default_diffusion_steps)),
            remask_frac=float(kwargs.get("remask_frac", self.default_remask_frac)),
            ar_refine=bool(kwargs.get("ar_refine", self.ar_refine)),
            ar_refine_threshold=float(kwargs.get("ar_refine_threshold", self.default_ar_threshold)),
            confidence_remask=bool(kwargs.get("confidence_remask", self.confidence_remask)),
            top_p=float(kwargs.get("top_p", 1.0)),
            top_k=int(kwargs.get("top_k", 0)),
        )

    def _autocast(self):
        if self.device.type == "cuda":
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return nullcontext()

    def sample(
        self,
        prompt: str,
        max_new_tokens: int,
        mode: Literal["completion", "fim"] = "completion",
        n_samples: int = 1,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> list[str]:
        if max_new_tokens <= 0:
            return [""] * n_samples

        seed = kwargs.get("seed", None)
        if seed is not None:
            torch.manual_seed(int(seed))

        cfg = self._resolve_config(kwargs)
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad(), self._autocast():
                if mode == "completion":
                    return self._sample_completion(prompt, max_new_tokens, n_samples, temperature, cfg)
                if mode == "fim":
                    return self._sample_fim(prompt, max_new_tokens, n_samples, temperature, cfg)
                raise ValueError(f"unknown mode: {mode!r}")
        finally:
            if was_training:
                self.model.train()

    def _sample_completion(
        self,
        prompt: str,
        max_new_tokens: int,
        n_samples: int,
        temperature: float,
        cfg: _SampleConfig,
    ) -> list[str]:
        prompt_ids = self.tokenizer.encode(prompt)
        max_seq_len = self.model.config.max_seq_len
        if len(prompt_ids) + max_new_tokens > max_seq_len:
            keep = max_seq_len - max_new_tokens
            if keep <= 0:
                raise ValueError(
                    f"max_new_tokens={max_new_tokens} >= max_seq_len={max_seq_len}; nothing left for prompt"
                )
            prompt_ids = prompt_ids[-keep:]

        prompt_len = len(prompt_ids)
        seq_len = prompt_len + max_new_tokens

        buf = torch.full(
            (n_samples, seq_len),
            self.tokenizer.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        if prompt_len > 0:
            prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
            buf[:, :prompt_len] = prompt_t

        gen_slice = slice(prompt_len, seq_len)
        confidences = self._run_diffusion(buf, gen_slice, temperature, cfg)

        if cfg.ar_refine:
            self._ar_refine(buf, gen_slice, confidences, temperature, cfg)

        outputs = []
        for row in buf:
            gen_ids = row[gen_slice].tolist()
            outputs.append(self.tokenizer.decode(gen_ids))
        return outputs

    def _sample_fim(
        self,
        prompt: str,
        max_new_tokens: int,
        n_samples: int,
        temperature: float,
        cfg: _SampleConfig,
    ) -> list[str]:
        prompt_ids = self.tokenizer.encode(prompt)
        ids_set = set(prompt_ids)
        required = {
            self.tokenizer.fim_prefix_id,
            self.tokenizer.fim_suffix_id,
            self.tokenizer.fim_middle_id,
        }
        if not required.issubset(ids_set):
            raise ValueError(
                "FIM mode requires prompt to contain <fim_prefix>, <fim_suffix>, and <fim_middle> markers"
            )

        max_seq_len = self.model.config.max_seq_len
        if len(prompt_ids) + max_new_tokens > max_seq_len:
            keep = max_seq_len - max_new_tokens
            if keep <= 0:
                raise ValueError(
                    f"max_new_tokens={max_new_tokens} >= max_seq_len={max_seq_len}; nothing left for prompt"
                )
            prompt_ids = prompt_ids[-keep:]
            if not required.issubset(set(prompt_ids)):
                raise ValueError(
                    "FIM markers were truncated by max_seq_len; shrink prompt or max_new_tokens"
                )

        prompt_len = len(prompt_ids)
        seq_len = prompt_len + max_new_tokens

        buf = torch.full(
            (n_samples, seq_len),
            self.tokenizer.mask_id,
            dtype=torch.long,
            device=self.device,
        )
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        buf[:, :prompt_len] = prompt_t

        gen_slice = slice(prompt_len, seq_len)
        confidences = self._run_diffusion(buf, gen_slice, temperature, cfg)

        if cfg.ar_refine:
            self._ar_refine(buf, gen_slice, confidences, temperature, cfg)

        stop_ids = {
            self.tokenizer.fim_prefix_id,
            self.tokenizer.fim_middle_id,
            self.tokenizer.fim_suffix_id,
            self.tokenizer.pad_id,
        }
        outputs = []
        for row in buf:
            gen_ids = row[gen_slice].tolist()
            cut = len(gen_ids)
            for i, tid in enumerate(gen_ids):
                if tid in stop_ids:
                    cut = i
                    break
            outputs.append(self.tokenizer.decode(gen_ids[:cut]))
        return outputs

    def _run_diffusion(
        self,
        buf: torch.Tensor,
        gen_slice: slice,
        temperature: float,
        cfg: _SampleConfig,
    ) -> torch.Tensor:
        bsz, _ = buf.shape
        gen_start = gen_slice.start
        gen_end = gen_slice.stop
        gen_len = gen_end - gen_start
        mask_id = self.tokenizer.mask_id

        confidences = torch.zeros((bsz, gen_len), dtype=torch.float32, device=self.device)

        steps = max(1, cfg.diffusion_steps)
        for s in range(steps):
            cur_mask = buf[:, gen_slice] == mask_id
            if not cur_mask.any():
                break

            logits = self.model(buf, causal=False)
            gen_logits = logits[:, gen_slice, :]

            flat_logits = gen_logits[cur_mask]
            tokens, conf = _sample_with_confidence(
                flat_logits,
                temperature=temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                forbid_ids=(mask_id,),
            )

            gen_buf = buf[:, gen_slice].clone()
            gen_buf[cur_mask] = tokens
            confidences = confidences.clone()
            confidences[cur_mask] = conf

            filled_this_step = cur_mask.clone()

            is_last = s == steps - 1
            if cfg.confidence_remask and not is_last:
                k_total = self._remask_k(gen_len, cfg.remask_frac, s, steps)
                if k_total > 0:
                    for b in range(bsz):
                        elig = filled_this_step[b].nonzero(as_tuple=False).squeeze(-1)
                        if elig.numel() == 0:
                            continue
                        k_b = min(k_total, elig.numel())
                        if k_b <= 0:
                            continue
                        confs_b = confidences[b, elig]
                        worst = torch.topk(confs_b, k_b, largest=False).indices
                        positions = elig[worst]
                        gen_buf[b, positions] = mask_id

            buf[:, gen_slice] = gen_buf

        cur_mask = buf[:, gen_slice] == mask_id
        if cur_mask.any():
            logits = self.model(buf, causal=False)
            gen_logits = logits[:, gen_slice, :]
            flat_logits = gen_logits[cur_mask].float()
            flat_logits[..., mask_id] = float("-inf")
            full_probs = torch.softmax(flat_logits, dim=-1)
            tokens = full_probs.argmax(dim=-1)
            conf = full_probs.max(dim=-1).values
            gen_buf = buf[:, gen_slice].clone()
            gen_buf[cur_mask] = tokens
            confidences = confidences.clone()
            confidences[cur_mask] = conf
            buf[:, gen_slice] = gen_buf

        return confidences

    def _ar_refine(
        self,
        buf: torch.Tensor,
        gen_slice: slice,
        confidences: torch.Tensor,
        temperature: float,
        cfg: _SampleConfig,
    ) -> None:
        gen_start = gen_slice.start
        gen_end = gen_slice.stop
        bsz = buf.size(0)

        for _ in range(2):
            low_conf = confidences < cfg.ar_refine_threshold
            if not low_conf.any():
                return

            logits = self.model(buf, causal=True)

            gen_buf = buf[:, gen_slice].clone()
            new_conf = confidences.clone()

            for b in range(bsz):
                positions = low_conf[b].nonzero(as_tuple=False).squeeze(-1).tolist()
                if not positions:
                    continue
                for p in positions:
                    abs_pos = gen_start + p
                    if abs_pos == 0:
                        continue
                    row_logits = logits[b : b + 1, abs_pos - 1, :]
                    tokens, conf = _sample_with_confidence(
                        row_logits,
                        temperature=temperature,
                        top_p=cfg.top_p,
                        top_k=cfg.top_k,
                        forbid_ids=(self.tokenizer.mask_id,),
                    )
                    gen_buf[b, p] = tokens[0]
                    new_conf[b, p] = conf[0]

            buf[:, gen_slice] = gen_buf
            confidences[...] = new_conf

"""DataLoader factory for ``CodeDiffusionDataset``."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, IterableDataset


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "target_ids": torch.stack([b["target_ids"] for b in batch], dim=0),
        "mask_positions": torch.stack([b["mask_positions"] for b in batch], dim=0),
        "mask_ratio": torch.as_tensor(
            [float(b["mask_ratio"]) for b in batch], dtype=torch.float32
        ),
    }


def make_dataloader(
    dataset: IterableDataset,
    batch_size: int,
    num_workers: int = 4,
    **kwargs: Any,
) -> DataLoader:
    """Return a DataLoader with a stack-collate suited for diffusion training.

    Extra ``kwargs`` forward to ``DataLoader`` (e.g. ``persistent_workers``).
    """
    defaults: Dict[str, Any] = {
        "pin_memory": True,
        "drop_last": True,
    }
    defaults.update(kwargs)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=_collate,
        **defaults,
    )

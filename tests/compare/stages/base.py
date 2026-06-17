"""Base class for reference computation stages."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

import torch

from tests.compare.config import DumpConfig
from tests.compare.cos import cos_sim, element_stats, l2_error
from tests.compare.weight_loader import WeightLoader


def _get_input(inputs: dict, key: str, gpu_idx: int) -> torch.Tensor:
    """Resolve a GPU-specific input key.

    All intermediate outputs are stored with _gpu{idx} suffix (e.g. "attn.q_proj_raw_gpu1").
    For shared values (e.g. norm1) the tensor is identical across GPUs so any key works.
    For gpu_idx == 0, also checks without suffix for backward compatibility.
    """
    if gpu_idx == 0:
        return inputs.get(key, inputs[f"{key}_gpu{gpu_idx}"])
    return inputs[f"{key}_gpu{gpu_idx}"]


class Stage(ABC):
    """Base class for a single reference computation stage.

    Each stage represents one sub-operation (e.g. norm1, q_proj) and knows:
    - its own name and cosine similarity threshold
    - how to compute the reference output given previous stages' outputs
    - which engine dump file to load and compare against
    """

    name: str           # e.g. "attn.q_proj_raw"
    threshold: float    # cosine similarity threshold (0.0-1.0)

    @abstractmethod
    def compute(
        self,
        inputs: Dict[str, torch.Tensor],
        weights: WeightLoader,
        config: DumpConfig,
        layer_idx: int,
        gpu_idx: int,
    ) -> torch.Tensor:
        """Compute the reference output for this stage.

        Args:
            inputs: dict of previously computed reference outputs (stage name → tensor).
            weights: WeightLoader for the model.
            config: DumpConfig with model parameters.
            layer_idx: Target layer index.
            gpu_idx: GPU index for TP sharding.

        Returns:
            Reference output as float32 torch.Tensor.
        """
        raise NotImplementedError

    def compare(
        self,
        dump_dir: str,
        ref: torch.Tensor,
        layer_idx: int,
        gpu_idx: int,
    ) -> dict:
        """Load engine dump and compare against reference.

        Args:
            dump_dir: Path to the engine dump directory for this layer.
            ref: Reference tensor from compute().
            layer_idx: Target layer index.
            gpu_idx: GPU index for TP sharding.

        Returns:
            dict with keys: cos, l2_err, max_diff, passed
        """
        from tests.compare import io

        # Build expected filename — e.g. "attn.norm1_gpu0.raw" or "mlp.down_ar_gpu0.raw"
        # Engine always includes _gpu{idx} suffix, even for GPU 0
        raw_path = Path(dump_dir) / f"{self.name}_gpu{gpu_idx}.raw"

        if not raw_path.exists():
            return {
                "cos": 0.0,
                "l2_err": 1.0,
                "max_diff": -1.0,
                "passed": False,
                "error": f"missing_engine_dump ({self.name}_gpu{gpu_idx}.raw)",
            }

        engine_flat = io.load_raw_bf16(str(raw_path), (-1,))

        if engine_flat.numel() != ref.numel():
            return {
                "cos": 0.0,
                "l2_err": 1.0,
                "max_diff": -1.0,
                "passed": False,
                "error": f"size_mismatch engine={engine_flat.numel()} ref={ref.numel()}",
            }

        engine_t = engine_flat.reshape(ref.shape).float()

        cos = cos_sim(engine_t, ref)
        l2 = l2_error(engine_t, ref)
        stats = element_stats(engine_t, ref)
        passed = cos >= self.threshold

        return {
            "cos": cos,
            "l2_err": l2,
            "max_diff": stats["max"],
            "passed": passed,
        }

"""Pure Stage-J2 local-step learning-rate schedule."""

import math


def joint_formation_j2_lr_scale(
    optimizer_step: int,
    *,
    warmup_steps: int = 50,
    total_steps: int = 710,
    min_ratio: float = 0.1,
) -> float:
    """Return the scale for the upcoming local optimizer step."""
    if optimizer_step < 1 or optimizer_step > total_steps:
        raise ValueError("optimizer_step must be in [1, total_steps]")
    if warmup_steps <= 0 or warmup_steps >= total_steps:
        raise ValueError("warmup_steps must be in (0, total_steps)")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError("min_ratio must be in [0, 1]")
    if optimizer_step <= warmup_steps:
        return float(optimizer_step) / float(warmup_steps)
    progress = float(optimizer_step - warmup_steps) / float(total_steps - warmup_steps)
    return float(min_ratio + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress)))


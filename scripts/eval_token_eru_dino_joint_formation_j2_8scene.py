"""User-run J2 held-out evaluator wrapper.

The implementation remains the validated isolated JointFormation evaluator;
only its configuration, checkpoint root, and output root are redirected to
Stage-J2.  This module never performs training.
"""

from scripts import audit_token_eru_dino_joint_formation_24window_paired as evaluator


evaluator.JOINT_CONFIG = (
    "semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_j2_ddp8"
)
evaluator.JOINT_ROOT = evaluator.ROOT / (
    "workspace/semantic_v6_absolute_units_true_shared_token_eru1_"
    "dino_metric_joint_formation_j2_ddp8"
)
evaluator.FORMAL_OUTPUT = evaluator.ROOT / (
    "workspace/token_eru_dino_joint_formation_j2_8scene_eval"
)


if __name__ == "__main__":
    evaluator.main()


"""240-only centralized path remap for the TokenGS LSM eval.

The workspaces/configs/metadata migrated from 108 still contain absolute
paths rooted at ``/space0/mawb/tokengs``, which does not exist on 240.
This module is the single place that rewrites those runtime paths to the
240 location.  It is deliberately opt-in (imported only by the 240 eval
launcher path) and does not touch the migrated config.yaml / metadata
files, which stay byte-identical to the 108 originals.
"""

from __future__ import annotations


_OLD_ROOT = "/space0/mawb/tokengs"
_NEW_ROOT = "/space/mawb/tokengs"


def remap_path(path):
    """Rewrite one 108-era absolute path to the 240 tree (no-op otherwise)."""
    if not path:
        return path
    if isinstance(path, str) and path.startswith(_OLD_ROOT):
        return _NEW_ROOT + path[len(_OLD_ROOT):]
    return path


def remap_opt(opt):
    """Rewrite the path-bearing Options fields used by the LSM eval."""
    for field in (
        "prompt_tokengs_checkpoint",
        "prompt_clip_model_path",
        "backbone_resume",
        "resume",
        "workspace",
    ):
        value = getattr(opt, field, None)
        remapped = remap_path(value)
        if remapped is not None:
            setattr(opt, field, remapped)

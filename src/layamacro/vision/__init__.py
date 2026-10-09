"""Vision side of the graft: frozen encoder → resampler → projector.

The resampler and projector are trained from scratch. The encoder is frozen by default —
the datasets are far too small to tune all of it; `tail.SiglipTail` can train its last few
blocks on cached hidden states.
"""

from layamacro.vision.projector import Projector
from layamacro.vision.resampler import PerceiverResampler

__all__ = ["PerceiverResampler", "Projector"]

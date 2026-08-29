"""Central configuration: thresholds, model paths, camera source, compute device.

Everything tunable lives here so experiments (mucs 6/7 in the plan) only touch one
file. Import `CFG` elsewhere: `from configs import CFG`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"      # weights (git-ignored)
DATA_DIR = ROOT / "data"          # test videos/images (git-ignored)
OUTPUTS_DIR = ROOT / "outputs"    # CSV / rendered videos (git-ignored)


@dataclass
class Config:
    # ---- input source ----
    camera_index: int = 0            # webcam id
    process_long_side: int = 640     # resize target for the long edge

    # ---- preprocessing ----
    use_clahe: bool = False          # toggle for CLAHE contrast enhancement
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: tuple[int, int] = (8, 8)

    # ---- detection & tracking ----
    # SCRFD detects down to this score. Keeping the floor below
    # `track_high_threshold` is what gives ByteTrack a low-score band to run its
    # second association on; with both at the same value that branch is dead and
    # a track loses its id the moment detection confidence dips.
    conf_threshold: float = 0.3
    track_high_threshold: float = 0.6   # ByteTrack first-association band
    track_buffer_frames: int = 90       # ~9 s of memory at the ~10 fps we really run

    # ---- attention rule ----
    yaw_threshold_deg: float = 22.0
    pitch_threshold_deg: float = 17.0
    attention_smooth_frames: int = 3  # temporal smoothing window

    # ---- periodic VLM branch (optional) ----
    vlm_enabled: bool = False
    vlm_period_seconds: float = 90.0


CFG = Config()

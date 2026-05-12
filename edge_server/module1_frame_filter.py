
import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple
from pathlib import Path

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s :: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("M1:FrameFilter")


@dataclass
class M1Result:
    frame_id:       int
    skip:           bool            # True = discard this frame
    skip_reason:    str             # human-readable reason
    mean_diff:      float           # inter-frame mean absolute diff
    fg_ratio:       float           # foreground area / total area
    fg_mask:        Optional[np.ndarray] = field(default=None, repr=False)
    elapsed_ms:     float = 0.0

    def log_summary(self):
        status = "SKIP" if self.skip else "PASS"
        log.info(
            f"[Frame {self.frame_id:5d}] {status:4s} | "
            f"mean_diff={self.mean_diff:6.2f} | "
            f"fg_ratio={self.fg_ratio:.4f} ({self.fg_ratio*100:.2f}%) | "
            f"reason='{self.skip_reason}' | "
            f"elapsed={self.elapsed_ms:.2f}ms"
        )


class M1Config:
    # Inter-frame difference
    # brightness units (0-255). Below = static frame.
    DIFF_THRESHOLD: float = 8.0

    # MOG2 background model
    # frames to build background model (reduced for Pi)
    MOG2_HISTORY:   int = 100
    # sensitivity. Higher = less sensitive to noise.
    MOG2_VAR_THRESH: float = 40.0
    MOG2_SHADOWS:   bool = False  # shadow detection off — saves compute on Pi

    # Foreground gate (used again in Module 3, but computed here)
    FG_RATIO_MIN:   float = 0.005  # 0.5% — below this = nothing meaningful moving

    # Morphological cleanup of MOG2 mask
    MORPH_KERNEL_SIZE: int = 3    # kernel for noise removal


class FrameLevelFilter:

    def __init__(self, config: M1Config = M1Config()):
        self.cfg = config
        self.frame_id = 0
        self.prev_gray = None
        self.skip_count = 0
        self.pass_count = 0

        # MOG2 background subtractor — the core of Module 1
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=config.MOG2_HISTORY,
            varThreshold=config.MOG2_VAR_THRESH,
            detectShadows=config.MOG2_SHADOWS
        )

        # Morphological kernel for cleaning MOG2 noise
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (config.MORPH_KERNEL_SIZE, config.MORPH_KERNEL_SIZE)
        )

        log.info("Module 1 initialized")
        log.debug(f"  Config: diff_thresh={config.DIFF_THRESHOLD} | "
                  f"mog2_history={config.MOG2_HISTORY} | "
                  f"fg_ratio_min={config.FG_RATIO_MIN}")

    def process(self, frame: np.ndarray) -> M1Result:
        t0 = time.perf_counter()
        self.frame_id += 1
        fid = self.frame_id

        log.debug(f"[Frame {fid}] Processing — shape={frame.shape}")

        # ── Step 1: Convert to grayscale ────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        log.debug(
            f"[Frame {fid}] Grayscale computed — dtype={gray.dtype} shape={gray.shape}")

        # ── Step 2: MOG2 — always update model even on skipped frames ───────
        fg_mask_raw = self.bg_sub.apply(frame)

        # Clean noise with morphological opening (remove tiny isolated blobs)
        fg_mask = cv2.morphologyEx(fg_mask_raw, cv2.MORPH_OPEN, self.kernel)

        # Compute foreground ratio
        fg_pixels = int(np.count_nonzero(fg_mask))
        total_px = int(fg_mask.size)
        fg_ratio = fg_pixels / total_px
        log.debug(f"[Frame {fid}] MOG2: fg_pixels={fg_pixels} / total={total_px} "
                  f"=> fg_ratio={fg_ratio:.4f} ({fg_ratio*100:.2f}%)")

        # ── Step 3: Inter-frame difference ───────────────────────────────────
        mean_diff = 0.0
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            mean_diff = float(np.mean(diff))
            log.debug(f"[Frame {fid}] Inter-frame diff: mean={mean_diff:.3f} "
                      f"(threshold={self.cfg.DIFF_THRESHOLD})")

            if mean_diff < self.cfg.DIFF_THRESHOLD:
                self.prev_gray = gray
                self.skip_count += 1
                elapsed = (time.perf_counter() - t0) * 1000
                result = M1Result(
                    frame_id=fid, skip=True,
                    skip_reason=f"static_diff({mean_diff:.2f}<{self.cfg.DIFF_THRESHOLD})",
                    mean_diff=mean_diff, fg_ratio=fg_ratio,
                    fg_mask=fg_mask, elapsed_ms=elapsed
                )
                result.log_summary()
                return result
        else:
            log.debug(
                f"[Frame {fid}] First frame — no previous for diff comparison")

        self.prev_gray = gray

        # ── Step 4: MOG2 foreground area check ───────────────────────────────
        if fg_ratio < self.cfg.FG_RATIO_MIN:
            self.skip_count += 1
            elapsed = (time.perf_counter() - t0) * 1000
            result = M1Result(
                frame_id=fid, skip=True,
                skip_reason=f"no_foreground(fg={fg_ratio:.4f}<{self.cfg.FG_RATIO_MIN})",
                mean_diff=mean_diff, fg_ratio=fg_ratio,
                fg_mask=fg_mask, elapsed_ms=elapsed
            )
            result.log_summary()
            return result

        # ── PASS
        self.pass_count += 1
        elapsed = (time.perf_counter() - t0) * 1000
        result = M1Result(
            frame_id=fid, skip=False,
            skip_reason="",
            mean_diff=mean_diff, fg_ratio=fg_ratio,
            fg_mask=fg_mask, elapsed_ms=elapsed
        )
        result.log_summary()
        return result

    def stats(self) -> dict:
        total = self.frame_id
        return {
            "total_frames":  total,
            "skipped":       self.skip_count,
            "passed":        self.pass_count,
            "skip_pct":      round(self.skip_count / max(total, 1) * 100, 2),
        }

    def print_stats(self):
        s = self.stats()
        log.info("─" * 55)
        log.info(f"MODULE 1 STATS")
        log.info(f"  Total frames : {s['total_frames']}")
        log.info(f"  Skipped      : {s['skipped']}  ({s['skip_pct']}%)")
        log.info(f"  Passed       : {s['passed']}")
        log.info("─" * 55)

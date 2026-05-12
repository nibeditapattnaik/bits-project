
import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("M2:BlurDetector")


@dataclass
class M2Result:
    frame_id:       int
    skip:           bool
    skip_reason:    str
    laplacian_var:  float       # the sharpness score (higher = sharper)
    is_blurry:      bool
    elapsed_ms:     float = 0.0

    def log_summary(self):
        status = "SKIP" if self.skip else "PASS"
        quality = "BLURRY" if self.is_blurry else "SHARP"
        log.info(
            f"[Frame {self.frame_id:5d}] {status:4s} | "
            f"laplacian_var={self.laplacian_var:8.2f} | "
            f"quality={quality} | "
            f"elapsed={self.elapsed_ms:.2f}ms"
        )


class M2Config:
    BLUR_THRESHOLD: float = 80.0
    USE_CENTER_ROI:  bool = True
    ROI_FRACTION:    float = 0.6   # use central 60% of frame


class BlurDetector:

    def __init__(self, config: M2Config = M2Config()):
        self.cfg = config
        self.frame_id = 0
        self.skip_count = 0
        self.pass_count = 0
        self._scores = []   # for calibration logging

        log.info("Module 2 initialized")
        log.debug(f"  Config: blur_threshold={config.BLUR_THRESHOLD} | "
                  f"use_center_roi={config.USE_CENTER_ROI} | "
                  f"roi_fraction={config.ROI_FRACTION}")

    def _extract_roi(self, gray: np.ndarray) -> np.ndarray:
        if not self.cfg.USE_CENTER_ROI:
            return gray
        H, W = gray.shape
        frac = self.cfg.ROI_FRACTION
        y1, y2 = int(H * (1 - frac) / 2), int(H * (1 + frac) / 2)
        x1, x2 = int(W * (1 - frac) / 2), int(W * (1 + frac) / 2)
        roi = gray[y1:y2, x1:x2]
        log.debug(f"  ROI extracted: [{y1}:{y2}, {x1}:{x2}] = {roi.shape}")
        return roi

    def process(self, frame: np.ndarray, frame_id: Optional[int] = None) -> M2Result:

        t0 = time.perf_counter()
        self.frame_id = frame_id or (self.frame_id + 1)
        fid = self.frame_id

        log.debug(f"[Frame {fid}] M2 processing — shape={frame.shape}")

        # ── Step 1: Grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Step 2: Extract center ROI
        roi = self._extract_roi(gray)

        # ── Step 3: Laplacian
        # cv2.CV_64F ensures we capture both positive and negative edges
        laplacian = cv2.Laplacian(roi, cv2.CV_64F)
        lap_var = float(laplacian.var())
        self._scores.append(lap_var)

        log.debug(f"[Frame {fid}] Laplacian variance = {lap_var:.2f} "
                  f"(threshold={self.cfg.BLUR_THRESHOLD}) "
                  f"min_so_far={min(self._scores):.2f} "
                  f"max_so_far={max(self._scores):.2f}")

        is_blurry = lap_var < self.cfg.BLUR_THRESHOLD

        # ── Step 4: Decision
        if is_blurry:
            self.skip_count += 1
            elapsed = (time.perf_counter() - t0) * 1000
            result = M2Result(
                frame_id=fid, skip=True,
                skip_reason=f"blurry(var={lap_var:.1f}<{self.cfg.BLUR_THRESHOLD})",
                laplacian_var=lap_var, is_blurry=True, elapsed_ms=elapsed
            )
            result.log_summary()
            return result

        self.pass_count += 1
        elapsed = (time.perf_counter() - t0) * 1000
        result = M2Result(
            frame_id=fid, skip=False, skip_reason="",
            laplacian_var=lap_var, is_blurry=False, elapsed_ms=elapsed
        )
        result.log_summary()
        return result

    def calibration_report(self):
        if not self._scores:
            log.warning(
                "No frames processed yet — cannot generate calibration report")
            return
        arr = np.array(self._scores)
        log.info("─" * 55)
        log.info("MODULE 2 CALIBRATION REPORT")
        log.info(f"  Frames sampled : {len(arr)}")
        log.info(f"  Min score      : {arr.min():.2f}  (very blurry)")
        log.info(f"  Median score   : {np.median(arr):.2f}")
        log.info(f"  Mean score     : {arr.mean():.2f}")
        log.info(f"  Max score      : {arr.max():.2f}  (very sharp)")
        log.info(f"  Current thresh : {self.cfg.BLUR_THRESHOLD}")
        pct_below = (arr < self.cfg.BLUR_THRESHOLD).mean() * 100
        log.info(f"  Frames below thresh: {pct_below:.1f}% would be skipped")
        log.info("  RECOMMENDATION: if pct_below > 15%, raise threshold.")
        log.info("                  if pct_below < 1%, lower threshold.")
        log.info("─" * 55)

    def stats(self) -> dict:
        total = self.skip_count + self.pass_count
        return {
            "total_frames": total,
            "skipped":      self.skip_count,
            "passed":       self.pass_count,
            "skip_pct":     round(self.skip_count / max(total, 1) * 100, 2),
            "mean_score":   round(float(np.mean(self._scores)), 2) if self._scores else 0,
        }

    def print_stats(self):
        s = self.stats()
        log.info("─" * 55)
        log.info("MODULE 2 STATS")
        log.info(f"  Total frames : {s['total_frames']}")
        log.info(f"  Skipped      : {s['skipped']}  ({s['skip_pct']}%)")
        log.info(f"  Passed       : {s['passed']}")
        log.info(f"  Mean score   : {s['mean_score']}")
        log.info("─" * 55)

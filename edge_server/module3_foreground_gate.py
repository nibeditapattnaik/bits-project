import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

log = logging.getLogger("M3:ForegroundGate")


@dataclass
class M3Result:
    frame_id:        int
    skip:            bool
    skip_reason:     str
    fg_ratio:        float          # foreground fraction (reused from M1)
    fg_area_px:      int            # absolute foreground pixel count
    largest_blob_px: int            # largest connected component size
    num_blobs:       int            # number of foreground blobs detected
    roi_used:        str            # "full_frame" or "lower_half"
    elapsed_ms:      float = 0.0

    def log_summary(self):
        status = "SKIP" if self.skip else "PASS"
        log.info(
            f"[Frame {self.frame_id:5d}] {status:4s} | "
            f"fg_ratio={self.fg_ratio:.4f} ({self.fg_ratio*100:.2f}%) | "
            f"fg_px={self.fg_area_px} | "
            f"largest_blob={self.largest_blob_px}px | "
            f"blobs={self.num_blobs} | "
            f"elapsed={self.elapsed_ms:.2f}ms"
        )


class M3Config:
    # Minimum foreground area fraction to proceed
    FG_RATIO_MIN: float = 0.02        # 2% of frame area
    MIN_BLOB_SIZE: int = 500          # pixels in largest connected component

    USE_LOWER_ROI: bool = True
    LOWER_ROI_FRACTION: float = 0.6   # use bottom 60% of frame


class ForegroundGate:

    def __init__(self, config: M3Config = M3Config()):
        self.cfg = config
        self.frame_id = 0
        self.skip_count = 0
        self.pass_count = 0

        log.info("Module 3 initialized")
        log.debug(f"  Config: fg_ratio_min={config.FG_RATIO_MIN} | "
                  f"min_blob_size={config.MIN_BLOB_SIZE} | "
                  f"use_lower_roi={config.USE_LOWER_ROI}")

    def _apply_roi(self, fg_mask: np.ndarray) -> Tuple[np.ndarray, str]:
        """Apply lower-ROI crop if configured."""
        if not self.cfg.USE_LOWER_ROI:
            return fg_mask, "full_frame"
        H = fg_mask.shape[0]
        y_start = int(H * (1 - self.cfg.LOWER_ROI_FRACTION))
        roi = fg_mask[y_start:, :]
        log.debug(
            f"  ROI: rows {y_start}:{H} (lower {self.cfg.LOWER_ROI_FRACTION*100:.0f}%)")
        return roi, f"lower_{int(self.cfg.LOWER_ROI_FRACTION*100)}pct"

    def _analyse_blobs(self, fg_mask: np.ndarray) -> Tuple[int, int]:

        # connectedComponents returns (num_labels, label_image)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8
        )
        if num_labels <= 1:
            return 0, 0

        # stats[:,4] = area of each component. Skip label 0 (background).
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest = int(areas.max()) if len(areas) > 0 else 0
        num_blobs = int(len(areas))

        return largest, num_blobs

    def process(
        self,
        frame_id:  int,
        fg_mask:   np.ndarray,    # MOG2 mask from M1Result.fg_mask
        fg_ratio:  float          # already computed in M1 for the full frame
    ) -> M3Result:

        t0 = time.perf_counter()
        self.frame_id = frame_id
        fid = frame_id

        log.debug(f"[Frame {fid}] M3 processing — mask shape={fg_mask.shape}")

        # ── Step 1: Apply ROI (lower half of frame = road) ──────────────────
        roi_mask, roi_label = self._apply_roi(fg_mask)

        # Recompute ratio on ROI (more accurate for vehicle presence)
        roi_fg_px = int(np.count_nonzero(roi_mask))
        roi_total_px = int(roi_mask.size)
        roi_fg_ratio = roi_fg_px / roi_total_px if roi_total_px > 0 else 0.0

        log.debug(f"[Frame {fid}] ROI fg: {roi_fg_px}/{roi_total_px} "
                  f"= {roi_fg_ratio:.4f} ({roi_fg_ratio*100:.2f}%)")

        # ── Step 2: Area threshold check
        if roi_fg_ratio < self.cfg.FG_RATIO_MIN:
            self.skip_count += 1
            elapsed = (time.perf_counter() - t0) * 1000
            result = M3Result(
                frame_id=fid, skip=True,
                skip_reason=f"fg_too_small(roi={roi_fg_ratio:.4f}<{self.cfg.FG_RATIO_MIN})",
                fg_ratio=roi_fg_ratio, fg_area_px=roi_fg_px,
                largest_blob_px=0, num_blobs=0,
                roi_used=roi_label, elapsed_ms=elapsed
            )
            result.log_summary()
            return result

        # ── Step 3: Blob analysis — is it one large object or scattered noise?
        largest_blob, num_blobs = self._analyse_blobs(roi_mask)

        if largest_blob < self.cfg.MIN_BLOB_SIZE:
            self.skip_count += 1
            elapsed = (time.perf_counter() - t0) * 1000
            result = M3Result(
                frame_id=fid, skip=True,
                skip_reason=f"no_vehicle_blob(largest={largest_blob}<{self.cfg.MIN_BLOB_SIZE}px)",
                fg_ratio=roi_fg_ratio, fg_area_px=roi_fg_px,
                largest_blob_px=largest_blob, num_blobs=num_blobs,
                roi_used=roi_label, elapsed_ms=elapsed
            )
            result.log_summary()
            return result

        # ── PASS
        self.pass_count += 1
        elapsed = (time.perf_counter() - t0) * 1000
        result = M3Result(
            frame_id=fid, skip=False, skip_reason="",
            fg_ratio=roi_fg_ratio, fg_area_px=roi_fg_px,
            largest_blob_px=largest_blob, num_blobs=num_blobs,
            roi_used=roi_label, elapsed_ms=elapsed
        )
        result.log_summary()
        return result

    def stats(self) -> dict:
        total = self.skip_count + self.pass_count
        return {
            "total_frames": total,
            "skipped":      self.skip_count,
            "passed":       self.pass_count,
            "skip_pct":     round(self.skip_count / max(total, 1) * 100, 2),
        }

    def print_stats(self):
        s = self.stats()
        log.info("─" * 55)
        log.info("MODULE 3 STATS")
        log.info(f"  Total frames : {s['total_frames']}")
        log.info(f"  Skipped      : {s['skipped']}  ({s['skip_pct']}%)")
        log.info(f"  Passed       : {s['passed']}  (go to Module 4 DL)")
        log.info("─" * 55)

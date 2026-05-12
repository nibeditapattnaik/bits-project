import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from module4_vehicle_detector import Detection, COCO_MOTORCYCLE, COCO_PERSON

log = logging.getLogger("M5a:PillionCounter")


@dataclass
class MotorcycleRiderGroup:
    """One motorcycle and all riders sitting on it."""
    motorcycle:      Detection
    riders:          List[Detection]
    expanded_bbox:   Tuple[int, int, int, int]   # EXPANDED to include full persons + head space
    rider_count:     int
    triple_flag:     bool
    helmet_flag:     bool   # always True if motorcycle present

    def __str__(self):
        return (f"Motorcycle@{self.motorcycle.center} "
                f"riders={self.rider_count} "
                f"triple={self.triple_flag}")


@dataclass
class M5aResult:
    frame_id:           int
    groups:             List[MotorcycleRiderGroup] = field(
        default_factory=list)
    has_helmet_candidate: bool = False   # any motorcycle present
    has_triple_riding:    bool = False   # any motorcycle with >=3 riders
    total_motorcycles:    int = 0
    total_persons_on_moto: int = 0
    unmatched_persons:    int = 0       # persons not on any motorcycle
    elapsed_ms:           float = 0.0

    def log_summary(self):
        flags = []
        if self.has_helmet_candidate:
            flags.append("HELMET_CHECK")
        if self.has_triple_riding:
            flags.append("TRIPLE_RIDING")
        flag_str = "+".join(flags) if flags else "no_flags"
        log.info(
            f"[Frame {self.frame_id:5d}] M5a | "
            f"motorcycles={self.total_motorcycles} | "
            f"riders_on_moto={self.total_persons_on_moto} | "
            f"unmatched_persons={self.unmatched_persons} | "
            f"flags=[{flag_str}] | "
            f"{self.elapsed_ms:.2f}ms"
        )
        for g in self.groups:
            log.debug(f"    {g}")


class M5aConfig:
    # CRITICAL FIX: Much larger vertical expansion to capture heads
    # Motorcycle bbox typically cuts at shoulder - need 80% UP to get helmet
    VERT_EXPAND_UP_PCT:   float = 0.80   # Expand 80% upward to catch heads
    VERT_EXPAND_DOWN_PCT: float = 0.15   # Small expansion downward
    
    # Horizontal expansion to catch riders leaning sideways
    HORIZ_EXPAND_PCT:  float = 0.15

    # Person count threshold for triple riding flag
    TRIPLE_THRESHOLD:  int = 3

    # Minimum overlap fraction for a person to be "on" a motorcycle
    MIN_OVERLAP_FRAC:  float = 0.30


class PillionCounter:
    def __init__(self, config: M5aConfig = M5aConfig()):
        self.cfg = config
        self.frame_count = 0
        self.triple_count = 0
        self.helmet_count = 0
        log.info("Module 5a — PillionCounter initialised (HEAD-INCLUSIVE CROP)")
        log.debug(f"  vert_expand_up={config.VERT_EXPAND_UP_PCT} | "
                  f"vert_expand_down={config.VERT_EXPAND_DOWN_PCT} | "
                  f"horiz_expand={config.HORIZ_EXPAND_PCT} | "
                  f"triple_threshold={config.TRIPLE_THRESHOLD}")

    def _expand_bbox(self, det: Detection,
                     frame_h: int, frame_w: int
                     ) -> Tuple[int, int, int, int]:
        """
        CRITICAL FIX: Expand bbox UPWARD significantly to include rider heads.
        Motorcycle bbox typically ends at shoulders - need to go UP to capture helmets.
        """
        w = det.x2 - det.x1
        h = det.y2 - det.y1
        
        # ASYMMETRIC expansion - more UP than DOWN
        dv_up = int(h * self.cfg.VERT_EXPAND_UP_PCT)    # 80% upward
        dv_down = int(h * self.cfg.VERT_EXPAND_DOWN_PCT)  # 15% downward
        dh = int(w * self.cfg.HORIZ_EXPAND_PCT)

        ex1 = max(0,        det.x1 - dh)
        ey1 = max(0,        det.y1 - dv_up)    # ← CRITICAL: Expand UP to catch heads
        ex2 = min(frame_w,  det.x2 + dh)
        ey2 = min(frame_h,  det.y2 + dv_down)
        
        log.debug(f"    Expand bbox: motor=[{det.x1},{det.y1},{det.x2},{det.y2}] "
                  f"→ expanded=[{ex1},{ey1},{ex2},{ey2}] (UP+{dv_up}px to catch heads)")

        return (ex1, ey1, ex2, ey2)

    def _bbox_intersection(self, a: Tuple[int, int, int, int],
                          b: Tuple[int, int, int, int]) -> float:
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        if x2 < x1 or y2 < y1:
            return 0.0
        inter = (x2 - x1) * (y2 - y1)
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        return inter / max(area_b, 1)

    def _match_persons_to_motorcycles(self,
                                      motorcycles: List[Detection],
                                      persons: List[Detection],
                                      frame_h: int, frame_w: int
                                      ) -> List[MotorcycleRiderGroup]:
        groups = []
        matched_persons = set()

        for moto in motorcycles:
            expanded_bbox = self._expand_bbox(moto, frame_h, frame_w)
            ex1, ey1, ex2, ey2 = expanded_bbox
            riders_on_this_moto = []

            for i, person in enumerate(persons):
                if i in matched_persons:
                    continue

                px, py = person.center
                person_bbox = (person.x1, person.y1, person.x2, person.y2)

                # Check if person is on this motorcycle
                inside = (ex1 <= px <= ex2 and ey1 <= py <= ey2)
                overlap = self._bbox_intersection(expanded_bbox, person_bbox)

                if inside or overlap >= self.cfg.MIN_OVERLAP_FRAC:
                    riders_on_this_moto.append(person)
                    matched_persons.add(i)
                    log.debug(f"      Person @{person.center} matched to moto @{moto.center}")

            rider_count = len(riders_on_this_moto)
            triple_flag = rider_count >= self.cfg.TRIPLE_THRESHOLD

            group = MotorcycleRiderGroup(
                motorcycle=moto,
                riders=riders_on_this_moto,
                expanded_bbox=expanded_bbox,
                rider_count=rider_count,
                triple_flag=triple_flag,
                helmet_flag=True  # Always flag for helmet check
            )
            groups.append(group)
            log.debug(f"    Group created: {group}")

        return groups, len(persons) - len(matched_persons)

    def process(self, frame_id: int, frame: np.ndarray,
                motorcycles: List[Detection],
                persons: List[Detection]) -> M5aResult:
        t0 = time.perf_counter()
        H, W = frame.shape[:2]
        log.debug(f"[Frame {frame_id}] M5a: {len(motorcycles)} motos, {len(persons)} persons")

        if len(motorcycles) == 0:
            result = M5aResult(
                frame_id=frame_id,
                groups=[],
                has_helmet_candidate=False,
                has_triple_riding=False,
                total_motorcycles=0,
                total_persons_on_moto=0,
                unmatched_persons=len(persons),
                elapsed_ms=(time.perf_counter() - t0) * 1000
            )
            result.log_summary()
            return result

        groups, unmatched = self._match_persons_to_motorcycles(
            motorcycles, persons, H, W)

        total_riders = sum(g.rider_count for g in groups)
        has_triple = any(g.triple_flag for g in groups)

        self.frame_count += 1
        if has_triple:
            self.triple_count += 1
        self.helmet_count += len(motorcycles)

        result = M5aResult(
            frame_id=frame_id,
            groups=groups,
            has_helmet_candidate=True,  # ALWAYS true if motorcycles present
            has_triple_riding=has_triple,
            total_motorcycles=len(motorcycles),
            total_persons_on_moto=total_riders,
            unmatched_persons=unmatched,
            elapsed_ms=(time.perf_counter() - t0) * 1000
        )
        result.log_summary()
        return result

    def stats(self):
        return {
            'total_frames': self.frame_count,
            'helmet_flags': self.helmet_count,
            'triple_flags': self.triple_count,
            'triple_pct': round(self.triple_count / max(self.frame_count, 1) * 100, 2)
        }

    def print_stats(self):
        s = self.stats()
        log.info("─" * 55)
        log.info(f"M5a STATS | frames={s['total_frames']} | "
                 f"helmet={s['helmet_flags']} | triple={s['triple_flags']} "
                 f"({s['triple_pct']}%)")
        log.info("─" * 55)

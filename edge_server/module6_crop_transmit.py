"""
MODULE 6 — CROP AND TRANSMIT  (fixed crop logic)
==================================================
Key fix: crop now uses a much larger upward expansion to always
capture the rider's head region. Previous 25% symmetric padding
was cutting off heads in dashcam footage where riders appear in
the lower half of the frame.

New crop strategy:
  - Expand UP by 60% of motorcycle height   (catches head above seat)
  - Expand DOWN by 15% of motorcycle height
  - Expand LEFT/RIGHT by 30% of motorcycle width
  This guarantees the head region is always in the crop.

Also added: send ALL motorcycles in a frame as one wide crop
(full-frame width strip) so distant/partial bikes aren't missed.
"""

import cv2
import numpy as np
import requests
import logging
import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from pathlib import Path
from datetime import datetime

from module4_vehicle_detector import Detection
from module5a_pillion_counter import M5aResult, MotorcycleRiderGroup

log = logging.getLogger("M6:CropTransmit")


@dataclass
class CropPacket:
    frame_id:       int
    violation_type: str
    crop_img:       np.ndarray = field(repr=False)
    metadata:       dict       = field(default_factory=dict)
    bbox:           Tuple[int, int, int, int] = (0, 0, 0, 0)

    def __str__(self):
        h, w = self.crop_img.shape[:2]
        return f"CropPacket(frame={self.frame_id} type={self.violation_type} size={w}x{h})"


@dataclass
class M6Result:
    frame_id:        int
    skip:            bool
    skip_reason:     str
    packets:         List[CropPacket] = field(default_factory=list)
    sent_count:      int   = 0
    queued_count:    int   = 0
    failed_count:    int   = 0
    server_response: Optional[dict] = None
    elapsed_ms:      float = 0.0

    def log_summary(self):
        if self.skip:
            log.info(f"[Frame {self.frame_id:5d}] M6 SKIP | {self.skip_reason}")
        else:
            log.info(
                f"[Frame {self.frame_id:5d}] M6 SEND | "
                f"packets={len(self.packets)} | sent={self.sent_count} | "
                f"queued={self.queued_count} | {self.elapsed_ms:.0f}ms"
            )


class M6Config:
    SERVER_URL:        str   = "http://192.168.0.5:8000/detect"
    # SERVER_URL:        str   = "http://10.247.9.100:8000/detect"
    TIMEOUT_S:         float = 30.0
    MAX_RETRIES:       int   = 2
    JPEG_QUALITY:      int   = 95
    LOCAL_QUEUE_DIR:   str   = "violation_outputs"
    ALWAYS_SAVE_LOCAL: bool  = True
    STUB_TRANSMIT:     bool  = False

    # ── Crop expansion ratios (relative to motorcycle bbox dimensions) ─────────
    # These are the KEY parameters that fix the "head cut off" problem.
    EXPAND_UP_RATIO:    float = 1.20   # expand 80% of moto height UPWARD  (head!)
    EXPAND_DOWN_RATIO:  float = 0.20   # expand 20% of moto height downward
    EXPAND_SIDE_RATIO:  float = 0.35   # expand 35% of moto width each side

    # Minimum crop size in pixels — reject tiny crops
    MIN_CROP_W:  int = 60
    MIN_CROP_H:  int = 80

    # Send a "wide strip" crop covering all motorcycles in the frame
    # This catches distant bikes that M4 may have detected at low confidence
    SEND_WIDE_STRIP:   bool  = True
    # The strip spans the full width and covers road-area rows only
    WIDE_STRIP_TOP_FRAC:   float = 0.30  # top 30% of frame ignored (sky)
    WIDE_STRIP_BOTTOM_FRAC: float = 0.05  # bottom 5% ignored (bonnet edge)


class CropTransmit:
    def __init__(self, config: M6Config = M6Config()):
        self.cfg         = config
        self.send_count  = 0
        self.queue_count = 0
        self.fail_count  = 0
        self.total_frames= 0
        self.skip_count  = 0
        os.makedirs(config.LOCAL_QUEUE_DIR, exist_ok=True)
        log.info("Module 6 — CropTransmit initialised")
        log.debug(
            f"  server={config.SERVER_URL} | "
            f"expand_up={config.EXPAND_UP_RATIO} | "
            f"expand_side={config.EXPAND_SIDE_RATIO} | "
            f"stub={config.STUB_TRANSMIT}"
        )

    # ── Crop with asymmetric expansion (large upward for head) ─────────────────
    def _make_crop(self, frame: np.ndarray,
                   moto_x1: int, moto_y1: int,
                   moto_x2: int, moto_y2: int) -> Tuple[np.ndarray, Tuple]:
        """
        Crop around the motorcycle with asymmetric padding.

        Why asymmetric?
          In dashcam footage the rider sits on top of the motorcycle.
          YOLOv8 draws the motorcycle bbox tightly around the bike body.
          The rider's head and torso are ABOVE the bbox top edge.
          A symmetric 25% expansion is nowhere near enough.
          This function expands 80% upward to always include the head.
        """
        H, W = frame.shape[:2]
        bw = moto_x2 - moto_x1
        bh = moto_y2 - moto_y1

        # Asymmetric expansion
        pad_up    = int(bh * self.cfg.EXPAND_UP_RATIO)
        pad_down  = int(bh * self.cfg.EXPAND_DOWN_RATIO)
        pad_side  = int(bw * self.cfg.EXPAND_SIDE_RATIO)

        cx1 = max(0, moto_x1 - pad_side)
        cy1 = max(0, moto_y1 - pad_up)      # <─ large upward expansion
        cx2 = min(W, moto_x2 + pad_side)
        cy2 = min(H, moto_y2 + pad_down)

        crop = frame[cy1:cy2, cx1:cx2]
        log.debug(
            f"    crop: moto=[{moto_x1},{moto_y1},{moto_x2},{moto_y2}] "
            f"expanded=[{cx1},{cy1},{cx2},{cy2}] "
            f"size={crop.shape[1]}x{crop.shape[0]} "
            f"pad_up={pad_up}px pad_side={pad_side}px"
        )
        return crop, (cx1, cy1, cx2, cy2)

    # ── Wide strip crop — catches bikes M4 detected at low confidence ──────────
    def _make_wide_strip(self, frame: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        """
        Return the full-width road strip of the frame.
        This ensures we send context for any motorcycle in the road zone,
        including distant/partial bikes with low M4 confidence.
        """
        H, W = frame.shape[:2]
        y1 = int(H * self.cfg.WIDE_STRIP_TOP_FRAC)
        y2 = int(H * (1.0 - self.cfg.WIDE_STRIP_BOTTOM_FRAC))
        strip = frame[y1:y2, 0:W]
        return strip, (0, y1, W, y2)

    # ── Build packets ─────────────────────────────────────────────────────────
    def _make_packets(self, frame: np.ndarray,
                      frame_id: int,
                      m5a_result: M5aResult) -> List[CropPacket]:
        packets = []
        H, W    = frame.shape[:2]
        ts      = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

        # ── Per-motorcycle crop (one per group) ──────────────────────────────
        for group in m5a_result.groups:
            moto  = group.motorcycle
            vtype = "TRIPLE_RIDING" if group.triple_flag else "HELMET_CHECK"

            crop, crop_bbox = self._make_crop(
                frame, moto.x1, moto.y1, moto.x2, moto.y2
            )

            # Reject crops that are too small to be useful
            ch, cw = crop.shape[:2]
            if cw < self.cfg.MIN_CROP_W or ch < self.cfg.MIN_CROP_H:
                log.warning(
                    f"    Skipping tiny crop {cw}x{ch} for moto at "
                    f"[{moto.x1},{moto.y1},{moto.x2},{moto.y2}]"
                )
                continue

            meta = {
                "frame_id":       frame_id,
                "violation_type": vtype,
                "vehicle_class":  "motorcycle",
                "confidence":     round(moto.confidence, 3),
                "bbox_moto":      [moto.x1, moto.y1, moto.x2, moto.y2],
                "bbox_expanded":  list(group.expanded_bbox),
                "bbox_crop":      list(crop_bbox),
                "rider_count":    group.rider_count,
                "riders": [
                    {"bbox": [r.x1, r.y1, r.x2, r.y2],
                     "conf": round(r.confidence, 3)}
                    for r in group.riders
                ],
                "timestamp":  ts,
                "frame_w":    W,
                "frame_h":    H,
                "crop_type":  "per_motorcycle",
            }
            packets.append(CropPacket(
                frame_id=frame_id,
                violation_type=vtype,
                crop_img=crop,
                metadata=meta,
                bbox=crop_bbox
            ))

        # ── Wide strip crop — always send alongside individual crops ──────────
        # This gives the server the full road context so it can detect
        # other bikers that the Pi's M5a may have missed (different motorcycle,
        # low confidence detection, no person match).
        if self.cfg.SEND_WIDE_STRIP and m5a_result.has_helmet_candidate:
            strip, strip_bbox = self._make_wide_strip(frame)
            meta_strip = {
                "frame_id":       frame_id,
                "violation_type": "HELMET_CHECK",
                "vehicle_class":  "full_road_strip",
                "confidence":     1.0,
                "bbox_crop":      list(strip_bbox),
                "rider_count":    m5a_result.total_persons_on_moto,
                "timestamp":      ts,
                "frame_w":        W,
                "frame_h":        H,
                "crop_type":      "wide_road_strip",
            }
            packets.append(CropPacket(
                frame_id=frame_id,
                violation_type="HELMET_CHECK",
                crop_img=strip,
                metadata=meta_strip,
                bbox=strip_bbox
            ))
            log.debug(
                f"    Wide strip added: {strip.shape[1]}x{strip.shape[0]}"
            )

        return packets

    # ── Local save ────────────────────────────────────────────────────────────
    def _save_locally(self, packet: CropPacket, suffix: str = "") -> str:
        fname     = f"{packet.violation_type}_f{packet.frame_id:05d}{suffix}.jpg"
        img_path  = os.path.join(self.cfg.LOCAL_QUEUE_DIR, fname)
        meta_path = img_path.replace(".jpg", "_meta.json")
        cv2.imwrite(img_path, packet.crop_img,
                    [cv2.IMWRITE_JPEG_QUALITY, self.cfg.JPEG_QUALITY])
        with open(meta_path, "w") as f:
            json.dump(packet.metadata, f, indent=2)
        log.debug(f"    Saved locally: {img_path}")
        return img_path

    # ── HTTP transmit ─────────────────────────────────────────────────────────
    def _transmit(self, packets: List[CropPacket]):
        if not packets:
            return True, {}
        if self.cfg.STUB_TRANSMIT:
            log.info(f"    STUB: {len(packets)} crops NOT sent (stub mode)")
            return True, {"status": "stub", "received": len(packets)}
        try:
            files = []
            for i, pkt in enumerate(packets):
                ok, buf = cv2.imencode(
                    ".jpg", pkt.crop_img,
                    [cv2.IMWRITE_JPEG_QUALITY, self.cfg.JPEG_QUALITY]
                )
                if ok:
                    files.append((
                        "images",
                        (f"crop_f{pkt.frame_id:05d}_{pkt.violation_type}_{pkt.metadata.get('crop_type','')}.jpg",
                         buf.tobytes(), "image/jpeg")
                    ))
            data = {"metadata": json.dumps([p.metadata for p in packets])}
            resp = requests.post(
                self.cfg.SERVER_URL, files=files, data=data,
                timeout=self.cfg.TIMEOUT_S
            )
            if resp.status_code == 200:
                log.info(f"    Server 200 OK")
                return True, resp.json()
            log.error(f"    Server {resp.status_code}: {resp.text[:200]}")
            return False, None
        except requests.exceptions.ConnectionError as e:
            log.warning(f"    Cannot connect to {self.cfg.SERVER_URL}: {e}")
            return False, None
        except requests.exceptions.Timeout:
            log.warning(f"    Timeout after {self.cfg.TIMEOUT_S}s")
            return False, None
        except Exception as e:
            log.error(f"    Transmit error: {e}")
            return False, None

    # ── Main process ──────────────────────────────────────────────────────────
    def process(self, frame: np.ndarray, frame_id: int,
                m5a_result: M5aResult) -> M6Result:
        t0 = time.perf_counter()
        self.total_frames += 1

        if not m5a_result.has_helmet_candidate:
            self.skip_count += 1
            res = M6Result(
                frame_id=frame_id, skip=True,
                skip_reason="no_motorcycle_found",
                elapsed_ms=(time.perf_counter()-t0)*1000
            )
            res.log_summary()
            return res

        packets = self._make_packets(frame, frame_id, m5a_result)
        if not packets:
            self.skip_count += 1
            res = M6Result(
                frame_id=frame_id, skip=True,
                skip_reason="all_crops_too_small",
                elapsed_ms=(time.perf_counter()-t0)*1000
            )
            res.log_summary()
            return res

        if self.cfg.ALWAYS_SAVE_LOCAL:
            for pkt in packets:
                self._save_locally(pkt)

        sent = queued = failed = 0
        server_resp = None

        for attempt in range(1, self.cfg.MAX_RETRIES + 1):
            ok, server_resp = self._transmit(packets)
            if ok:
                sent = len(packets)
                self.send_count += sent
                break
            if attempt == self.cfg.MAX_RETRIES:
                for pkt in packets:
                    self._save_locally(pkt, "_queued")
                queued = failed = len(packets)
                self.queue_count += queued
                self.fail_count  += failed
            else:
                time.sleep(0.5)

        res = M6Result(
            frame_id=frame_id, skip=False, skip_reason="",
            packets=packets, sent_count=sent,
            queued_count=queued, failed_count=failed,
            server_response=server_resp,
            elapsed_ms=(time.perf_counter()-t0)*1000
        )
        res.log_summary()
        return res

    def stats(self):
        return dict(
            total=self.total_frames, skipped=self.skip_count,
            sent=self.send_count, queued=self.queue_count,
            failed=self.fail_count
        )

    def print_stats(self):
        s = self.stats()
        log.info("─"*55)
        log.info(f"M6 STATS | total={s['total']} | skipped={s['skipped']} | "
                 f"sent={s['sent']} | queued={s['queued']}")
        log.info("─"*55)

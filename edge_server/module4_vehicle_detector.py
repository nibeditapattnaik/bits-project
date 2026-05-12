from pathlib import Path

import cv2
import numpy as np
import logging
import time
from dataclasses import dataclass, field
from typing import List, Tuple

log = logging.getLogger("M4:VehicleDetector")

COCO_PERSON = 0
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7

HELMET_CLASSES = {COCO_PERSON, COCO_MOTORCYCLE}
LANE_CLASSES = {COCO_CAR, COCO_BUS, COCO_TRUCK}

COCO_NAMES = {
    0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"
}


@dataclass
class Detection:
    class_id:   int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int  # pixel coords in working resolution

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self): return self.x2 - self.x1
    @property
    def height(self): return self.y2 - self.y1
    @property
    def area(self): return self.width * self.height
    @property
    def center(self): return ((self.x1+self.x2)//2, (self.y1+self.y2)//2)

    def __str__(self):
        return (f"{self.class_name}({self.confidence:.2f}) "
                f"[{self.x1},{self.y1},{self.x2},{self.y2}]")


@dataclass
class M4Result:
    frame_id:     int
    skip:         bool         # True if no relevant detections found
    skip_reason:  str
    motorcycles:  List[Detection] = field(default_factory=list)
    persons:      List[Detection] = field(default_factory=list)
    cars:         List[Detection] = field(default_factory=list)   # future use
    all_dets:     List[Detection] = field(default_factory=list)
    inference_ms: float = 0.0
    elapsed_ms:   float = 0.0

    @property
    def has_motorcycle(self): return len(self.motorcycles) > 0
    @property
    def has_person(self): return len(self.persons) > 0
    @property
    def has_car_type(self): return len(self.cars) > 0

    def log_summary(self):
        status = "SKIP" if self.skip else "PASS"
        log.info(
            f"[Frame {self.frame_id:5d}] M4 {status:4s} | "
            f"motorcycles={len(self.motorcycles)} | "
            f"persons={len(self.persons)} | "
            f"cars/bus/truck={len(self.cars)} | "
            f"infer={self.inference_ms:.0f}ms | "
            f"total={self.elapsed_ms:.0f}ms"
        )
        for d in self.all_dets:
            log.debug(f"    det: {d}")


class M4Config:
    CONFIDENCE_THRESHOLD: float = 0.25   # general classes
    MOTORCYCLE_CONFIDENCE_THRESHOLD: float = 0.25  # same as infer_tflite.py
    NMS_IOU_THRESHOLD:    float = 0.45
    # INPUT_SIZE:           int = 416    # YOLOv8-nano input (320x320)
    INPUT_SIZE:           int = 320    # YOLOv8-nano input (320x320)
    SAVE_DETECTED_FRAMES: bool = False          # ← New
    SAVE_DIR: str = "m4_detected_frames"

    # Skip frame if no motorcycle AND no person found
    # (Car/bus/truck skipping is handled when 5b/5c are added)
    SKIP_IF_NO_HELMET_CLASS: bool = True


class VehicleDetector:

    def __init__(self, model_path: str, config: M4Config = M4Config()):
        self.cfg = config
        self.skip_count = 0
        self.pass_count = 0
        self.infer_times = []
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.frame_h = None   # set on first process() call
        self.frame_w = None

        if not model_path:
            raise ValueError(
                "M4: model_path is required. Please provide path to yolov8n_int8.tflite")

        # ── Load TFLite model ─────────────────────────────────────────────
        try:
            import tflite_runtime.interpreter as tflite
            log.info(f"M4: Loading model via tflite_runtime: {model_path}")
            self.interpreter = tflite.Interpreter(model_path=model_path)
        except ImportError:
            try:
                import tensorflow as tf
                log.info(
                    f"M4: Loading model via tensorflow.lite: {model_path}")
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
            except ImportError:
                raise ImportError(
                    "M4: Neither tflite_runtime nor tensorflow found. "
                    "Install one of them: "
                    "pip install tflite-runtime OR pip install tensorflow"
                )

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        inp = self.input_details[0]
        log.info(f"M4: Model loaded | "
                 f"input shape={inp['shape']} dtype={inp['dtype'].__name__}")
        log.info(f"M4: Confidence thresholds: "
                 f"default={self.cfg.CONFIDENCE_THRESHOLD}, "
                 f"motorcycle={self.cfg.MOTORCYCLE_CONFIDENCE_THRESHOLD}")
        log.debug(
            f"  Output tensors: {[o['name'] for o in self.output_details]}")

        # Test inference to validate model
        log.info("M4: Running test inference to validate model...")
        try:
            dummy = np.zeros(inp['shape'], dtype=inp['dtype'])
            self.interpreter.set_tensor(inp['index'], dummy)
            self.interpreter.invoke()
            log.info("✓ M4: Model validation successful")
        except Exception as e:
            raise RuntimeError(f"M4: Model test inference failed: {e}")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:

        resized = cv2.resize(frame, (self.cfg.INPUT_SIZE, self.cfg.INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # ALWAYS normalize to float32 [0.0, 1.0]
        # This is correct for both INT8 and Float32 YOLOv8 models
        blob = rgb.astype(np.float32) / 255.0

        return np.expand_dims(blob, axis=0)   # (1, 320, 320, 3)

    def _parse_yolov8_output(self, raw_output: np.ndarray,
                             frame_h: int, frame_w: int) -> List[Detection]:

        # Step 1: remove batch dim → (84, N)
        out = raw_output[0]

        log.debug(f"  YOLO raw output shape after [0]: {out.shape}")

        # Step 2: always transpose → (N, 84) — same as infer_tflite.py
        out = out.T   # (N, 84) where each row = one anchor box

        log.debug(f"  YOLO output shape after .T: {out.shape}")

        detections = []

        for row in out:
            # row layout: [cx, cy, bw, bh, score_cls0, score_cls1, ..., score_cls79]
            # All coordinates are NORMALISED to [0, 1]
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            class_scores = row[4:]

            if len(class_scores) < 80:
                continue

            best_cls = int(np.argmax(class_scores))
            best_conf = float(class_scores[best_cls])

            # Filter: only keep our 5 relevant COCO classes
            if best_cls not in (COCO_PERSON, COCO_MOTORCYCLE,
                                COCO_CAR, COCO_BUS, COCO_TRUCK):
                continue

            # Class-specific confidence thresholds
            if best_cls == COCO_MOTORCYCLE:
                thresh = self.cfg.MOTORCYCLE_CONFIDENCE_THRESHOLD
            else:
                thresh = self.cfg.CONFIDENCE_THRESHOLD

            if best_conf < thresh:
                log.debug(f"  Skip {COCO_NAMES.get(best_cls)} "
                          f"conf={best_conf:.3f} < {thresh:.3f}")
                continue

            # Convert normalised [0,1] coords → pixel coords in ORIGINAL FRAME
            # (same formula as infer_tflite.py line: x1 = int((x - bw/2) * w))
            x1 = int((cx - bw / 2) * frame_w)
            y1 = int((cy - bh / 2) * frame_h)
            x2 = int((cx + bw / 2) * frame_w)
            y2 = int((cy + bh / 2) * frame_h)

            # Clamp to frame boundaries
            x1 = max(0, min(x1, frame_w - 1))
            y1 = max(0, min(y1, frame_h - 1))
            x2 = max(0, min(x2, frame_w - 1))
            y2 = max(0, min(y2, frame_h - 1))

            if x2 <= x1 or y2 <= y1:
                continue

            det = Detection(
                class_id=best_cls,
                class_name=COCO_NAMES.get(best_cls, str(best_cls)),
                confidence=best_conf,
                x1=x1, y1=y1, x2=x2, y2=y2
            )
            log.debug(f"  Det: {det}")
            detections.append(det)

        log.info(f"  Raw detections before NMS: {len(detections)}")
        # Apply Non-Max Suppression
        final_dets = self._nms(detections)

        if final_dets:
            log.info(f"M4 detected {len(final_dets)} objects after NMS")
            for d in final_dets[:6]:   # show top 6 for debugging
                log.info(f"   → {d}")
        else:
            log.warning("M4: No valid detections after parsing + NMS")

        return final_dets

    def _nms(self, dets: List[Detection]) -> List[Detection]:
        """Non-maximum suppression per class."""
        if len(dets) <= 1:
            return dets
        result = []
        for cls_id in set(d.class_id for d in dets):
            cls_dets = [d for d in dets if d.class_id == cls_id]
            # Sort by confidence descending
            cls_dets.sort(key=lambda d: d.confidence, reverse=True)
            keep = []
            while cls_dets:
                best = cls_dets.pop(0)
                keep.append(best)
                cls_dets = [d for d in cls_dets
                            if self._iou(best, d) < self.cfg.NMS_IOU_THRESHOLD]
            result.extend(keep)
        log.debug(f"  Detections after NMS: {len(result)}")
        return result

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0, ix2-ix1) * max(0, iy2-iy1)
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    # ── Main process
    def process(self, frame: np.ndarray, frame_id: int) -> M4Result:
        t0 = time.perf_counter()
        fid = frame_id
        H, W = frame.shape[:2]
        log.debug(f"[Frame {fid}] M4 processing — frame {W}x{H}")

        # ── Run inference
        t_infer = time.perf_counter()
        blob = self._preprocess(frame)
        self.interpreter.set_tensor(self.input_details[0]['index'], blob)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_details[0]['index'])
        log.debug(f"RAW TFLite inference: {raw}")
        infer_ms = (time.perf_counter() - t_infer) * 1000
        all_dets = self._parse_yolov8_output(raw, H, W)

        self.infer_times.append(infer_ms)
        log.debug(f"[Frame {fid}]   TFLite inference: {infer_ms:.1f}ms")

        # ── Split by class
        motorcycles = [d for d in all_dets if d.class_id == COCO_MOTORCYCLE]
        persons = [d for d in all_dets if d.class_id == COCO_PERSON]
        cars = [d for d in all_dets if d.class_id in LANE_CLASSES]

        log.debug(
            f"[Frame {fid}]   Split: moto={len(motorcycles)} person={len(persons)} car_type={len(cars)}")
        # ── Skip decision
        if self.cfg.SKIP_IF_NO_HELMET_CLASS:
            if len(motorcycles) == 0 and len(persons) == 0:
                self.skip_count += 1
                res = M4Result(
                    frame_id=fid, skip=True,
                    skip_reason="no_motorcycle_or_person",
                    motorcycles=motorcycles, persons=persons,
                    cars=cars, all_dets=all_dets,
                    inference_ms=infer_ms,
                    elapsed_ms=(time.perf_counter()-t0)*1000
                )
                res.log_summary()
                return res

        self.pass_count += 1
        res = M4Result(
            frame_id=fid, skip=False, skip_reason="",
            motorcycles=motorcycles, persons=persons,
            cars=cars, all_dets=all_dets,
            inference_ms=infer_ms,
            elapsed_ms=(time.perf_counter()-t0)*1000
        )
        res.log_summary()

        return res

    def stats(self):
        t = self.skip_count + self.pass_count
        avg_infer = np.mean(self.infer_times) if self.infer_times else 0
        return dict(total=t, skipped=self.skip_count, passed=self.pass_count,
                    skip_pct=round(self.skip_count/max(t, 1)*100, 2),
                    avg_inference_ms=round(avg_infer, 1))

    def print_stats(self):
        s = self.stats()
        log.info("─"*55)
        log.info(f"M4 STATS | total={s['total']} | "
                 f"skipped={s['skipped']} ({s['skip_pct']}%) | "
                 f"passed={s['passed']} | "
                 f"avg_infer={s['avg_inference_ms']}ms")
        log.info("─"*55)

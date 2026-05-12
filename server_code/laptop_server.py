"""
LAPTOP SERVER — S6
=====================================
FastAPI server that receives crops from Pi, runs:
  - YOLOv8-medium  accurate re-detection
  - Helmet classifier  (EfficientNetB3 or YOLO class fallback)
  - ANPR  (PaddleOCR + Indian plate regex)
  - Violation logger  (JSON + annotated JPEG)
  - OTA endpoint  (manually triggered model push to Pi)

INSTALL:
    pip install fastapi uvicorn ultralytics pillow python-multipart
    pip install paddlepaddle paddleocr      # for ANPR
    pip install timm torch torchvision      # for EfficientNetB3

RUN:
    python3 laptop_server.py

ENDPOINTS:
    GET  /              → health + model status
    POST /detect        → main pipeline (receives crops from Pi)
    GET  /results       → list saved violations
    GET  /results/{f}   → serve annotated image
    POST /ota/push      → manually push new TFLite model to Pi
    GET  /ota/status    → check last OTA result
"""

import os, json, time, re, uuid, shutil, logging, hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-7s] %(name)-20s :: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("Server")


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG  — edit before running
# ═══════════════════════════════════════════════════════════════════════════════
class Config:
    HOST             = "0.0.0.0"
    PORT             = 8000

    # Directories
    OUTPUT_DIR       = Path("yolo_outputs")
    UPLOAD_DIR       = Path("received_images")
    VIOLATIONS_DIR   = Path("violation_records")
    UNREADABLE_DIR   = Path("unreadable_plates")
    OTA_MODEL_DIR    = Path("ota_models")

    # M8 — accurate detector
    YOLO_MODEL       = "yolov8m.pt"   # downloads ~50MB automatically first run
    YOLO_CONF        = 0.45           # lowered from 0.65 — crops may be partial
    YOLO_IOU         = 0.45

    # M9a — helmet classifier
    # Set to your trained EfficientNetB3 .pth path after Colab training.
    # None = use YOLO-label fallback (still works, lower accuracy)
    HELMET_MODEL_PATH = "helmet_efficientnet_b3.onnx"           # e.g. "helmet_efficientnet_b3.pth"
    HELMET_THRESHOLD  = 0.40           # P(no_helmet) >= this → violation
    HEAD_CROP_FRAC    = 0.45           # upper 45% of rider bbox = head region

    # M10 — ANPR
    # Lower 30% of vehicle bbox = plate zone
    # PLATE_ZONE_FRAC   = 0.30
    PLATE_ZONE_TOP_FRAC    = 0.45   # plate starts here (above tyre)
    PLATE_ZONE_BOTTOM_FRAC = 0.65   # plate ends here

    OCR_CONF_MIN      = 0.55           # below = save as unreadable
    INDIAN_PLATE_RE   = re.compile(
        r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}$", re.IGNORECASE
    )

    # OTA — Pi connection
    PI_BASE_URL       = "http://192.168.0.3:8001"   # Pi FastAPI address
    OTA_ENDPOINT      = "/ota/update"                # endpoint on Pi


cfg = Config()

for d in [cfg.OUTPUT_DIR, cfg.UPLOAD_DIR, cfg.VIOLATIONS_DIR, cfg.UNREADABLE_DIR, cfg.OTA_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════════════════════════════════════

# YOLOv8-medium
yolo_model = None
try:
    from ultralytics import YOLO
    log.info(f"Loading YOLOv8: {cfg.YOLO_MODEL}")
    yolo_model = YOLO(cfg.YOLO_MODEL)
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    yolo_model(dummy, verbose=False)   # warm up
    log.info("YOLOv8-medium ready")
except Exception as e:
    log.error(f"YOLO load failed: {e}  →  stub detections")

import onnxruntime as ort

helmet_model   = None   # kept for compatibility — will hold ort.InferenceSession
helmet_device  = None   # not used for ONNX but kept to avoid NameError
helmet_tf      = None   # not used for ONNX but kept to avoid NameError

try:
    if cfg.HELMET_MODEL_PATH and Path(cfg.HELMET_MODEL_PATH).exists():
        import onnxruntime as ort  # noqa: F401  (suppress linter warning)
        helmet_model = ort.InferenceSession(
            cfg.HELMET_MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        log.info(f"EfficientNetB3 ONNX loaded from {cfg.HELMET_MODEL_PATH}")
    else:
        log.warning("No helmet model path set → YOLO-label fallback for helmet check")
except Exception as e:
    log.error(f"ONNX load failed: {e}  →  fallback")


# PaddleOCR
ocr_engine = None
try:
    from paddleocr import PaddleOCR
    import paddle
    paddle.device.set_device("cpu")
    ocr_engine = PaddleOCR(use_angle_cls=True,
                           lang="en",
                        #    use_gpu=False,           # Explicitly disable GPU
                           enable_mkldnn=False,      # Good performance boost on CPU
                        #    , show_log=False,
                           )
    log.info("PaddleOCR ready")
except Exception as e:
    log.warning(f"PaddleOCR not available: {e}  →  ANPR returns placeholder")
    ocr_engine = None

# Global stats + OTA state
stats = {
    "requests":          0,
    "crops":             0,
    "images_received":   0,
    "helmet_violations": 0,
    "triple_violations": 0,
    "anpr_readable":     0,
    "total_violations":  0,
    "stage3_skips":      0,
    "start":             datetime.now().isoformat()
}
ota_state = {"last_push": None, "last_model": None, "last_status": "never_run"}


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 8 — ACCURATE DETECTOR
# ═══════════════════════════════════════════════════════════════════════════════

COCO_NAMES  = {0:"person", 2:"car", 3:"motorcycle", 5:"bus", 7:"truck"}
LANE_CLS    = {2, 5, 7}
HELMET_CLS  = {0, 3}


def run_m8(img_bgr: np.ndarray) -> dict:
    """
    Run YOLOv8-medium on the received crop.
    Returns motorcycles, persons, cars detected.
    """
    H, W = img_bgr.shape[:2]
    if yolo_model is None:
        # Stub: synthetic centre detection
        cx, cy = W // 2, int(H * 0.65)
        mw, mh = int(W * 0.3), int(H * 0.35)
        ph = int(H * 0.30)
        log.warning("M8 STUB: returning synthetic detections")
        return {
            "motorcycles": [{"x1": cx-mw//2, "y1": cy-mh//2,
                             "x2": cx+mw//2, "y2": cy+mh//2, "conf": 0.80}],
            "persons":     [{"x1": cx-30, "y1": cy-mh//2-ph,
                             "x2": cx+30, "y2": cy-mh//2, "conf": 0.75}],
            "cars":        [],
            "all_raw":     [],
            "infer_ms":    0,
        }

    t0      = time.perf_counter()
    results = yolo_model(img_bgr, conf=cfg.YOLO_CONF, iou=cfg.YOLO_IOU,
                          imgsz=640, verbose=False)
    infer_ms= (time.perf_counter() - t0) * 1000

    motorcycles, persons, cars, all_raw = [], [], [], []
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        name = COCO_NAMES.get(cls_id, f"cls{cls_id}")
        det  = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf,
                "class": name, "class_id": cls_id}
        all_raw.append(det)
        if cls_id == 3:   motorcycles.append(det)
        elif cls_id == 0: persons.append(det)
        elif cls_id in LANE_CLS: cars.append(det)

    log.info(f"M8: moto={len(motorcycles)} person={len(persons)} "
             f"cars={len(cars)}  {infer_ms:.0f}ms")
    return dict(motorcycles=motorcycles, persons=persons, cars=cars,
                all_raw=all_raw, infer_ms=infer_ms)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 9a — HELMET CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_head_crop(img_bgr: np.ndarray, person: dict) -> Optional[np.ndarray]:
    """Upper 45% of person bbox = head region."""
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = person["x1"], person["y1"], person["x2"], person["y2"]
    rh      = y2 - y1
    head_h  = int(rh * cfg.HEAD_CROP_FRAC)
    hx1     = max(0, x1 - 5)
    hy1     = max(0, y1 - 8)       # a few pixels above top of person bbox
    hx2     = min(W, x2 + 5)
    hy2     = min(H, y1 + head_h + 5)
    if hx2 - hx1 < 16 or hy2 - hy1 < 16:
        return None
    return img_bgr[hy1:hy2, hx1:hx2]


# def _classify_head_efficientnet(crop_bgr: np.ndarray) -> tuple:
#     """Returns (p_helmet, p_no_helmet) using EfficientNetB3."""
#     import torch
#     rgb   = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
#     inp   = helmet_tf(rgb).unsqueeze(0).to(helmet_device)
#     with torch.no_grad():
#         probs = torch.softmax(helmet_model(inp), dim=1)[0].cpu().tolist()
#     return probs[0], probs[1]   # (p_helmet, p_no_helmet)

def _classify_head_efficientnet(crop_bgr: np.ndarray) -> tuple:
    """Returns (p_helmet, p_no_helmet) using EfficientNetB3 ONNX."""
    rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img  = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img  = ((img - mean) / std).transpose(2, 0, 1)[np.newaxis]  # (1,3,224,224)
    input_name = helmet_model.get_inputs()[0].name
    logits     = helmet_model.run(None, {input_name: img})[0][0]
    exp        = np.exp(logits - logits.max())   # numerically stable softmax
    probs      = exp / exp.sum()
    return float(probs[0]), float(probs[1])      # (p_helmet, p_no_helmet)

def run_m9a(img_bgr: np.ndarray, persons: list, all_raw: list,
            frame_id: int) -> list:
    """
    Classify each rider's head.
    Returns list of rider result dicts.
    """
    results = []
    for i, person in enumerate(persons):
        crop = _extract_head_crop(img_bgr, person)

        if helmet_model is not None and crop is not None:
            p_helmet, p_no_helmet = _classify_head_efficientnet(crop)
            helmet_worn  = p_no_helmet < cfg.HELMET_THRESHOLD
            conf_score   = round(p_no_helmet, 4)
            crop_fname   = f"head_f{frame_id:05d}_r{i:02d}.jpg"
            cv2.imwrite(str(cfg.OUTPUT_DIR / crop_fname), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])
            # log.info(f"M9a rider {i}: P(no_helmet)={p_no_helmet:.3f} "
            #          f"→ {'VIOLATION' if not helmet_worn else 'OK'}")
            log.info(f"M9a rider {i}: P(no_helmet)={p_no_helmet:.3f} "
                     f"threshold={cfg.HELMET_THRESHOLD} "
                     f"→ {'VIOLATION' if not helmet_worn else 'OK (helmet worn)'}")
        else:
            px1, py1, px2, py2 = (person["x1"], person["y1"],
                                   person["x2"], person["y2"])
            helmet_worn = False   # FIX: was 'helmet_won' — typo
            no_helmet   = False
            for det in all_raw:
                if det.get("class") not in ("helmet", "no_helmet", "no-helmet"):
                    continue
                ix1 = max(px1, det["x1"]); iy1 = max(py1, det["y1"])
                ix2 = min(px2, det["x2"]); iy2 = min(py2, det["y2"])
                if max(0, ix2-ix1) * max(0, iy2-iy1) > 200:
                    if det["class"] == "helmet":
                        helmet_worn = True   # FIX: was 'helmet_won'
                    else:
                        no_helmet   = True

            if no_helmet:
                helmet_worn = False; conf_score = 0.85
            elif helmet_worn:
                conf_score = 0.15
            else:
                # Cannot determine — flag as VIOLATION (conservative)
                # YOLO standard model has no helmet class so this branch
                # is always hit. Flag violation so M11 logs it for review.
                helmet_worn = False; conf_score = 0.70
            log.info(f"M9a rider {i} (fallback): helmet={helmet_worn} "
                     f"conf={conf_score}")

        results.append({
            "rider_idx":   i,
            "bbox":        [person["x1"], person["y1"],
                            person["x2"], person["y2"]],
            "helmet_worn": helmet_worn,
            "p_no_helmet": conf_score,
            "violation":   not helmet_worn,
        })
    return results

def _deblur_plate(zone: np.ndarray) -> np.ndarray:
    """
    Sharpen a blurry plate crop using:
      1. Upscale 2x (helps OCR on small plates)
      2. Unsharp mask  (edge enhancement)
      3. CLAHE         (contrast boost for faded plates)
      4. Bilateral filter (reduce noise without blurring edges)
    """
    # 1. Upscale 2x — small plate crops are too low-res for OCR
    zone = cv2.resize(zone, (zone.shape[1]*2, zone.shape[0]*2),
                      interpolation=cv2.INTER_CUBIC)

    # 2. Unsharp mask — sharpens edges
    gaussian = cv2.GaussianBlur(zone, (0, 0), sigmaX=2)
    zone = cv2.addWeighted(zone, 1.8, gaussian, -0.8, 0)

    # 3. CLAHE — boost local contrast (helps faded/dark plates)
    lab  = cv2.cvtColor(zone, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
    l     = clahe.apply(l)
    zone  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # 4. Bilateral filter — smooth noise, keep plate edges crisp
    zone = cv2.bilateralFilter(zone, d=5, sigmaColor=50, sigmaSpace=50)

    return zone


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 10 — ANPR
# ═══════════════════════════════════════════════════════════════════════════════

def run_m10(img_bgr: np.ndarray, vehicle_bbox: dict,
            frame_id: int, vid: str) -> dict:
    """
    Run PaddleOCR on the lower 30% of the vehicle bbox.
    Returns plate text, confidence, readability flag.
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = (vehicle_bbox["x1"], vehicle_bbox["y1"],
                       vehicle_bbox["x2"], vehicle_bbox["y2"])
    # bh     = y2 - y1
    # pz_y1  = max(0, y2 - int(bh * cfg.PLATE_ZONE_FRAC))
    # pz_y2  = min(H, y2 + 8)
    # zone   = img_bgr[pz_y1:pz_y2, max(0, x1-5):min(W, x2+5)]

    bh    = y2 - y1
    pz_y1 = max(0, y1 + int(bh * cfg.PLATE_ZONE_TOP_FRAC))
    pz_y2 = max(0, y1 + int(bh * cfg.PLATE_ZONE_BOTTOM_FRAC))
    zone  = img_bgr[pz_y1:pz_y2, max(0, x1-10):min(W, x2+10)]

    # ── Deblur before saving and OCR ─────────────────────────────────────────
    if zone.shape[0] >= 8 and zone.shape[1] >= 16:
        zone = _deblur_plate(zone)

    # Save plate zone crop
    zone_fname = f"plate_f{frame_id:05d}_{vid[:8]}.jpg"
    zone_path  = cfg.OUTPUT_DIR / zone_fname
    cv2.imwrite(str(zone_path), zone, [cv2.IMWRITE_JPEG_QUALITY, 90])

    if zone.shape[0] < 8 or zone.shape[1] < 16:
        return dict(plate_text="UNREADABLE", plate_conf=0.0,
                    readable=False, valid_indian=False,
                    zone_path=str(zone_path))

    if ocr_engine is None:
        return dict(plate_text="ANPR_PLACEHOLDER", plate_conf=0.0,
                    readable=False, valid_indian=False,
                    zone_path=str(zone_path))

    try:
        # ocr_result = ocr_engine.ocr(zone, cls=True)
        try:
            ocr_result = ocr_engine.predict(zone)          # new API
        except TypeError:
            ocr_result = ocr_engine.ocr(zone, cls=True)    # old API fallback
        if not ocr_result or not ocr_result[0]:
            return dict(plate_text="UNREADABLE", plate_conf=0.0,
                        readable=False, valid_indian=False,
                        zone_path=str(zone_path))

        lines = []
        if ocr_result:
            first = ocr_result[0]
            if isinstance(first, dict):
                # New predict() API
                for item in ocr_result:
                    text  = item.get("rec_text", "") or item.get("text", "")
                    score = float(item.get("rec_score", 0.0) or item.get("confidence", 0.0))
                    if text:
                        lines.append((text, score))
            elif isinstance(first, list) and first:
                # Old ocr() API
                for line in first:
                    if line and len(line) >= 2:
                        text, conf = line[1]
                        lines.append((text, float(conf)))

        if not lines:
            log.warning("M10: OCR returned no text — saving as UNREADABLE")
            unread_p = cfg.UNREADABLE_DIR / zone_fname
            shutil.copy2(str(zone_path), str(unread_p))
            log.debug(f"M10: saved to unreadable_plates/{zone_fname}")
            return dict(plate_text="UNREADABLE", plate_conf=0.0,
                        readable=False, valid_indian=False,
                        zone_path=str(zone_path))

        best_text, best_conf = max(lines, key=lambda x: x[1])
        log.debug(f"  OCR best: '{best_text}'  conf={best_conf:.3f}")

        # ── Clean + validate ──────────────────────────────────────────────────
        cleaned  = re.sub(r"[^A-Z0-9]", "", best_text.upper())
        readable = best_conf >= cfg.OCR_CONF_MIN
        valid    = bool(cfg.INDIAN_PLATE_RE.match(cleaned)) if readable else False

        log.info(f"M10 ANPR: raw='{best_text}' cleaned='{cleaned}' "
                f"conf={best_conf:.3f} readable={readable} valid_indian={valid}")

        if not readable:
            unread_p = cfg.UNREADABLE_DIR / zone_fname
            shutil.copy2(str(zone_path), str(unread_p))
            log.warning(f"M10: low OCR conf ({best_conf:.3f}) — saved to unreadable_plates/")

        return dict(plate_text=cleaned, plate_conf=round(best_conf, 4),
                    readable=readable, valid_indian=valid,
                    zone_path=str(zone_path))
    except Exception as e:
        log.error(f"M10 OCR error: {e}")
        return dict(plate_text="OCR_ERROR", plate_conf=0.0,
                    readable=False, valid_indian=False,
                    zone_path=str(zone_path))

# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE 11 — VIOLATION LOGGER
# ═══════════════════════════════════════════════════════════════════════════════

def _annotate(img_bgr: np.ndarray, m8: dict, riders: list,
              anpr: dict, vtype: str) -> np.ndarray:
    vis = img_bgr.copy()
    # Draw all M8 detections
    for det in m8.get("all_raw", []):
        col = {"motorcycle": (255, 255, 0), "person": (0, 255, 0),
               "car": (0, 200, 255)}.get(det["class"], (180, 180, 180))
        cv2.rectangle(vis, (det["x1"], det["y1"]),
                      (det["x2"], det["y2"]), col, 2)
        cv2.putText(vis, f"{det['class']} {det['conf']:.2f}",
                    (det["x1"], det["y1"] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    # Draw rider helmet results
    for r in riders:
        x1, y1, x2, y2 = r["bbox"]
        col = (0, 0, 255) if r["violation"] else (0, 255, 0)
        lbl = f"NO HELMET {r['p_no_helmet']:.2f}" if r["violation"] \
              else f"helmet {r['p_no_helmet']:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
        cv2.putText(vis, lbl, (x1, y2 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
    # Plate text
    plate = anpr.get("plate_text", "")
    if plate and plate not in ("ANPR_PLACEHOLDER", "UNREADABLE", ""):
        cv2.putText(vis, f"PLATE: {plate}", (10, vis.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    # Banner
    cv2.rectangle(vis, (0, 0), (vis.shape[1], 28), (0, 0, 180), -1)
    cv2.putText(vis, vtype, (6, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return vis


def run_m11(frame_id: int, vtype: str, vehicle: dict,
            img_bgr: np.ndarray, raw_crop_path: str,
            m8: dict, riders: list, anpr: dict,
            pi_meta: dict, processing_ms: float,
            device_id: str = "unknown") -> dict:
    """Build violation record, save annotated image, write JSON."""
    vid = str(uuid.uuid4())[:12]
    ts  = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    ann_fname = f"violation_{vid}_{vtype}.jpg"
    ann_path  = cfg.OUTPUT_DIR / ann_fname
    annotated = _annotate(img_bgr, m8, riders, anpr, vtype)
    cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])

    record = {
        "violation_id":       vid,
        "frame_id":           frame_id,
        "timestamp":          ts,
        "violation_type":     vtype,
        "device_id":          device_id,
        "vehicle_class":      vehicle.get("class", "motorcycle"),
        "vehicle_bbox":       [vehicle["x1"], vehicle["y1"],
                               vehicle["x2"], vehicle["y2"]],
        "plate_text":         anpr.get("plate_text", ""),
        "plate_conf":         anpr.get("plate_conf", 0.0),
        "plate_readable":     anpr.get("readable", False),
        "plate_valid_indian": anpr.get("valid_indian", False),
        "gps_lat":            pi_meta.get("gps_lat"),
        "gps_lon":            pi_meta.get("gps_lon"),
        "riders":             riders,
        "raw_crop_path":      raw_crop_path,
        "annotated_path":     str(ann_path),
        "annotated_file":     ann_fname,
        "processing_ms":      round(processing_ms, 2),
        "pi_meta":            pi_meta,
    }

    json_path = cfg.VIOLATIONS_DIR / f"{vid}.json"
    with open(json_path, "w") as f:
        json.dump(record, f, indent=2)

    helmetless_count = sum(1 for r in riders if r["violation"])
    log.info(f"M11: violation [{vid}] | type={vtype} | "
             f"plate={record['plate_text']} | saved {ann_fname}")
    log.warning(
        f"VIOLATION | id={vid} | type={vtype} | device={device_id} | "
        f"plate='{record['plate_text']}' readable={record['plate_readable']} | "
        f"riders={len(riders)} helmetless={helmetless_count} | {ann_fname}"
    )
    return record


# ═══════════════════════════════════════════════════════════════════════════════
#  STAGE 3 PIPELINE  (orchestrates M8 → M9a → M10 → M11)
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage3(img_bgr: np.ndarray, crop_path: str, pi_meta: dict,
               device_id: str = "unknown") -> Optional[dict]:
    """
    Full Stage 3 for one received crop.
    Returns violation record dict or None if no confirmed violation.
    """
    t0         = time.perf_counter()
    frame_id   = pi_meta.get("frame_id", 0)
    vtype_flag = pi_meta.get("violation_type", "HELMET_CHECK")
    crop_type  = pi_meta.get("crop_type", "per_motorcycle")

    log.info(f"Stage3: frame={frame_id} flag={vtype_flag} crop_type={crop_type}")

    m8 = run_m8(img_bgr)

    if vtype_flag in ("HELMET_CHECK", "TRIPLE_RIDING"):
        persons     = m8["persons"]
        motorcycles = m8["motorcycles"]

        if not motorcycles and crop_type != "wide_road_strip":
            stats["stage3_skips"] += 1
            log.warning(f"Stage3 SKIP — no motorcycle found by M8 (frame={frame_id} crop_type={crop_type})")
            return None

        if not persons:
            stats["stage3_skips"] += 1
            log.warning(f"Stage3 SKIP — no persons detected by M8 (frame={frame_id})")
            return None

        riders     = run_m9a(img_bgr, persons, m8["all_raw"], frame_id)
        helmetless = [r for r in riders if r["violation"]]
        

        # FIX: triple riding only valid on per_motorcycle crop
        # Wide strip spans the whole road — 3 persons across different
        # bikes must NOT be counted as triple riding on one motorcycle.
        triple     = (len(riders) >= 3) and (crop_type == "per_motorcycle")

        if not helmetless and not triple:
            stats["stage3_skips"] += 1
            log.warning(f"Stage3 SKIP — all {len(riders)} rider(s) have helmets (frame={frame_id})")
            return None

        final_vtype = "TRIPLE_RIDING" if triple else "HELMET_VIOLATION"
        primary_veh = motorcycles[0] if motorcycles else {
            "x1": 0, "y1": 0,
            "x2": img_bgr.shape[1], "y2": img_bgr.shape[0],
            "conf": 1.0, "class": "motorcycle"
        }

        log.warning(f"Stage3: {final_vtype} confirmed | "
                    f"helmetless={len(helmetless)} triple={triple}")
        stats["helmet_violations"] += 1
        if triple: stats["triple_violations"] += 1

    else:
        log.warning(f"Stage3: unknown vtype_flag={vtype_flag}")
        return None

    # M10: ANPR — always run and always log result
    anpr = run_m10(img_bgr, primary_veh, frame_id, str(uuid.uuid4())[:8])
    log.info(f"M10 ANPR: plate='{anpr['plate_text']}' "
             f"conf={anpr['plate_conf']} readable={anpr['readable']} "
             f"valid_indian={anpr.get('valid_indian', False)}")
    if anpr["readable"]:
        stats["anpr_readable"] += 1

    processing_ms = (time.perf_counter() - t0) * 1000
    record = run_m11(
        frame_id, final_vtype, primary_veh,
        img_bgr, crop_path, m8, riders, anpr, pi_meta, processing_ms, device_id
    )
    stats["total_violations"] += 1
    return record

# ═══════════════════════════════════════════════════════════════════════════════
#  OTA UPDATE  —  manually push new TFLite model to Pi
# ═══════════════════════════════════════════════════════════════════════════════

def _do_ota_push(model_path: str):
    """
    Background task: POST the new TFLite file to Pi's /ota/update endpoint.
    Pi saves it and hot-reloads Module 4 on the next pipeline run.
    """
    import requests
    try:
        p = Path(model_path)
        if not p.exists():
            ota_state["last_status"] = f"error: file not found {model_path}"
            return
        sha256 = hashlib.sha256(p.read_bytes()).hexdigest()
        log.info(f"OTA push: {p.name}  sha256={sha256[:16]}…")

        with open(p, "rb") as f:
            resp = requests.post(
                cfg.PI_BASE_URL + cfg.OTA_ENDPOINT,
                files={"model": (p.name, f, "application/octet-stream")},
                data={"sha256": sha256},
                timeout=60,
            )
        if resp.status_code == 200:
            msg = f"ok — Pi response: {resp.json()}"
        else:
            msg = f"error {resp.status_code}: {resp.text[:200]}"

        ota_state.update({
            "last_push":   datetime.now().isoformat(),
            "last_model":  p.name,
            "last_status": msg,
        })
        log.info(f"OTA result: {msg}")

    except Exception as e:
        ota_state["last_status"] = f"exception: {e}"
        log.error(f"OTA push failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title="Traffic Violation Detection — Stage 3 Server",
    description="Receives Pi crops → M8 → M9a → M10 → M11",
    version="2.0.0",
)

# Add this block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # For development only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {
        "status":   "running",
        "models": {
            "yolo_medium":    yolo_model is not None,
            "helmet_efficientnetb3": helmet_model is not None,
            "paddleocr":      ocr_engine is not None,
        },
        "stats":    stats,
        "violations_saved": len(list(cfg.VIOLATIONS_DIR.glob("*.json"))),
    }


@app.post("/detect")
async def detect(
    request: Request,
    images:   List[UploadFile] = File(...),
    metadata: str = Form(default="[]"),
):
    """
    Main endpoint — receives crops from Pi Module 6.
    Runs M8 → M9a → M10 → M11 on each crop.
    """
    if not images:
        raise HTTPException(400, "No images received")
    
    device_id = request.headers.get("X-Device-ID",
                request.client.host if request.client else "unknown")

    stats["requests"] += 1
    stats["images_received"] += len(images)
    stats["crops"]           += len(images)

    now = datetime.now().isoformat()
    if device_id in DEVICE_REGISTRY:
        DEVICE_REGISTRY[device_id]["last_seen"] = now
        DEVICE_REGISTRY[device_id]["crops_sent"] = \
            DEVICE_REGISTRY[device_id].get("crops_sent", 0) + len(images)
    else:
        # Auto-register unknown device on first crop
        DEVICE_REGISTRY[device_id] = {
            "device_id":     device_id,
            "device_name":   device_id,
            "device_type":   "raspberry_pi",
            "ip_address":    request.client.host if request.client else "unknown",
            "status":        "active",
            "registered_at": now,
            "last_seen":     now,
            "crops_sent":    len(images),
            "violations":    0,
        }
        log.info(f"Auto-registered new device: {device_id}")
    save_device_registry()

    t_req = time.perf_counter()

    try:
        meta_list = json.loads(metadata) if metadata else []
    except json.JSONDecodeError:
        meta_list = []

    results = []

    for i, upload in enumerate(images):
        stats["crops"] += 1
        pi_meta  = meta_list[i] if i < len(meta_list) else {}

        raw_bytes = await upload.read()
        np_arr    = np.frombuffer(raw_bytes, np.uint8)
        img_bgr   = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            log.error(f"Cannot decode image {i}: {upload.filename}")
            results.append({"image_index": i, "error": "decode_failed", "violation": None})
            continue

        ts        = int(time.time())
        orig_name = Path(upload.filename).stem if upload.filename else f"crop_{i}"
        recv_path = cfg.UPLOAD_DIR / f"{ts}_{orig_name}.jpg"
        cv2.imwrite(str(recv_path), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

        log.info(f"Received crop {i} from device={device_id}: "
                 f"{img_bgr.shape[1]}x{img_bgr.shape[0]} "
                 f"type={pi_meta.get('violation_type','?')} "
                 f"frame={pi_meta.get('frame_id','?')}")

        try:
            record = run_stage3(img_bgr, str(recv_path), pi_meta, device_id)
        except Exception as e:
            log.error(f"Stage3 error on crop {i}: {e}", exc_info=True)
            record = None

        if record:
            DEVICE_REGISTRY[device_id]["violations"] = \
                DEVICE_REGISTRY[device_id].get("violations", 0) + 1

        results.append({
            "image_index": i,
            "filename":    orig_name,
            "violation":   record,
            "confirmed":   record is not None,
        })

    req_ms    = (time.perf_counter() - t_req) * 1000
    confirmed = sum(1 for r in results if r["confirmed"])
    log.info(f"Request done: {len(images)} crops → {confirmed} violations | {req_ms:.0f}ms")
    
    if confirmed > 0:
        log.warning(f"Request done: {len(images)} crops → ⚠ {confirmed} VIOLATION(S) | {req_ms:.0f}ms")
    else:
        log.info(f"Request done: {len(images)} crops → 0 violations | {req_ms:.0f}ms")

    return JSONResponse({
        "status":               "ok",
        "crops_received":       len(images),
        "violations_confirmed": confirmed,
        "processing_ms":        round(req_ms, 2),
        "results":              results,
    })


@app.get("/results")
def list_results():
    images = sorted(cfg.OUTPUT_DIR.glob("violation_*.jpg"))
    logs   = sorted(cfg.VIOLATIONS_DIR.glob("*.json"))
    return {
        "violation_images": [f.name for f in images],
        "violation_logs":   [f.name for f in logs],
        "total":            len(images),
    }


@app.get("/results/{filename}")
def get_result(filename: str):
    for d in [cfg.OUTPUT_DIR, cfg.UPLOAD_DIR]:
        p = d / filename
        if p.exists():
            return FileResponse(str(p))
    raise HTTPException(404, "File not found")


@app.get("/violations")
def list_violations(limit: int = 50):
    records = sorted(cfg.VIOLATIONS_DIR.glob("*.json"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in records[:limit]:
        try:
            d = json.load(open(p))
            out.append({
                "id":         d["violation_id"],
                "timestamp":  d["timestamp"],
                "type":       d["violation_type"],
                "plate":      d["plate_text"],
                "annotated":  Path(d["annotated_path"]).name,
            })
        except Exception:
            continue
    return {"count": len(out), "violations": out}


@app.get("/violations/{vid}")
def get_violation(vid: str):
    matches = list(cfg.VIOLATIONS_DIR.glob(f"{vid}*.json"))
    if not matches:
        raise HTTPException(404, f"Violation {vid} not found")
    return json.load(open(matches[0]))


# ── OTA endpoints ─────────────────────────────────────────────────────────────

@app.post("/ota/push")
async def ota_push(background_tasks: BackgroundTasks,
                   model_filename: str = Form(...)):
    """
    Manually trigger OTA push of a TFLite model to the Pi.

    Usage:
        curl -X POST http://localhost:8000/ota/push \\
             -F "model_filename=yolov8n_coco_320_int8.tflite"

    The model file must already be in the ota_models/ directory on the server.
    The Pi must be running with its /ota/update endpoint active.

    Flow:
        1. Validates model file exists in ota_models/
        2. Computes SHA-256 hash for integrity check
        3. POSTs file to Pi's /ota/update endpoint (background task)
        4. Pi saves file, verifies hash, hot-reloads Module 4
        5. Poll GET /ota/status to check result
    """
    model_path = cfg.OTA_MODEL_DIR / model_filename
    if not model_path.exists():
        # List what's available
        available = [f.name for f in cfg.OTA_MODEL_DIR.glob("*.tflite")]
        raise HTTPException(
            404,
            f"Model '{model_filename}' not found in ota_models/. "
            f"Available: {available}. "
            f"Copy your .tflite file to ota_models/ first."
        )

    size_mb = model_path.stat().st_size / 1e6
    sha256  = hashlib.sha256(model_path.read_bytes()).hexdigest()

    ota_state["last_status"] = "pushing..."
    background_tasks.add_task(_do_ota_push, str(model_path))

    log.info(f"OTA push queued: {model_filename} ({size_mb:.1f}MB)")
    return {
        "status":   "push_queued",
        "model":    model_filename,
        "size_mb":  round(size_mb, 2),
        "sha256":   sha256,
        "pi_url":   cfg.PI_BASE_URL + cfg.OTA_ENDPOINT,
        "note":     "Poll GET /ota/status for result",
    }


@app.get("/ota/status")
def ota_status():
    """Check the result of the last OTA push."""
    return {
        "ota_state":    ota_state,
        "models_ready": [f.name for f in cfg.OTA_MODEL_DIR.glob("*.tflite")],
        "pi_url":       cfg.PI_BASE_URL,
    }


@app.get("/stats")
def get_stats():
    return {
        **stats,
        "violation_files":   len(list(cfg.VIOLATIONS_DIR.glob("*.json"))),
        "annotated_images":  len(list(cfg.OUTPUT_DIR.glob("violation_*.jpg"))),
        "unreadable_plates": len(list(cfg.UNREADABLE_DIR.glob("*.jpg"))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

from datetime import datetime, timedelta
from typing import Dict, List
import json

# In-memory device registry (use SQLite/DB for production)
DEVICE_REGISTRY: Dict[str, dict] = {}
DEVICES_FILE = Path("device_registry.json")

# Load existing devices on startup
if DEVICES_FILE.exists():
    try:
        DEVICE_REGISTRY = json.loads(DEVICES_FILE.read_text())
        log.info(f"Loaded {len(DEVICE_REGISTRY)} registered devices")
    except:
        pass

def save_device_registry():
    """Persist device registry to disk"""
    try:
        DEVICES_FILE.write_text(json.dumps(DEVICE_REGISTRY, indent=2))
    except Exception as e:
        log.error(f"Failed to save device registry: {e}")


@app.post("/devices/register")
async def register_device(device_data: dict):
    """
    Register or update an edge device.
    
    Payload:
    {
        "device_id": "rpi_0000000012345678",
        "device_name": "raspberrypi",
        "device_type": "raspberry_pi",
        "ip_address": "192.168.0.3",
        "cpu_serial": "0000000012345678",
        "mac_address": "b8:27:eb:12:34:56",
        "status": "active" | "inactive",
        "timestamp": "2026-05-10T14:30:00",
        "capabilities": {...}
    }
    """
    device_id = device_data.get("device_id")
    if not device_id:
        raise HTTPException(400, "device_id required")
    
    # Update or create device entry
    if device_id in DEVICE_REGISTRY:
        # Existing device - update
        DEVICE_REGISTRY[device_id].update(device_data)
        DEVICE_REGISTRY[device_id]["last_updated"] = datetime.now().isoformat()
        action = "updated"
    else:
        # New device - register
        device_data["registered_at"] = datetime.now().isoformat()
        device_data["last_updated"] = datetime.now().isoformat()
        DEVICE_REGISTRY[device_id] = device_data
        action = "registered"
    
    save_device_registry()
    
    log.info(f"Device {action}: {device_id} ({device_data.get('device_name')}) → {device_data.get('status')}")
    
    return {
        "status": "success",
        "action": action,
        "device_id": device_id,
        "total_devices": len(DEVICE_REGISTRY),
    }


@app.post("/devices/heartbeat")
async def device_heartbeat(heartbeat_data: dict):
    """
    Receive heartbeat from edge device.
    
    Payload:
    {
        "device_id": "rpi_0000000012345678",
        "timestamp": "2026-05-10T14:30:00",
        "status": "active",
        "pipeline_running": true
    }
    """
    device_id = heartbeat_data.get("device_id")
    if not device_id or device_id not in DEVICE_REGISTRY:
        raise HTTPException(404, f"Device {device_id} not registered")
    
    # Update last_seen timestamp
    DEVICE_REGISTRY[device_id]["last_seen"] = datetime.now().isoformat()
    DEVICE_REGISTRY[device_id]["status"] = heartbeat_data.get("status", "active")
    DEVICE_REGISTRY[device_id]["pipeline_running"] = heartbeat_data.get("pipeline_running", False)
    
    save_device_registry()
    
    return {"status": "ok", "device_id": device_id}


@app.get("/devices")
def list_devices(status: str = None):
    """
    List all registered devices.
    
    Query params:
        status: Filter by status (active | inactive)
    """
    devices = []
    
    for device_id, data in DEVICE_REGISTRY.items():
        if status and data.get("status") != status:
            continue
        
        # Calculate if device is online (heartbeat within 5 minutes)
        last_seen = data.get("last_seen")
        online = False
        if last_seen:
            try:
                last_seen_dt = datetime.fromisoformat(last_seen)
                online = (datetime.now() - last_seen_dt) < timedelta(minutes=5)
            except:
                pass
        
        devices.append({
            "device_id": data.get("device_id"),
            "device_name": data.get("device_name"),
            "device_type": data.get("device_type"),
            "ip_address": data.get("ip_address"),
            "status": data.get("status"),
            "online": online,
            "pipeline_running": data.get("pipeline_running", False),
            "registered_at": data.get("registered_at"),
            "last_seen": last_seen,
        })
    
    return {
        "total": len(devices),
        "devices": devices,
    }


@app.get("/devices/{device_id}")
def get_device(device_id: str):
    """Get detailed info about a specific device"""
    if device_id not in DEVICE_REGISTRY:
        raise HTTPException(404, f"Device {device_id} not found")
    
    return DEVICE_REGISTRY[device_id]


@app.delete("/devices/{device_id}")
def unregister_device(device_id: str):
    """Manually unregister a device"""
    if device_id not in DEVICE_REGISTRY:
        raise HTTPException(404, f"Device {device_id} not found")
    
    device_name = DEVICE_REGISTRY[device_id].get("device_name")
    del DEVICE_REGISTRY[device_id]
    save_device_registry()
    
    log.info(f"Device unregistered: {device_id} ({device_name})")
    
    return {
        "status": "success",
        "device_id": device_id,
        "remaining_devices": len(DEVICE_REGISTRY),
    }

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'─'*60}")
    print(f"  Traffic Violation Detection — Stage 3 Server v2")
    print(f"  Listening : http://{cfg.HOST}:{cfg.PORT}")
    print(f"  Pi POST   : http://<THIS_IP>:{cfg.PORT}/detect")
    print(f"  Violations: http://localhost:{cfg.PORT}/violations")
    print(f"  OTA push  : POST /ota/push  (model must be in ota_models/)")
    print(f"  OTA status: GET  /ota/status")
    print(f"{'─'*60}")
    print(f"  YOLO-medium  : {'ready' if yolo_model else 'NOT loaded'}")
    print(f"  EfficientNetB3: {'ready' if helmet_model else 'NOT set — YOLO fallback'}")
    print(f"  PaddleOCR    : {'ready' if ocr_engine else 'NOT installed'}")
    print(f"{'─'*60}\n")
    uvicorn.run(app, host=cfg.HOST, port=cfg.PORT, reload=False)

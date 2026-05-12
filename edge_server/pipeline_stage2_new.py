"""
pipeline_stage2_new.py — Stage 1 + Stage 2 pipeline running on Raspberry Pi.

BUFFERING ARCHITECTURE:
  Producer thread  →  stage2_queue (bounded)  →  Consumer thread
  ──────────────────────────────────────────────────────────────
  Thread 1 (Producer): reads frames + runs Stage 1 (M1+M2+M3)
                        puts passed frames into bounded queue
                        blocks on queue.put() when full → backpressure, no OOM
  Thread 2 (Consumer): pulls from queue + runs Stage 2 (M4+M5a+M6)
                        YOLO inference runs without dropping Stage 1 frames
  Main thread:          _run_batch() waits on producer.join() then queue.join()

WHY 901 FRAMES FOR A 1800-FRAME VIDEO?
  With --every_n 2 the producer reads every 2nd frame → 900 frames reach
  the every_n gate. Of those, Stage 1 (M1/M2/M3) discards static/blurry
  frames. So "901" = frames that passed the every_n gate, not frames that
  passed all Stage 1 filters. Previously pipeline_id was incremented BEFORE
  Stage 1, so it counted every-n frames, not just Stage-1-passed frames.
  This is now fixed: pipeline_id only increments when a frame passes Stage 1.

QUEUE SIZE GUIDELINES (--queue-size):
  Pi 1GB RAM → 2   Pi 2GB RAM → 5 (default)   Pi 4GB+ → 10

MODES:
  --mode trigger   REST endpoint triggers processing
  --mode schedule  Auto-runs every SCHEDULE_HOURS from VIDEO_FOLDER
  --mode once      Process --video file and exit
"""

import cv2
import numpy as np
import argparse
import logging
import time
import json
import hashlib
import queue
import threading
import uuid
import socket
import platform
import dataclasses
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import requests
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

# Stage 1
from module1_frame_filter  import FrameLevelFilter, M1Config
from module2_blur_detector import BlurDetector,      M2Config
from module3_foreground_gate import ForegroundGate,  M3Config

# Stage 2
from module4_vehicle_detector   import VehicleDetector,  M4Config
from module5a_pillion_counter   import PillionCounter,   M5aConfig
from module6_crop_transmit      import CropTransmit,     M6Config


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

VIDEO_FOLDER      = Path("./videos")
VIDEO_EXTS        = {".mp4", ".avi", ".mkv", ".mov"}
SCHEDULE_HOURS    = 1
OTA_PORT          = 8001
OTA_MODEL_DIR     = Path("ota_models")
OTA_MODEL_DIR.mkdir(parents=True, exist_ok=True)
LAPTOP_SERVER_URL = "http://192.168.0.5:8000"   # ← update to your laptop IP


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGGING
# ═══════════════════════════════════════════════════════════════════════════════

def setup_logging(level_str: str):
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-7s] %(name)-22s :: %(message)s",
        datefmt="%H:%M:%S"
    )

log = logging.getLogger("Pipeline:Stage1+2")
logging.getLogger("M4:VehicleDetector").setLevel(logging.INFO)


# ═══════════════════════════════════════════════════════════════════════════════
#  QUEUE ITEM — passed from producer to consumer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass
class _QueueItem:
    """Frame that passed Stage 1, waiting for Stage 2 in the consumer."""
    frame_id:    int          # raw video frame number (1-based)
    pipeline_id: int          # counter of Stage-1-passed frames (1-based)
    frame:       np.ndarray   # already resized, copied
    m1r:         object
    m2r:         object
    m3r:         object
    # Timing — producer records when each stage finished
    t_read_ms:   float = 0.0  # cap.read() wall time
    t_m1_ms:     float = 0.0
    t_m2_ms:     float = 0.0
    t_m3_ms:     float = 0.0
    t_queued_ms: float = 0.0  # total producer time before queue.put()


# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED STATE
# ═══════════════════════════════════════════════════════════════════════════════

_ota_state = {
    "pending_model": None,
    "current_model": None,
    "last_update":   None,
    "last_status":   "no_update_received",
    "lock":          threading.Lock(),
}

_pipeline_state = {
    "running":          False,
    "stop_requested":   False,
    "current_video":    None,
    "videos_total":     0,
    "videos_done":      0,
    "videos_skipped":   0,
    # frame counters (updated live by both threads)
    "frames_read":      0,    # all frames read from video
    "frames_every_n":   0,    # frames that passed the every_n gate
    "frames_stage1":    0,    # frames that passed Stage 1 (M1+M2+M3)
    "frames_stage2":    0,    # frames that reached M4
    "crops_sent":       0,
    "queue_depth":      0,    # live queue depth
    "started_at":       None,
    "finished_at":      None,
    "last_run_at":      None,
    "shutdown":         False,
    "lock":             threading.Lock(),
}

_modules = {
    "m1": None, "m2": None, "m3": None,
    "m4": None, "m5a": None, "m6": None,
    "args": None,
}


# ═══════════════════════════════════════════════════════════════════════════════
#  DEVICE IDENTITY
# ═══════════════════════════════════════════════════════════════════════════════

def _get_cpu_serial() -> Optional[str]:
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('Serial'):
                    s = line.split(':')[1].strip()
                    if s and s != '0000000000000000':
                        return s
    except Exception:
        pass
    return None

def _get_mac() -> str:
    return ':'.join([f'{(uuid.getnode() >> i) & 0xff:02x}'
                     for i in range(0, 48, 8)][::-1])

def _get_device_id() -> str:
    serial = _get_cpu_serial()
    if serial:
        return f"rpi_{serial}"
    mac = _get_mac()
    if mac and mac != '00:00:00:00:00:00':
        return f"mac_{mac.replace(':', '')}"
    uid_file = Path('/home/pi/.device_uuid')
    if uid_file.exists():
        return uid_file.read_text().strip()
    did = f"uuid_{uuid.uuid4()}"
    try:
        uid_file.write_text(did)
    except Exception:
        pass
    return did

def _get_device_info() -> dict:
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "unknown"
    return {
        "device_id":      _get_device_id(),
        "device_name":    hostname,
        "device_type":    "raspberry_pi",
        "ip_address":     local_ip,
        "cpu_serial":     _get_cpu_serial(),
        "mac_address":    _get_mac(),
        "os_info":        f"{platform.system()} {platform.release()}",
        "python_version": platform.python_version(),
    }

DEVICE_INFO       = _get_device_info()
LAST_REGISTRATION = None


# ═══════════════════════════════════════════════════════════════════════════════
#  DEVICE REGISTRATION / HEARTBEAT
# ═══════════════════════════════════════════════════════════════════════════════

def register_device(server_url: str, status: str = "active") -> bool:
    global LAST_REGISTRATION
    try:
        resp = requests.post(
            f"{server_url}/devices/register",
            json={**DEVICE_INFO, "status": status,
                  "timestamp": datetime.now().isoformat(),
                  "capabilities": {"video_processing": True,
                                   "model_type": "yolov8n_int8",
                                   "ota_enabled": True}},
            timeout=10,
        )
        if resp.status_code == 200:
            LAST_REGISTRATION = datetime.now()
            log.info(f"✓ Device registered: {DEVICE_INFO['device_id']} → {status}")
            return True
        log.warning(f"Registration failed: {resp.status_code}")
    except Exception as e:
        log.error(f"Registration error: {e}")
    return False

def send_heartbeat(server_url: str) -> bool:
    try:
        resp = requests.post(
            f"{server_url}/devices/heartbeat",
            json={"device_id": DEVICE_INFO["device_id"],
                  "timestamp": datetime.now().isoformat(),
                  "status": "active",
                  "pipeline_running": _pipeline_state["running"]},
            timeout=5,
        )
        return resp.status_code == 200
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
#  FASTAPI APP  (port 8001)
# ═══════════════════════════════════════════════════════════════════════════════

app = FastAPI(title="Pi Pipeline Server")

def _pipeline_status_dict() -> dict:
    with _pipeline_state["lock"]:
        fn  = _pipeline_state["frames_every_n"]
        fs1 = _pipeline_state["frames_stage1"]
        fs2 = _pipeline_state["frames_stage2"]
        return {
            "running":           _pipeline_state["running"],
            "current_video":     _pipeline_state["current_video"],
            "videos_total":      _pipeline_state["videos_total"],
            "videos_done":       _pipeline_state["videos_done"],
            "videos_skipped":    _pipeline_state["videos_skipped"],
            "frames_read":       _pipeline_state["frames_read"],
            "frames_every_n":    fn,
            "frames_stage1_passed": fs1,
            "frames_stage2_reached": fs2,
            "stage1_pass_rate":  f"{fs1/max(fn,1)*100:.1f}%",
            "crops_sent":        _pipeline_state["crops_sent"],
            "queue_depth":       _pipeline_state["queue_depth"],
            "started_at":        _pipeline_state["started_at"],
            "finished_at":       _pipeline_state["finished_at"],
            "last_run_at":       _pipeline_state["last_run_at"],
            "video_folder":      str(VIDEO_FOLDER),
        }

@app.get("/health")
def health():
    return {"status": "running", "device": DEVICE_INFO,
            "current_model": _ota_state["current_model"],
            "pipeline": _pipeline_status_dict()}

@app.get("/device/info")
def device_info_ep():
    return {**DEVICE_INFO,
            "status": "active" if not _pipeline_state.get("shutdown") else "inactive",
            "last_registration": LAST_REGISTRATION.isoformat() if LAST_REGISTRATION else None,
            "pipeline_running": _pipeline_state["running"]}

@app.post("/ota/update")
async def ota_update(model: UploadFile = File(...), sha256: str = Form(...)):
    _log = logging.getLogger("OTA")
    if not model.filename.endswith(".tflite"):
        raise HTTPException(400, "Only .tflite files accepted")
    save_path  = OTA_MODEL_DIR / model.filename
    raw        = await model.read()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != sha256:
        _log.error("OTA hash mismatch")
        raise HTTPException(400, "SHA-256 mismatch — file corrupted in transit")
    save_path.write_bytes(raw)
    _log.info(f"OTA: received {model.filename} ({len(raw)/1e6:.2f}MB) — hash OK")
    with _ota_state["lock"]:
        _ota_state["pending_model"] = str(save_path)
        _ota_state["last_status"]   = f"pending_reload: {model.filename}"
    return JSONResponse({"status": "accepted", "model": model.filename,
                         "size_mb": round(len(raw)/1e6, 2), "sha256": actual_sha,
                         "note": "Consumer will hot-swap M4 on next frame."})

@app.get("/ota/status")
def ota_status():
    return {"current_model": _ota_state["current_model"],
            "pending_model": _ota_state["pending_model"],
            "last_update":   _ota_state["last_update"],
            "last_status":   _ota_state["last_status"],
            "available_models": [f.name for f in OTA_MODEL_DIR.glob("*.tflite")]}

@app.get("/pipeline/status")
def pipeline_status(): return _pipeline_status_dict()

@app.post("/pipeline/start")
def pipeline_start():
    with _pipeline_state["lock"]:
        if _pipeline_state["running"]:
            raise HTTPException(409, "Already running — POST /pipeline/stop first")
    videos = _get_video_list()
    if not videos:
        raise HTTPException(404, f"No videos in {VIDEO_FOLDER}")
    threading.Thread(target=_run_batch, args=(videos,), daemon=True).start()
    return {"status": "started", "video_folder": str(VIDEO_FOLDER),
            "videos_found": len(videos), "files": [v.name for v in videos]}

@app.post("/pipeline/stop")
def pipeline_stop():
    with _pipeline_state["lock"]:
        if not _pipeline_state["running"]:
            return {"status": "not_running"}
        _pipeline_state["stop_requested"] = True
    return {"status": "stop_requested"}

@app.get("/pipeline/videos")
def pipeline_videos():
    videos = _get_video_list()
    return {"video_folder": str(VIDEO_FOLDER), "count": len(videos),
            "files": [v.name for v in videos]}

import asyncio

@app.on_event("startup")
async def on_startup():
    _pipeline_state["shutdown"] = False
    register_device(LAPTOP_SERVER_URL, status="active")
    log.info("=" * 60)
    log.info(f"Edge Device ID: {DEVICE_INFO['device_id']}")
    log.info(f"Device Name:    {DEVICE_INFO['device_name']}")
    log.info(f"IP Address:     {DEVICE_INFO['ip_address']}")
    log.info("=" * 60)

@app.on_event("shutdown")
async def on_shutdown():
    _pipeline_state["shutdown"] = True
    register_device(LAPTOP_SERVER_URL, status="inactive")

@app.on_event("startup")
async def _start_heartbeat():
    async def _hb():
        while True:
            await asyncio.sleep(60)
            if not _pipeline_state.get("shutdown"):
                ok = send_heartbeat(LAPTOP_SERVER_URL)
                log.debug("Heartbeat " + ("ok" if ok else "failed"))
    asyncio.create_task(_hb())

def _start_api_server():
    log.info(f"API server starting on port {OTA_PORT}…")
    uvicorn.run(app, host="0.0.0.0", port=OTA_PORT, log_level="warning")


# ═══════════════════════════════════════════════════════════════════════════════
#  OTA HOT-SWAP  (called from consumer thread)
# ═══════════════════════════════════════════════════════════════════════════════

def _check_and_apply_ota(m4: VehicleDetector) -> VehicleDetector:
    with _ota_state["lock"]:
        pending = _ota_state["pending_model"]
        if pending is None:
            return m4
        new_path = pending
        _ota_state["pending_model"] = None
    log.info(f"[OTA] Hot-swapping M4 → {new_path}")
    try:
        new_cfg = m4.config if hasattr(m4, "config") else M4Config()
        new_m4  = VehicleDetector(model_path=new_path, config=new_cfg)
        with _ota_state["lock"]:
            _ota_state["current_model"] = new_path
            _ota_state["last_update"]   = datetime.now().isoformat()
            _ota_state["last_status"]   = f"loaded: {Path(new_path).name}"
        _modules["m4"] = new_m4
        log.info(f"[OTA] ✓ Hot-swap complete — {Path(new_path).name}")
        return new_m4
    except Exception as e:
        log.error(f"[OTA] Hot-swap FAILED: {e} — keeping old model")
        with _ota_state["lock"]:
            _ota_state["last_status"] = f"reload_failed: {e}"
        return m4


# ═══════════════════════════════════════════════════════════════════════════════
#  PRODUCER THREAD  — Frame reader + Stage 1 (M1 + M2 + M3)
# ═══════════════════════════════════════════════════════════════════════════════

class _ProducerThread(threading.Thread):
    """
    Reads frames, applies every_n gate, runs M1→M2→M3.
    Frames that pass all three are put into out_queue.
    Blocks on queue.put() when full → backpressure (no OOM on Pi).

    BUG FIXED vs original:
      pipeline_id now increments ONLY when a frame passes Stage 1,
      not at the every_n gate. This gives the correct Stage-1-passed count.
    """

    def __init__(self, video_path: Path, out_queue: queue.Queue,
                 stats: dict, args):
        super().__init__(daemon=True, name="Producer")
        self.video_path = video_path
        self.out_queue  = out_queue
        self.stats      = stats
        self.args       = args
        self.stopped    = False

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {video_path}")
        self.cap      = cap
        self.fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.orig_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.work_w   = int(self.orig_w * args.scale)
        self.work_h   = int(self.orig_h * args.scale)

        # Stage 1 module references
        self.m1 = _modules["m1"]
        self.m2 = _modules["m2"]
        self.m3 = _modules["m3"]

        log.info(
            f"Producer: {video_path.name}  "
            f"{self.orig_w}x{self.orig_h} @ {self.fps:.1f}fps  "
            f"{self.n_frames} total frames  "
            f"every_n={args.every_n} → ~{self.n_frames//args.every_n} candidate frames"
        )

    def run(self):
        args        = self.args
        s           = self.stats
        frame_id    = 0    # every frame read from video (1-based)
        every_n_id  = 0    # frames that passed the every_n gate
        pipeline_id = 0    # frames that passed Stage 1  ← was wrong before

        # Running averages for timing (EMA, α=0.1)
        ema_read = ema_m1 = ema_m2 = ema_m3 = 0.0
        alpha    = 0.1

        log.info("Producer: started — reading + Stage 1 (M1+M2+M3)")
        t_producer_start = time.perf_counter()

        try:
            while not self.stopped:
                with _pipeline_state["lock"]:
                    if _pipeline_state["stop_requested"]:
                        log.info("Producer: stop requested")
                        break

                # ── Frame read ────────────────────────────────────────────────
                t0 = time.perf_counter()
                ret, frame = self.cap.read()
                t_read = (time.perf_counter() - t0) * 1000

                if not ret:
                    log.info("Producer: end of video")
                    break

                frame_id += 1
                s["frames_read"] += 1
                with _pipeline_state["lock"]:
                    _pipeline_state["frames_read"] += 1

                ema_read = alpha * t_read + (1 - alpha) * ema_read

                # ── every_n gate ──────────────────────────────────────────────
                if frame_id % args.every_n != 0:
                    continue
                every_n_id += 1
                s["frames_every_n"] += 1
                with _pipeline_state["lock"]:
                    _pipeline_state["frames_every_n"] += 1

                # ── Resize ────────────────────────────────────────────────────
                if args.scale != 1.0:
                    frame = cv2.resize(frame, (self.work_w, self.work_h))

                # ── M1 ────────────────────────────────────────────────────────
                t0  = time.perf_counter()
                m1r = self.m1.process(frame)
                t_m1 = (time.perf_counter() - t0) * 1000
                ema_m1 = alpha * t_m1 + (1 - alpha) * ema_m1
                s["m1_total_ms"] += t_m1
                s["m1_count"]    += 1

                if m1r.skip:
                    s["m1_skip"] += 1
                    continue

                # ── M2 ────────────────────────────────────────────────────────
                t0  = time.perf_counter()
                m2r = self.m2.process(frame, frame_id=every_n_id)
                t_m2 = (time.perf_counter() - t0) * 1000
                ema_m2 = alpha * t_m2 + (1 - alpha) * ema_m2
                s["m2_total_ms"] += t_m2
                s["m2_count"]    += 1

                if m2r.skip:
                    s["m2_skip"] += 1
                    continue

                # ── M3 ────────────────────────────────────────────────────────
                t0  = time.perf_counter()
                m3r = self.m3.process(every_n_id, m1r.fg_mask, m1r.fg_ratio)
                t_m3 = (time.perf_counter() - t0) * 1000
                ema_m3 = alpha * t_m3 + (1 - alpha) * ema_m3
                s["m3_total_ms"] += t_m3
                s["m3_count"]    += 1

                if m3r.skip:
                    s["m3_skip"] += 1
                    continue

                # ── Passed Stage 1 ────────────────────────────────────────────
                pipeline_id += 1          # ← correct: only counts S1-passed frames
                s["stage1_passed"] += 1
                with _pipeline_state["lock"]:
                    _pipeline_state["frames_stage1"] += 1

                t_total = t_read + t_m1 + t_m2 + t_m3

                item = _QueueItem(
                    frame_id    = frame_id,
                    pipeline_id = pipeline_id,
                    frame       = frame.copy(),
                    m1r=m1r, m2r=m2r, m3r=m3r,
                    t_read_ms   = t_read,
                    t_m1_ms     = t_m1,
                    t_m2_ms     = t_m2,
                    t_m3_ms     = t_m3,
                    t_queued_ms = t_total,
                )

                # Blocks when queue is full → backpressure (prevents Pi OOM)
                self.out_queue.put(item)

                with _pipeline_state["lock"]:
                    _pipeline_state["queue_depth"] = self.out_queue.qsize()

                # ── Progress log every 50 Stage-1-passed frames ───────────────
                if pipeline_id % 50 == 0:
                    log.info(
                        f"Producer | frame {frame_id}/{self.n_frames} "
                        f"every_n={every_n_id} s1_passed={pipeline_id} "
                        f"q={self.out_queue.qsize()} | "
                        f"read≈{ema_read:.1f}ms "
                        f"M1≈{ema_m1:.1f}ms M2≈{ema_m2:.1f}ms M3≈{ema_m3:.1f}ms"
                    )

        finally:
            self.cap.release()
            elapsed = (time.perf_counter() - t_producer_start)
            log.info(
                f"Producer done | "
                f"read={frame_id} every_n={every_n_id} s1_passed={pipeline_id} | "
                f"m1_skip={s['m1_skip']} m2_skip={s['m2_skip']} m3_skip={s['m3_skip']} | "
                f"M1_avg={s['m1_total_ms']/max(s['m1_count'],1):.1f}ms "
                f"M2_avg={s['m2_total_ms']/max(s['m2_count'],1):.1f}ms "
                f"M3_avg={s['m3_total_ms']/max(s['m3_count'],1):.1f}ms | "
                f"elapsed={elapsed:.1f}s"
            )

    def stop(self):
        self.stopped = True


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSUMER THREAD  — Stage 2 (M4 + M5a + M6)
# ═══════════════════════════════════════════════════════════════════════════════

class _ConsumerThread(threading.Thread):
    """
    Pulls _QueueItem from the queue and runs Stage 2.
    Calls task_done() in a finally block — guaranteed even on exception.
    OTA hot-swap applied between items.

    IMPORTANT: task_done() MUST match every queue.get() exactly once.
    Missing it causes queue.join() to hang forever. The try/finally
    in run() guarantees this even if M4/M5a/M6 raises an exception.
    """

    def __init__(self, in_queue: queue.Queue, stats: dict):
        super().__init__(daemon=True, name="Consumer")
        self.in_queue = in_queue
        self.stats    = stats
        self.stopped  = False
        self.m4       = _modules["m4"]    # consumer owns its m4 reference

        log.info("Consumer: Stage 2 modules ready")

    def run(self):
        s = self.stats

        # Running averages (EMA)
        ema_wait = ema_m4 = ema_m5 = ema_m6 = ema_total = 0.0
        alpha    = 0.1

        log.info("Consumer: started — Stage 2 (M4+M5a+M6)")
        t_consumer_start = time.perf_counter()
        items_processed  = 0

        try:
            while not self.stopped:
                # ── Get from queue ────────────────────────────────────────────
                t_wait0 = time.perf_counter()
                try:
                    item: _QueueItem = self.in_queue.get(timeout=1.0)
                except queue.Empty:
                    # Producer may still be running — keep waiting
                    continue
                t_wait = (time.perf_counter() - t_wait0) * 1000
                ema_wait = alpha * t_wait + (1 - alpha) * ema_wait

                try:
                    t_frame_start = time.perf_counter()

                    # ── OTA hot-swap between frames ───────────────────────────
                    self.m4    = _check_and_apply_ota(self.m4)
                    _modules["m4"] = self.m4

                    # ── M4: Vehicle detection (YOLO) ──────────────────────────
                    t0  = time.perf_counter()
                    m4r = self.m4.process(item.frame, item.pipeline_id)
                    t_m4 = (time.perf_counter() - t0) * 1000
                    ema_m4 = alpha * t_m4 + (1 - alpha) * ema_m4
                    s["m4_total_ms"] += t_m4
                    s["m4_count"]    += 1

                    if m4r.skip:
                        s["m4_skip"] += 1
                        log.debug(
                            f"Consumer | pid={item.pipeline_id} "
                            f"M4={t_m4:.0f}ms → SKIP (no detections)"
                        )
                        continue   # task_done() in finally

                    with _pipeline_state["lock"]:
                        _pipeline_state["frames_stage2"] += 1

                    s["m4_detections"] += len(m4r.all_dets) \
                        if hasattr(m4r, "all_dets") else 0

                    # ── M5a: Pillion counter ──────────────────────────────────
                    t0   = time.perf_counter()
                    m5ar = _modules["m5a"].process(
                        item.pipeline_id, item.frame,
                        m4r.motorcycles, m4r.persons
                    )
                    t_m5 = (time.perf_counter() - t0) * 1000
                    ema_m5 = alpha * t_m5 + (1 - alpha) * ema_m5
                    s["m5_total_ms"] += t_m5
                    s["m5_count"]    += 1
                    s["m5a_processed"] += 1

                    if m5ar.has_helmet_candidate:
                        s["m5a_helmet_candidates"] += 1
                    if m5ar.has_triple_riding:
                        s["m5a_triple_riding"] += 1

                    # ── M6: Crop and transmit ─────────────────────────────────
                    t0  = time.perf_counter()
                    m6r = _modules["m6"].process(
                        item.frame, item.pipeline_id, m5ar
                    )
                    t_m6 = (time.perf_counter() - t0) * 1000
                    ema_m6 = alpha * t_m6 + (1 - alpha) * ema_m6
                    s["m6_total_ms"] += t_m6
                    s["m6_count"]    += 1

                    sent = m6r.sent_count if hasattr(m6r, "sent_count") else 0
                    s["m6_crops_sent"] += sent

                    t_total_frame = (time.perf_counter() - t_frame_start) * 1000
                    ema_total = alpha * t_total_frame + (1 - alpha) * ema_total

                    with _pipeline_state["lock"]:
                        _pipeline_state["frames_processed"] += 1
                        _pipeline_state["crops_sent"]       += sent
                        _pipeline_state["queue_depth"]       = self.in_queue.qsize()

                    items_processed += 1

                    # ── Per-frame timing log (every 10 Stage-2 frames) ────────
                    if items_processed % 10 == 0:
                        log.info(
                            f"Consumer | pid={item.pipeline_id} "
                            f"q_wait={t_wait:.0f}ms "
                            f"M4={t_m4:.0f}ms "
                            f"M5a={t_m5:.0f}ms "
                            f"M6={t_m6:.0f}ms "
                            f"frame_total={t_total_frame:.0f}ms | "
                            f"EMA: M4≈{ema_m4:.0f}ms M5a≈{ema_m5:.0f}ms "
                            f"M6≈{ema_m6:.0f}ms total≈{ema_total:.0f}ms | "
                            f"queue={self.in_queue.qsize()} "
                            f"crops_sent={s['m6_crops_sent']}"
                        )

                    # ── Per-frame timing breakdown at DEBUG ───────────────────
                    log.debug(
                        f"Consumer | pid={item.pipeline_id} fid={item.frame_id} | "
                        f"[Producer] read={item.t_read_ms:.1f}ms "
                        f"M1={item.t_m1_ms:.1f}ms "
                        f"M2={item.t_m2_ms:.1f}ms "
                        f"M3={item.t_m3_ms:.1f}ms "
                        f"s1_total={item.t_queued_ms:.1f}ms | "
                        f"[Consumer] q_wait={t_wait:.1f}ms "
                        f"M4={t_m4:.1f}ms M5a={t_m5:.1f}ms M6={t_m6:.1f}ms "
                        f"s2_total={t_total_frame:.1f}ms"
                    )

                except Exception as e:
                    log.error(
                        f"Consumer error on pid={item.pipeline_id}: {e}",
                        exc_info=True
                    )
                finally:
                    # ── CRITICAL: always call task_done() ─────────────────────
                    # Missing this causes queue.join() to hang forever
                    self.in_queue.task_done()

        finally:
            elapsed = time.perf_counter() - t_consumer_start
            log.info(
                f"Consumer done | processed={items_processed} | "
                f"m4_skip={s['m4_skip']} m4_dets={s['m4_detections']} | "
                f"helmet_candidates={s['m5a_helmet_candidates']} "
                f"triple={s['m5a_triple_riding']} | "
                f"crops_sent={s['m6_crops_sent']} | "
                f"M4_avg={s['m4_total_ms']/max(s['m4_count'],1):.0f}ms "
                f"M5a_avg={s['m5_total_ms']/max(s['m5_count'],1):.0f}ms "
                f"M6_avg={s['m6_total_ms']/max(s['m6_count'],1):.0f}ms | "
                f"elapsed={elapsed:.1f}s"
            )

    def stop(self):
        self.stopped = True


# ═══════════════════════════════════════════════════════════════════════════════
#  SINGLE VIDEO PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════════

def _process_video(video_path: Path, queue_size: int) -> dict:
    """
    Runs one video through the producer-consumer buffered pipeline.
    Returns a summary dict.
    """
    # Per-video shared stats dict (not shared with _pipeline_state — avoids lock contention)
    stats = {
        "frames_read":      0,
        "frames_every_n":   0,
        "m1_count": 0, "m1_skip": 0, "m1_total_ms": 0.0,
        "m2_count": 0, "m2_skip": 0, "m2_total_ms": 0.0,
        "m3_count": 0, "m3_skip": 0, "m3_total_ms": 0.0,
        "stage1_passed":         0,
        "m4_count": 0, "m4_skip": 0, "m4_total_ms": 0.0, "m4_detections": 0,
        "m5_count": 0, "m5_total_ms": 0.0,
        "m5a_processed": 0, "m5a_helmet_candidates": 0, "m5a_triple_riding": 0,
        "m6_count": 0, "m6_total_ms": 0.0, "m6_crops_sent": 0,
    }

    # Bounded queue — maxsize prevents RAM exhaustion on Pi
    stage2_q = queue.Queue(maxsize=queue_size)
    log.info(f"▶ {video_path.name} | queue_size={queue_size} (~{queue_size*0.3:.1f}MB buffer)")

    try:
        producer = _ProducerThread(video_path, stage2_q, stats,
                                   _modules["args"])
    except ValueError as e:
        log.error(str(e))
        return {"video": video_path.name, "error": "cannot_open"}

    consumer = _ConsumerThread(stage2_q, stats)

    t0 = time.perf_counter()
    producer.start()
    consumer.start()

    try:
        producer.join()   # wait for all frames to be read + Stage 1 filtered
        log.info(f"  Producer finished — draining queue ({stage2_q.qsize()} items)")
        stage2_q.join()   # wait for consumer to call task_done() on every item
    except KeyboardInterrupt:
        log.info("  Interrupted — stopping threads")
        producer.stop()
    finally:
        consumer.stop()
        consumer.join(timeout=5.0)

    elapsed = time.perf_counter() - t0

    # ── Final per-video summary ───────────────────────────────────────────────
    fn  = stats["frames_every_n"]
    fs1 = stats["stage1_passed"]
    fs2 = stats["m4_count"] - stats["m4_skip"]
    log.info("─" * 70)
    log.info(f"VIDEO SUMMARY: {video_path.name}  ({elapsed:.1f}s)")
    log.info(f"  Frames  : read={stats['frames_read']}  "
             f"every_n={fn}  "
             f"stage1_passed={fs1} ({fs1/max(fn,1)*100:.1f}%)  "
             f"stage2_reached={fs2}")
    log.info(f"  Stage 1 : M1_skip={stats['m1_skip']}  "
             f"M2_skip={stats['m2_skip']}  "
             f"M3_skip={stats['m3_skip']}")
    log.info(f"  Timing  : M1_avg={stats['m1_total_ms']/max(stats['m1_count'],1):.1f}ms  "
             f"M2_avg={stats['m2_total_ms']/max(stats['m2_count'],1):.1f}ms  "
             f"M3_avg={stats['m3_total_ms']/max(stats['m3_count'],1):.1f}ms")
    log.info(f"  Stage 2 : M4_skip={stats['m4_skip']}  "
             f"M4_dets={stats['m4_detections']}  "
             f"helmet_candidates={stats['m5a_helmet_candidates']}  "
             f"triple={stats['m5a_triple_riding']}  "
             f"crops_sent={stats['m6_crops_sent']}")
    log.info(f"  Timing  : M4_avg={stats['m4_total_ms']/max(stats['m4_count'],1):.0f}ms  "
             f"M5a_avg={stats['m5_total_ms']/max(stats['m5_count'],1):.0f}ms  "
             f"M6_avg={stats['m6_total_ms']/max(stats['m6_count'],1):.0f}ms")
    log.info("─" * 70)

    return {
        "video":       video_path.name,
        "elapsed_s":   round(elapsed, 2),
        "frames_read": stats["frames_read"],
        "stage1_passed": fs1,
        "crops_sent":  stats["m6_crops_sent"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  VIDEO FOLDER SCANNER
# ═══════════════════════════════════════════════════════════════════════════════

def _get_video_list() -> List[Path]:
    if not VIDEO_FOLDER.exists():
        log.warning(f"VIDEO_FOLDER does not exist: {VIDEO_FOLDER}")
        return []
    videos = sorted([f for f in VIDEO_FOLDER.iterdir()
                     if f.suffix.lower() in VIDEO_EXTS])
    log.info(f"Found {len(videos)} video(s) in {VIDEO_FOLDER}")
    return videos


# ═══════════════════════════════════════════════════════════════════════════════
#  BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

def _run_batch(videos: List[Path]):
    args = _modules["args"]
    with _pipeline_state["lock"]:
        _pipeline_state.update({
            "running": True, "stop_requested": False,
            "videos_total": len(videos), "videos_done": 0, "videos_skipped": 0,
            "frames_read": 0, "frames_every_n": 0,
            "frames_stage1": 0, "frames_stage2": 0,
            "frames_processed": 0, "crops_sent": 0, "queue_depth": 0,
            "started_at": datetime.now().isoformat(), "finished_at": None,
        })

    log.info(f"Batch start — {len(videos)} video(s)")
    t_batch = time.perf_counter()

    for video_path in videos:
        with _pipeline_state["lock"]:
            if _pipeline_state["stop_requested"]:
                log.info("Batch cancelled")
                break
            _pipeline_state["current_video"] = video_path.name

        result = _process_video(video_path, queue_size=args.queue_size)

        with _pipeline_state["lock"]:
            if "error" in result:
                _pipeline_state["videos_skipped"] += 1
            else:
                _pipeline_state["videos_done"] += 1

    elapsed = time.perf_counter() - t_batch
    now = datetime.now().isoformat()
    with _pipeline_state["lock"]:
        _pipeline_state.update({
            "running": False, "stop_requested": False,
            "current_video": None, "queue_depth": 0,
            "finished_at": now, "last_run_at": now,
        })
    log.info(f"Batch complete in {elapsed:.1f}s | "
             f"done={_pipeline_state['videos_done']} | "
             f"skipped={_pipeline_state['videos_skipped']} | "
             f"total_crops_sent={_pipeline_state['crops_sent']}")


# ═══════════════════════════════════════════════════════════════════════════════
#  SCHEDULER
# ═══════════════════════════════════════════════════════════════════════════════

def _scheduler_loop():
    log.info(f"Scheduler: every {SCHEDULE_HOURS}h from {VIDEO_FOLDER}")
    while True:
        time.sleep(SCHEDULE_HOURS * 3600)
        with _pipeline_state["lock"]:
            running = _pipeline_state["running"]
        if running:
            log.info("Scheduler: skipping — pipeline already running")
            continue
        videos = _get_video_list()
        if not videos:
            log.info("Scheduler: no videos found")
            continue
        log.info(f"Scheduler: triggering batch — {len(videos)} video(s)")
        threading.Thread(target=_run_batch, args=(videos,), daemon=True).start()


# ═══════════════════════════════════════════════════════════════════════════════
#  MODULE INIT
# ═══════════════════════════════════════════════════════════════════════════════

def _init_modules(args):
    _modules["args"] = args
    _modules["m1"]   = FrameLevelFilter(M1Config())
    _modules["m2"]   = BlurDetector(M2Config())
    _modules["m3"]   = ForegroundGate(M3Config())

    m4_cfg = M4Config()
    m4_cfg.SAVE_DETECTED_FRAMES = args.save_m4_frames
    _modules["m4"] = VehicleDetector(model_path=args.model, config=m4_cfg)

    with _ota_state["lock"]:
        _ota_state["current_model"] = str(Path(args.model).resolve())

    _modules["m5a"] = PillionCounter(M5aConfig())

    m6_cfg               = M6Config()
    m6_cfg.SERVER_URL    = f"{args.server}/detect"
    m6_cfg.STUB_TRANSMIT = args.stub_transmit
    m6_cfg.ALWAYS_SAVE_LOCAL = True
    _modules["m6"]   = CropTransmit(m6_cfg)

    log.info("All modules initialised")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Traffic Violation Detection — Stage 1+2 (buffered producer-consumer)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Modes:
  --mode trigger   REST endpoint triggers processing
  --mode schedule  Auto-runs every {SCHEDULE_HOURS}h from VIDEO_FOLDER ({VIDEO_FOLDER})
  --mode once      Process --video file and exit

Queue size (--queue-size):
  Pi 1GB → 2   Pi 2GB → 5 (default)   Pi 4GB+ → 10

Examples:
  python3 pipeline_stage2_new.py --mode once --video ./day-1.mp4 --model yolov8n_coco_320_int8.tflite
  python3 pipeline_stage2_new.py --mode trigger --model yolov8n_coco_320_int8.tflite
  python3 pipeline_stage2_new.py --mode schedule --model yolov8n_coco_320_int8.tflite --queue-size 3
        """
    )
    parser.add_argument("--mode",           default="once",
                        choices=["once", "trigger", "schedule"])
    parser.add_argument("--video",          default=None)
    parser.add_argument("--model",          required=True)
    parser.add_argument("--server",         default=LAPTOP_SERVER_URL)
    parser.add_argument("--queue-size",     type=int, default=5, dest="queue_size",
                        help="Bounded queue between Stage 1 and Stage 2 (default 5)")
    parser.add_argument("--scale",          type=float, default=0.5)
    parser.add_argument("--every_n",        type=int,   default=2)
    parser.add_argument("--log",            default="INFO")
    parser.add_argument("--stub_transmit",  action="store_true")
    parser.add_argument("--save_m4_frames", action="store_true")
    parser.add_argument("--save_debug_crops", action="store_true")
    parser.add_argument("--display",        action="store_true")
    parser.add_argument("--no_display",     action="store_true")

    args = parser.parse_args()
    if args.no_display:
        args.display = False

    setup_logging(args.log)

    if not Path(args.model).exists():
        log.error(f"Model not found: {args.model}")
        exit(1)

    # Start API server in background
    threading.Thread(target=_start_api_server, daemon=True).start()
    time.sleep(0.6)
    log.info(f"API ready → http://0.0.0.0:{OTA_PORT}")

    if args.mode == "once":
        log.info("Mode: ONCE")
        if not args.video:
            log.error("--mode once requires --video"); exit(1)
        vp = Path(args.video)
        if not vp.exists():
            log.error(f"Video not found: {vp}"); exit(1)
        _init_modules(args)
        try:
            _run_batch([vp])
        except KeyboardInterrupt:
            log.info("Stopped.")

    elif args.mode == "trigger":
        log.info("Mode: TRIGGER")
        log.info(f"  Video folder : {VIDEO_FOLDER}")
        log.info(f"  Queue size   : {args.queue_size} frames")
        log.info(f"  Trigger via  : POST http://0.0.0.0:{OTA_PORT}/pipeline/start")
        _init_modules(args)
        try:
            while True: time.sleep(5)
        except KeyboardInterrupt:
            log.info("Stopped.")

    elif args.mode == "schedule":
        log.info(f"Mode: SCHEDULE  (every {SCHEDULE_HOURS}h from {VIDEO_FOLDER})")
        log.info(f"  Queue size   : {args.queue_size} frames")
        _init_modules(args)
        videos = _get_video_list()
        if videos:
            log.info(f"Initial run: {len(videos)} video(s)")
            threading.Thread(target=_run_batch, args=(videos,), daemon=True).start()
        threading.Thread(target=_scheduler_loop, daemon=True).start()
        try:
            while True: time.sleep(5)
        except KeyboardInterrupt:
            log.info("Stopped.")
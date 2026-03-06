"""
Vehicle Detection & License Plate Recognition Service
Highway-camera-style real-time detection pipeline:
  1. YOLOv8 for vehicle detection (bounding boxes)
  2. Image processing for license plate localization
  3. EasyOCR for plate text recognition
  4. Centroid tracker to track vehicles & avoid duplicates
"""
import cv2
import numpy as np
import threading
import time
import os
import re
import logging
import math
import gc
from collections import OrderedDict
from datetime import date, datetime
from pathlib import Path
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

# Global state for camera streaming
camera_lock = threading.Lock()
camera_instance = None
detection_running = False
detection_thread = None

# Lazy-loaded OpenALPR instance
alpr_instance = None
alpr_lock = threading.Lock()

def get_alpr():
    """Lazy load OpenALPR engine."""
    global alpr_instance
    with alpr_lock:
        if alpr_instance is None:
            try:
                # Direct python to load DLLs from this directory first
                openalpr_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'openalpr-win', 'openalpr_64')
                
                if hasattr(os, 'add_dll_directory'):
                    os.add_dll_directory(openalpr_dir)
                os.environ['PATH'] = openalpr_dir + os.pathsep + os.environ.get('PATH', '')
                
                from openalpr import Alpr
                
                conf_path = os.path.join(openalpr_dir, 'openalpr.conf')
                runtime_path = os.path.join(openalpr_dir, 'runtime_data')
                
                # By default, openalpr-win precompiled package has a US dataset plus maybe EU, but not IN
                # We can request "in" if it exists, otherwise fallback to "us" if it fails.
                # Here we attempt whatever dataset you prefer, but "us" is safely in the defaults folder usually.
                alpr_instance = Alpr("us", conf_path, runtime_path)
                
                if not getattr(alpr_instance, 'loaded', getattr(alpr_instance, 'is_loaded', lambda: False)()):
                    logger.error("Error loading OpenALPR engine. Checking for DLL footprint failure.")
                    alpr_instance = None
                else:
                    alpr_instance.set_top_n(3)
                    logger.info("OpenALPR loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load OpenALPR: {e}")
                alpr_instance = False # Indicate failed load so we don't spam errors
        return alpr_instance if alpr_instance is not False else None
# ============================================================
# Centroid Tracker - tracks vehicles across frames
# ============================================================

class CentroidTracker:
    """
    Simple centroid-based object tracker.
    Assigns IDs to detected objects and tracks them across frames
    to avoid counting the same vehicle multiple times.
    """

    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_object_id = 1
        self.objects = OrderedDict()       # id -> centroid
        self.bboxes = OrderedDict()        # id -> bbox
        self.disappeared = OrderedDict()   # id -> frame count since last seen
        self.plate_texts = OrderedDict()   # id -> best plate text
        self.plate_confs = OrderedDict()   # id -> best plate confidence
        self.saved = OrderedDict()         # id -> True if saved to DB
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid, bbox, plate_text=None, plate_conf=0.0):
        object_id = self.next_object_id
        self.objects[object_id] = centroid
        self.bboxes[object_id] = bbox
        self.disappeared[object_id] = 0
        self.plate_texts[object_id] = plate_text
        self.plate_confs[object_id] = plate_conf
        self.saved[object_id] = False
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        del self.plate_texts[object_id]
        del self.plate_confs[object_id]
        del self.saved[object_id]

    def update(self, detections):
        """
        Update tracker with new detections.
        detections: list of dicts with 'bbox', 'plate', 'plate_conf'
        Returns: dict of object_id -> detection info
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)
            return self.objects

        input_centroids = []
        input_bboxes = []
        input_plates = []
        input_confs = []

        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            input_centroids.append((cx, cy))
            input_bboxes.append(det['bbox'])
            input_plates.append(det.get('plate'))
            input_confs.append(det.get('plate_conf', 0.0))

        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i], input_bboxes[i],
                              input_plates[i], input_confs[i])
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())

            # Compute distance matrix
            D = np.zeros((len(object_centroids), len(input_centroids)))
            for i, oc in enumerate(object_centroids):
                for j, ic in enumerate(input_centroids):
                    D[i, j] = math.dist(oc, ic)

            # Match closest pairs
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue

                obj_id = object_ids[row]
                self.objects[obj_id] = input_centroids[col]
                self.bboxes[obj_id] = input_bboxes[col]
                self.disappeared[obj_id] = 0

                # Update plate if new one is better
                if input_plates[col] and input_confs[col] > self.plate_confs.get(obj_id, 0):
                    self.plate_texts[obj_id] = input_plates[col]
                    self.plate_confs[obj_id] = input_confs[col]

                used_rows.add(row)
                used_cols.add(col)

            # Handle unmatched existing objects
            unused_rows = set(range(len(object_centroids))) - used_rows
            for row in unused_rows:
                obj_id = object_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

            # Register new objects
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self.register(input_centroids[col], input_bboxes[col],
                              input_plates[col], input_confs[col])

        return self.objects


# Global tracker instance
vehicle_tracker = CentroidTracker(max_disappeared=40, max_distance=100)
total_vehicles_counted = 0
detected_plates_log = []  # Recent plate detections for HUD display


# ============================================================
# Camera Stream Manager
# ============================================================

class CameraStream:
    """Manages camera/CCTV stream."""

    def __init__(self, source=0):
        self.source = source
        self.cap = None
        self.frame = None
        self.running = False
        self.lock = threading.Lock()
        self.last_frame_time = 0

    def start(self):
        """Start camera capture."""
        try:
            src = int(self.source)
        except (ValueError, TypeError):
            src = self.source

        if isinstance(src, str) and src.startswith('rtsp://'):
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
            self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        else:
            self.cap = cv2.VideoCapture(src)

        if not self.cap.isOpened():
            logger.error(f"Cannot open camera source: {src}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True

        t = threading.Thread(target=self._capture_loop, daemon=True)
        t.start()
        logger.info(f"Camera started: {src}")
        return True

    def _capture_loop(self):
        """Continuously capture frames."""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        # cap.read() returns a new array, no need to copy which saves memory
                        self.frame = frame
                        self.last_frame_time = time.time()
                else:
                    time.sleep(0.1)
                    try:
                        src = int(self.source)
                    except (ValueError, TypeError):
                        src = self.source
                    self.cap.release()
                    if isinstance(src, str) and src.startswith('rtsp://'):
                        self.cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
                    else:
                        self.cap = cv2.VideoCapture(src)
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(0.5)

    def get_frame(self):
        """Get the latest frame."""
        with self.lock:
            if self.frame is not None:
                return self.frame.copy()
        return None

    def stop(self):
        """Stop camera capture."""
        self.running = False
        if self.cap:
            self.cap.release()
        logger.info("Camera stopped")

    def is_active(self):
        """Check if camera is actively capturing."""
        return self.running and (time.time() - self.last_frame_time < 5)


def get_camera(source=0):
    """Get or create camera instance."""
    global camera_instance
    with camera_lock:
        if camera_instance is None or not camera_instance.is_active():
            if camera_instance:
                camera_instance.stop()
            camera_instance = CameraStream(source)
            camera_instance.start()
        return camera_instance


def stop_camera():
    """Stop the global camera instance."""
    global camera_instance
    with camera_lock:
        if camera_instance:
            camera_instance.stop()
            camera_instance = None


def generate_video_feed(source=0):
    """Generator for streaming video frames as MJPEG."""
    cam = get_camera(source)
    while True:
        frame = cam.get_frame()
        if frame is not None:
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)


def detect_vehicles_in_frame(frame):
    """
    Detection pipeline using OpenALPR.
    OpenALPR detects the plate directly, so we infer "vehicle" presence
    around the plate region rather than doing full YOLO.
    """
    alpr = get_alpr()
    detections = []
    
    if alpr is None:
        return detections

    try:
        # OpenALPR Python bindings take raw image bytes or file paths
        # We need to encode the OpenCV frame to jpeg bytes first
        ret, enc = cv2.imencode('.jpg', frame)
        if not ret:
            return detections
            
        results = alpr.recognize_array(enc.tobytes())
        
        for plate_result in results.get('results', []):
            if not plate_result.get('candidates'):
                continue
                
            best_plate = plate_result['candidates'][0]
            plate_text = best_plate['plate']
            conf = best_plate['confidence'] / 100.0  # Normalize to 0.0 - 1.0
            
            # Extract plate coordinates
            coords = plate_result['coordinates']
            x_min = min(c['x'] for c in coords)
            y_min = min(c['y'] for c in coords)
            x_max = max(c['x'] for c in coords)
            y_max = max(c['y'] for c in coords)
            
            pw = x_max - x_min
            ph = y_max - y_min
            plate_bbox = (x_min, y_min, pw, ph)
            
            # Estimate a vehicle bounding box around the plate
            # Typically a car is about 3-5x wider than a license plate and much taller
            vw = int(pw * 4)
            vh = int(ph * 6)
            
            # Center roughly around plate
            vx1 = max(0, x_min - int(pw * 1.5))
            vy1 = max(0, y_min - int(ph * 4))
            vx2 = min(frame.shape[1], vx1 + vw)
            vy2 = min(frame.shape[0], vy1 + vh)
            
            vehicle_crop = frame[vy1:vy2, vx1:vx2]
            plate_img = frame[y_min:y_max, x_min:x_max]
            
            detections.append({
                'type': 'car', # OpenALPR only reads plates, so we guess 'car'
                'plate': plate_text,
                'plate_conf': conf,
                'confidence': conf, # using plate conf as overall confidence
                'bbox': (vx1, vy1, vx2, vy2),
                'vehicle_crop': vehicle_crop,
                'plate_img': plate_img,
                'plate_bbox': plate_bbox,
            })
            
    except Exception as e:
        logger.error(f"OpenALPR detection error: {e}")

    return detections


# ============================================================
# Highway Camera Style HUD Drawing
# ============================================================

def draw_highway_hud(frame, detections, tracker):
    """
    Draw highway-camera-style HUD overlay:
    - Vehicle bounding boxes with IDs
    - License plate boxes (highlighted)
    - Plate text readout
    - Vehicle count & timestamp
    - Detection zone lines
    """
    global total_vehicles_counted, detected_plates_log

    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # --- Color palette ---
    COLOR_VEHICLE = (0, 255, 100)      # Green for vehicles
    COLOR_PLATE = (0, 200, 255)        # Yellow/amber for plates
    COLOR_PLATE_BG = (0, 50, 120)      # Dark amber background
    COLOR_TEXT = (255, 255, 255)        # White text
    COLOR_HUD_BG = (30, 30, 30)        # Dark HUD background
    COLOR_MOTORCYCLE = (255, 200, 0)   # Cyan for motorcycles
    COLOR_TRUCK = (0, 140, 255)        # Orange for trucks
    COLOR_BUS = (255, 80, 80)          # Blue for buses
    COLOR_ID = (100, 255, 255)         # Yellow for IDs

    type_colors = {
        'car': COLOR_VEHICLE,
        'motorcycle': COLOR_MOTORCYCLE,
        'truck': COLOR_TRUCK,
        'bus': COLOR_BUS,
        'auto': (255, 255, 0),
        'unknown': (180, 180, 180),
    }

    # --- Update tracker ---
    tracker_dets = []
    for det in detections:
        tracker_dets.append({
            'bbox': det['bbox'],
            'plate': det.get('plate'),
            'plate_conf': det.get('plate_conf', 0.0),
        })

    tracker.update(tracker_dets)

    # --- Draw each tracked vehicle ---
    for obj_id in list(tracker.objects.keys()):
        if obj_id not in tracker.bboxes:
            continue

        bbox = tracker.bboxes[obj_id]
        x1, y1, x2, y2 = bbox
        cx, cy = tracker.objects[obj_id]

        # Find matching detection for type info
        det_type = 'car'
        det_conf = 0.0
        plate_bbox_local = None
        for det in detections:
            dx1, dy1, dx2, dy2 = det['bbox']
            if abs(dx1 - x1) < 50 and abs(dy1 - y1) < 50:
                det_type = det.get('type', 'car')
                det_conf = det.get('confidence', 0.0)
                plate_bbox_local = det.get('plate_bbox')
                break

        color = type_colors.get(det_type, COLOR_VEHICLE)
        plate_text = tracker.plate_texts.get(obj_id)

        # --- Draw vehicle bounding box ---
        # Corner-style box (like highway cameras)
        corner_len = min(30, (x2 - x1) // 4, (y2 - y1) // 4)
        thickness = 2

        # Top-left corner
        cv2.line(annotated, (x1, y1), (x1 + corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x1, y1), (x1, y1 + corner_len), color, thickness, cv2.LINE_AA)
        # Top-right corner
        cv2.line(annotated, (x2, y1), (x2 - corner_len, y1), color, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x2, y1), (x2, y1 + corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-left corner
        cv2.line(annotated, (x1, y2), (x1 + corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x1, y2), (x1, y2 - corner_len), color, thickness, cv2.LINE_AA)
        # Bottom-right corner
        cv2.line(annotated, (x2, y2), (x2 - corner_len, y2), color, thickness, cv2.LINE_AA)
        cv2.line(annotated, (x2, y2), (x2, y2 - corner_len), color, thickness, cv2.LINE_AA)

        # Thin border connecting corners
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

        # --- Vehicle ID & type label ---
        label = f"ID:{obj_id} {det_type.upper()}"
        if det_conf > 0:
            label += f" {det_conf:.0%}"

        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_w, label_h = label_size

        # Label background
        cv2.rectangle(annotated, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # --- Draw centroid dot ---
        cv2.circle(annotated, (cx, cy), 4, color, -1)

        # --- Draw license plate box ---
        if plate_bbox_local:
            px, py, pw, ph = plate_bbox_local
            # Bright plate highlight box
            cv2.rectangle(annotated, (px - 2, py - 2), (px + pw + 2, py + ph + 2),
                          COLOR_PLATE, 2, cv2.LINE_AA)

            # Plate text below the plate box
            if plate_text:
                plate_label = f"PLATE: {plate_text}"
                pl_size, _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                pl_w, pl_h = pl_size

                cv2.rectangle(annotated, (px, py + ph + 2), (px + pl_w + 10, py + ph + pl_h + 12),
                              COLOR_PLATE_BG, -1)
                cv2.putText(annotated, plate_label, (px + 5, py + ph + pl_h + 7),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PLATE, 2, cv2.LINE_AA)

        elif plate_text:
            # Show plate text at bottom of vehicle box even without exact plate bbox
            plate_label = f"PLATE: {plate_text}"
            pl_size, _ = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            pl_w, pl_h = pl_size

            cv2.rectangle(annotated, (x1, y2), (x1 + pl_w + 10, y2 + pl_h + 10),
                          COLOR_PLATE_BG, -1)
            cv2.putText(annotated, plate_label, (x1 + 5, y2 + pl_h + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_PLATE, 2, cv2.LINE_AA)

        # --- Track and count new vehicles across the entire full cam view ---
        if not tracker.saved.get(obj_id, False):
            tracker.saved[obj_id] = True
            total_vehicles_counted += 1
            if plate_text:
                detected_plates_log.insert(0, {
                    'plate': plate_text,
                    'type': det_type,
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'id': obj_id,
                })
                # Keep only last 10
                if len(detected_plates_log) > 10:
                    detected_plates_log.pop()

    return annotated


# Legacy draw function (kept for backward compat)
def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame (highway style)."""
    return draw_highway_hud(frame, detections, vehicle_tracker)


# ============================================================
# Save Detection to Database
# ============================================================

def save_detection(detection):
    """Save a detection to the database."""
    from .models import VehicleRecord, DailyReport

    plate = detection.get('plate')
    if not plate:
        plate = 'UNREADABLE'

    # Save captured image
    capture_path = None
    plate_path = None

    today_str = date.today().strftime('%Y/%m/%d')
    timestamp = datetime.now().strftime('%H%M%S_%f')

    captures_dir = Path(settings.MEDIA_ROOT) / 'captures' / today_str
    captures_dir.mkdir(parents=True, exist_ok=True)

    plates_dir = Path(settings.MEDIA_ROOT) / 'plates' / today_str
    plates_dir.mkdir(parents=True, exist_ok=True)

    if detection.get('vehicle_crop') is not None:
        img_name = f"{plate}_{timestamp}.jpg"
        img_path = captures_dir / img_name
        cv2.imwrite(str(img_path), detection['vehicle_crop'])
        capture_path = f"captures/{today_str}/{img_name}"

    if detection.get('plate_img') is not None:
        plate_name = f"{plate}_{timestamp}_plate.jpg"
        plate_path_full = plates_dir / plate_name
        cv2.imwrite(str(plate_path_full), detection['plate_img'])
        plate_path = f"plates/{today_str}/{plate_name}"

    # Create vehicle record
    record = VehicleRecord.objects.create(
        license_plate=plate,
        vehicle_type=detection.get('type', 'unknown'),
        confidence=detection.get('confidence', 0.0),
        captured_image=capture_path,
        plate_image=plate_path,
        detected_at=timezone.now(),
    )

    # Update daily report
    today = date.today()
    report, created = DailyReport.objects.get_or_create(date=today)
    report.total_vehicles = VehicleRecord.objects.filter(
        detected_at__date=today
    ).count()

    type_counts = {
        'total_cars': VehicleRecord.objects.filter(detected_at__date=today, vehicle_type='car').count(),
        'total_trucks': VehicleRecord.objects.filter(detected_at__date=today, vehicle_type='truck').count(),
        'total_buses': VehicleRecord.objects.filter(detected_at__date=today, vehicle_type='bus').count(),
        'total_motorcycles': VehicleRecord.objects.filter(detected_at__date=today, vehicle_type='motorcycle').count(),
        'total_autos': VehicleRecord.objects.filter(detected_at__date=today, vehicle_type='auto').count(),
        'total_unknown': VehicleRecord.objects.filter(detected_at__date=today, vehicle_type='unknown').count(),
    }

    for key, val in type_counts.items():
        setattr(report, key, val)

    report.unique_plates = VehicleRecord.objects.filter(
        detected_at__date=today
    ).exclude(license_plate='UNREADABLE').values('license_plate').distinct().count()

    report.save()

    return record


# ============================================================
# Detection Feed Generator (Real-time Highway Camera Mode)
# ============================================================

def generate_detection_feed(source=0):
    """
    Generator for streaming video with real-time vehicle detection.
    Highway-camera-style with tracking, plate reading, and HUD overlay.
    Runs detection in a background thread to prevent lag.
    """
    global vehicle_tracker, total_vehicles_counted, detected_plates_log

    cam = get_camera(source)
    
    # Reset tracker for new feed
    vehicle_tracker = CentroidTracker(max_disappeared=40, max_distance=100)
    total_vehicles_counted = 0
    detected_plates_log = []
    
    saved_plates = set()
    # Threading setup for async detection
    import queue
    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    
    def detection_worker():
        while True:
            try:
                frame_to_detect = frame_queue.get(timeout=1.0)
                if frame_to_detect is None:
                    break  # Stop signal
                    
                dets = detect_vehicles_in_frame(frame_to_detect)
                
                # Try to put results in queue without blocking
                try:
                    # Clear out old results if falling behind
                    while not result_queue.empty():
                        result_queue.get_nowait()
                    result_queue.put_nowait(dets)
                except queue.Full:
                    pass
                # Clear large objects from memory
                del dets
                del frame_to_detect
                gc.collect()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    # Start background detection thread
    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()

    frame_count = 0
    detect_interval = 3    # Send frame to detector every N frames
    last_detections = []
    fps_timer = time.time()
    current_fps = "--"

    try:
        while True:
            frame = cam.get_frame()
            if frame is not None:
                frame_count += 1

                # Calculate FPS
                if frame_count % 30 == 0:
                    current_fps = f"{30.0 / (time.time() - fps_timer):.1f}"
                    fps_timer = time.time()

                # Send frame to detection worker periodically (copying only when needed)
                if frame_count % detect_interval == 0:
                    try:
                        # Clear old frame if still in queue (to save memory)
                        while not frame_queue.empty():
                            frame_queue.get_nowait()
                        # Queue the freshest frame
                        frame_queue.put_nowait(frame.copy())
                    except queue.Full:
                        pass
                
                # Check if new detections are ready
                try:
                    last_detections = result_queue.get_nowait()
                    
                    # Process new detections (save to DB)
                    for det in last_detections:
                        plate = det.get('plate')
                        if plate and plate not in saved_plates:
                            try:
                                save_detection(det)
                                saved_plates.add(plate)
                                logger.info(f"Saved plate: {plate}")
                            except Exception as e:
                                logger.error(f"Save error: {e}")
                                
                except queue.Empty:
                    pass # Keep using last_detections

                # Draw HUD on every frame (using last known detections)
                annotated = draw_highway_hud(frame, last_detections, vehicle_tracker)
                
                ret, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
                
                # Clear references from memory
                del frame
                del annotated
                
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                del jpeg
            time.sleep(0.03)  # ~30fps loop limit
    finally:
        # Cleanup when client disconnects
        try:
            while not frame_queue.empty():
                frame_queue.get_nowait()
            frame_queue.put_nowait(None)
        except Exception:
            pass


def capture_snapshot(source=0):
    """Capture a single frame and run detection."""
    cam = get_camera(source)
    frame = cam.get_frame()

    if frame is None:
        return None, []

    detections = detect_vehicles_in_frame(frame)

    # Save each detection with a plate
    saved_records = []
    for det in detections:
        try:
            record = save_detection(det)
            saved_records.append(record)
        except Exception as e:
            logger.error(f"Save error: {e}")

    annotated = draw_highway_hud(frame, detections, vehicle_tracker)
    return annotated, saved_records

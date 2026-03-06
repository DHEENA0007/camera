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

# Lazy-loaded models
_yolo_model = None
_ocr_reader = None


def get_yolo_model():
    """Lazy load YOLO model."""
    global _yolo_model
    if _yolo_model is None:
        try:
            from ultralytics import YOLO
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'yolov8n.pt')
            if not os.path.exists(model_path):
                model_path = 'yolov8n.pt'
            _yolo_model = YOLO(model_path)
            logger.info("YOLOv8 model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
    return _yolo_model


def get_ocr_reader():
    """Lazy load EasyOCR reader."""
    global _ocr_reader
    if _ocr_reader is None:
        try:
            import easyocr
            _ocr_reader = easyocr.Reader(['en'], gpu=False)
            logger.info("EasyOCR reader loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load EasyOCR: {e}")
    return _ocr_reader


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
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame.copy()
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


# ============================================================
# COCO Vehicle Classes
# ============================================================

VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}


# ============================================================
# License Plate Detection (Image Processing Pipeline)
# ============================================================

def detect_plate_region(vehicle_img):
    """
    Detect license plate region within a vehicle image using
    image processing techniques (edge detection + contour analysis).
    Returns list of plate candidate regions as (x, y, w, h) relative to vehicle_img.
    """
    if vehicle_img is None or vehicle_img.size == 0:
        return []

    h, w = vehicle_img.shape[:2]
    if h < 20 or w < 20:
        return []

    # Focus on lower 60% of vehicle (plates are usually at bottom)
    roi_y_start = int(h * 0.35)
    roi = vehicle_img[roi_y_start:, :]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply bilateral filter to reduce noise while keeping edges sharp
    gray = cv2.bilateralFilter(gray, 11, 17, 17)

    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # Edge detection
    edges = cv2.Canny(gray, 30, 200)

    # Dilate edges to connect nearby edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate_candidates = []

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contour)
        if area < 500:
            continue

        # Approximate contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        x, y, cw, ch = cv2.boundingRect(contour)
        aspect_ratio = cw / ch if ch > 0 else 0

        # License plates typically have aspect ratio between 2.0 and 6.0
        # Indian plates: ~4.5:1 ratio
        if 1.5 <= aspect_ratio <= 7.0 and ch > 15 and cw > 60:
            # Adjust coordinates back to full vehicle image
            plate_candidates.append((x, y + roi_y_start, cw, ch))

        # Also check for rectangular approximations (4 corners)
        if len(approx) >= 4 and len(approx) <= 8:
            x, y, cw, ch = cv2.boundingRect(approx)
            aspect_ratio = cw / ch if ch > 0 else 0
            if 1.5 <= aspect_ratio <= 7.0 and ch > 15 and cw > 60:
                plate_candidates.append((x, y + roi_y_start, cw, ch))

    # Also try morphological approach as fallback
    if len(plate_candidates) == 0:
        # Blackhat morphology to find dark regions on light background (plate chars)
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

        # Threshold
        _, thresh = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours in thresholded image
        contours2, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in sorted(contours2, key=cv2.contourArea, reverse=True)[:10]:
            x, y, cw, ch = cv2.boundingRect(contour)
            aspect_ratio = cw / ch if ch > 0 else 0
            if 1.5 <= aspect_ratio <= 7.0 and ch > 10 and cw > 40:
                plate_candidates.append((x, y + roi_y_start, cw, ch))

    # Remove duplicate/overlapping candidates
    plate_candidates = _non_max_suppression(plate_candidates)

    return plate_candidates[:3]  # Return top 3 candidates


def _non_max_suppression(boxes, overlap_thresh=0.5):
    """Remove overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    boxes_arr = np.array(boxes, dtype=float)
    x1 = boxes_arr[:, 0]
    y1 = boxes_arr[:, 1]
    x2 = x1 + boxes_arr[:, 2]
    y2 = y1 + boxes_arr[:, 3]
    areas = boxes_arr[:, 2] * boxes_arr[:, 3]

    idxs = np.argsort(areas)[::-1]
    pick = []

    while len(idxs) > 0:
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)

        overlap = (w * h) / areas[idxs[1:]]
        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))

    return [boxes[i] for i in pick]


def preprocess_plate_for_ocr(plate_img):
    """
    Preprocess a plate image for optimal OCR results.
    Returns multiple preprocessed versions to try OCR on.
    """
    results = []

    if plate_img is None or plate_img.size == 0:
        return results

    h, w = plate_img.shape[:2]

    # Resize plate to standard height for better OCR
    target_h = 80
    scale = target_h / h
    resized = cv2.resize(plate_img, (int(w * scale), target_h), interpolation=cv2.INTER_CUBIC)

    # Version 1: Grayscale + adaptive threshold
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    results.append(gray)

    # Version 2: CLAHE enhanced
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    results.append(enhanced)

    # Version 3: Otsu threshold
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    results.append(otsu)

    # Version 4: Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    results.append(adaptive)

    # Version 5: Inverted (for dark plates with light text)
    results.append(cv2.bitwise_not(otsu))

    return results


def clean_plate_text(text):
    """Clean and validate license plate text."""
    # Remove non-alphanumeric characters
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text.upper().strip())

    # Common OCR substitutions
    substitutions = {
        'O': '0', 'I': '1', 'S': '5', 'B': '8',
        'G': '6', 'Z': '2', 'D': '0', 'Q': '0',
    }

    # Indian plates typically: XX 00 XX 0000
    # Apply substitutions only in expected number positions
    if len(cleaned) >= 4:
        # First 2 chars should be letters (state code)
        # Next 2 should be numbers (district code)
        # Then letters (series), then numbers
        corrected = list(cleaned)

        # Fix positions that should be digits
        for i in range(len(corrected)):
            if i in (2, 3) or (i >= len(corrected) - 4):
                if corrected[i] in substitutions:
                    corrected[i] = substitutions[corrected[i]]

        cleaned = ''.join(corrected)

    # Valid plate: minimum 4 chars, maximum 12
    if 4 <= len(cleaned) <= 12:
        return cleaned
    return None


def read_plate_text(plate_img):
    """
    Read text from a plate image using EasyOCR with preprocessing.
    Returns (text, confidence) or (None, 0.0)
    """
    reader = get_ocr_reader()
    if reader is None:
        return None, 0.0

    preprocessed_versions = preprocess_plate_for_ocr(plate_img)

    best_text = None
    best_conf = 0.0

    for processed in preprocessed_versions:
        try:
            ocr_results = reader.readtext(processed, detail=1, paragraph=False)
            for (bbox_ocr, text, conf) in ocr_results:
                cleaned = clean_plate_text(text)
                if cleaned and conf > best_conf:
                    best_text = cleaned
                    best_conf = conf
        except Exception as e:
            logger.debug(f"OCR attempt error: {e}")
            continue

    # Also try reading the original color image
    if best_conf < 0.5:
        try:
            ocr_results = reader.readtext(plate_img, detail=1, paragraph=False)
            for (bbox_ocr, text, conf) in ocr_results:
                cleaned = clean_plate_text(text)
                if cleaned and conf > best_conf:
                    best_text = cleaned
                    best_conf = conf
        except Exception:
            pass

    return best_text, best_conf


# ============================================================
# Vehicle Detection Pipeline
# ============================================================

def detect_vehicles_in_frame(frame):
    """
    Full detection pipeline:
    1. YOLOv8 detects vehicles
    2. For each vehicle, locate license plate region
    3. OCR reads plate text
    Returns list of detections with bbox, type, plate info
    """
    model = get_yolo_model()
    if model is None:
        return []

    detections = []

    try:
        results = model(frame, verbose=False, conf=0.35)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue

                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                vehicle_type = VEHICLE_CLASSES[cls_id]

                # Ensure bbox is within frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                # Crop vehicle region
                vehicle_crop = frame[y1:y2, x1:x2]

                plate_text = None
                plate_conf = 0.0
                plate_img = None
                plate_bbox = None  # Relative to frame

                if vehicle_crop.size > 0:
                    # Step 2: Detect plate region within vehicle
                    plate_candidates = detect_plate_region(vehicle_crop)

                    for (px, py, pw, ph) in plate_candidates:
                        # Extract plate image from vehicle crop
                        plate_roi = vehicle_crop[py:py+ph, px:px+pw]

                        if plate_roi.size > 0:
                            # Step 3: OCR the plate
                            text, text_conf = read_plate_text(plate_roi)

                            if text and text_conf > plate_conf:
                                plate_text = text
                                plate_conf = text_conf
                                plate_img = plate_roi.copy()
                                # Convert plate bbox to frame coordinates
                                plate_bbox = (x1 + px, y1 + py, pw, ph)

                detections.append({
                    'type': vehicle_type,
                    'plate': plate_text,
                    'plate_conf': plate_conf,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'vehicle_crop': vehicle_crop,
                    'plate_img': plate_img,
                    'plate_bbox': plate_bbox,
                })

    except Exception as e:
        logger.error(f"Detection error: {e}")

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

    # --- Draw detection zone lines ---
    # Horizontal scan lines (like highway cameras)
    zone_y1 = int(h * 0.25)
    zone_y2 = int(h * 0.85)
    cv2.line(annotated, (0, zone_y1), (w, zone_y1), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.line(annotated, (0, zone_y2), (w, zone_y2), (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(annotated, "DETECTION ZONE", (10, zone_y1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

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

        # --- Track and count new vehicles that cross the detection zone ---
        if not tracker.saved.get(obj_id, False):
            if cy > zone_y2:
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

    # ============================================================
    # Draw HUD Overlay (top bar & side panel)
    # ============================================================

    # --- Top HUD bar ---
    hud_h = 45
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, hud_h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    # Title
    cv2.putText(annotated, "HIGHWAY VEHICLE MONITORING SYSTEM", (15, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)

    # Timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    ts_size, _ = cv2.getTextSize(timestamp, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(annotated, timestamp, (w - ts_size[0] - 15, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)

    # Active vehicles count
    active_text = f"ACTIVE: {len(tracker.objects)}"
    cv2.putText(annotated, active_text, (15, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_VEHICLE, 1, cv2.LINE_AA)

    # Total count
    total_text = f"TOTAL: {total_vehicles_counted}"
    cv2.putText(annotated, total_text, (180, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 255), 1, cv2.LINE_AA)

    # Recording indicator
    if int(time.time() * 2) % 2 == 0:  # Blinking
        cv2.circle(annotated, (w - 30, 35), 6, (0, 0, 255), -1)
        cv2.putText(annotated, "REC", (w - 60, 39),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

    # --- Side panel: Recent plate detections ---
    if detected_plates_log:
        panel_w = 220
        panel_x = w - panel_w - 10
        panel_y = hud_h + 10
        panel_h = min(len(detected_plates_log) * 30 + 35, 340)

        overlay2 = annotated.copy()
        cv2.rectangle(overlay2, (panel_x, panel_y), (w - 10, panel_y + panel_h),
                      COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay2, 0.7, annotated, 0.3, 0, annotated)

        cv2.putText(annotated, "DETECTED PLATES", (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_PLATE, 1, cv2.LINE_AA)
        cv2.line(annotated, (panel_x + 10, panel_y + 28),
                 (w - 20, panel_y + 28), COLOR_PLATE, 1)

        for i, entry in enumerate(detected_plates_log[:10]):
            ey = panel_y + 48 + i * 28
            cv2.putText(annotated, f"{entry['plate']}", (panel_x + 10, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT, 1, cv2.LINE_AA)
            cv2.putText(annotated, f"{entry['time']}", (panel_x + 140, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1, cv2.LINE_AA)

    # --- Bottom status bar ---
    bar_y = h - 25
    overlay3 = annotated.copy()
    cv2.rectangle(overlay3, (0, bar_y), (w, h), COLOR_HUD_BG, -1)
    cv2.addWeighted(overlay3, 0.7, annotated, 0.3, 0, annotated)

    status_text = f"YOLO v8  |  FPS: --  |  Vehicles in frame: {len(detections)}  |  Plates read: {len([d for d in detections if d.get('plate')])}"
    cv2.putText(annotated, status_text, (15, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA)

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
    """
    global vehicle_tracker, total_vehicles_counted, detected_plates_log

    cam = get_camera(source)
    frame_count = 0
    detect_interval = 3    # Detect every N frames (lower = more responsive)
    last_detections = []
    fps_timer = time.time()
    fps = 0.0
    saved_plates = set()   # Track which plates we've already saved

    # Reset tracker for new feed
    vehicle_tracker = CentroidTracker(max_disappeared=40, max_distance=100)
    total_vehicles_counted = 0
    detected_plates_log = []

    while True:
        frame = cam.get_frame()
        if frame is not None:
            frame_count += 1

            # Calculate FPS
            if frame_count % 30 == 0:
                fps = 30.0 / (time.time() - fps_timer)
                fps_timer = time.time()

            # Run detection periodically
            if frame_count % detect_interval == 0:
                detections = detect_vehicles_in_frame(frame)
                last_detections = detections

                # Save new plate detections to database
                for det in detections:
                    plate = det.get('plate')
                    if plate and plate not in saved_plates:
                        try:
                            save_detection(det)
                            saved_plates.add(plate)
                            logger.info(f"Saved plate: {plate}")
                        except Exception as e:
                            logger.error(f"Save error: {e}")

            # Draw HUD on every frame (using last known detections)
            annotated = draw_highway_hud(frame, last_detections, vehicle_tracker)

            ret, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)


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

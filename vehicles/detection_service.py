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

# Lazy-loaded YOLOv8 model (vehicle detection)
yolo_model = None
yolo_lock = threading.Lock()

# Lazy-loaded custom Indian plate detector (YOLOv8 trained on Indian plates)
plate_detector_model = None
plate_detector_lock = threading.Lock()

# Lazy-loaded EasyOCR reader (for plate text recognition)
ocr_reader = None
ocr_lock = threading.Lock()

# COCO class IDs for vehicles
VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Path to custom trained Indian plate model
INDIAN_PLATE_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), 'models', 'indian_plates_yolov8.pt'
)

def get_yolo():
    """Lazy load YOLOv8 model for vehicle detection."""
    global yolo_model
    with yolo_lock:
        if yolo_model is None:
            try:
                from ultralytics import YOLO
                yolo_model = YOLO('yolov8n.pt')
                logger.info("YOLOv8 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load YOLOv8: {e}")
                yolo_model = False
        return yolo_model if yolo_model is not False else None

def get_plate_detector():
    """Lazy load custom YOLOv8 Indian plate detection model."""
    global plate_detector_model
    with plate_detector_lock:
        if plate_detector_model is None:
            if os.path.exists(INDIAN_PLATE_MODEL_PATH):
                try:
                    from ultralytics import YOLO
                    plate_detector_model = YOLO(INDIAN_PLATE_MODEL_PATH)
                    logger.info(f"Indian plate detector loaded: {INDIAN_PLATE_MODEL_PATH}")
                except Exception as e:
                    logger.error(f"Failed to load Indian plate detector: {e}")
                    plate_detector_model = False
            else:
                logger.warning(
                    f"Indian plate model not found at {INDIAN_PLATE_MODEL_PATH}. "
                    "Run train_indian_plates.py to train and generate the model."
                )
                plate_detector_model = False
        return plate_detector_model if plate_detector_model is not False else None


def get_ocr():
    """Lazy load EasyOCR reader for plate text recognition."""
    global ocr_reader
    with ocr_lock:
        if ocr_reader is None:
            try:
                import easyocr
                ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
                logger.info("EasyOCR loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load EasyOCR: {e}")
                ocr_reader = False
        return ocr_reader if ocr_reader is not False else None


def _detect_plates_with_custom_model(frame):
    """
    Detect Indian license plates using custom-trained YOLOv8 + EasyOCR.

    Returns a list in the same format as OpenALPR results so the rest of the
    pipeline (plate-vehicle association, tracking, etc.) works unchanged:
      [{'coordinates': [{'x':..,'y':..}, ...],
        'candidates': [{'plate': 'TEXT', 'confidence': 85.0}, ...]}, ...]
    """
    detector = get_plate_detector()
    if detector is None:
        return []

    ocr = get_ocr()
    h, w = frame.shape[:2]

    # Upscale small frames for better detection
    scale = 1.0
    process_frame = frame
    if w < 800:
        scale = 800.0 / w
        process_frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                   interpolation=cv2.INTER_CUBIC)

    try:
        results = detector.predict(process_frame, conf=0.35, verbose=False, imgsz=640)
    except Exception as e:
        logger.error(f"Plate detector inference error: {e}")
        return []

    plate_results = []
    boxes = results[0].boxes if results else []

    for box in boxes:
        det_conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

        # Scale coords back to original frame size
        if scale != 1.0:
            x1 = int(x1 / scale)
            y1 = int(y1 / scale)
            x2 = int(x2 / scale)
            y2 = int(y2 / scale)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            continue

        plate_crop = frame[y1:y2, x1:x2]
        plate_text = ''
        ocr_conf = det_conf

        if ocr is not None and plate_crop.size > 0:
            try:
                # Preprocess plate crop for better OCR
                plate_gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
                # Upscale plate crop for OCR accuracy
                ph, pw = plate_gray.shape
                if pw < 200:
                    scale_ocr = 200.0 / pw
                    plate_gray = cv2.resize(plate_gray, None, fx=scale_ocr, fy=scale_ocr,
                                            interpolation=cv2.INTER_CUBIC)
                _, plate_bin = cv2.threshold(plate_gray, 0, 255,
                                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ocr_results = ocr.readtext(plate_bin, allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                if ocr_results:
                    texts = [r[1].upper().replace(' ', '') for r in ocr_results if r[2] > 0.3]
                    confs = [r[2] for r in ocr_results if r[2] > 0.3]
                    if texts:
                        plate_text = ''.join(texts)
                        ocr_conf = float(np.mean(confs))
            except Exception as e:
                logger.warning(f"OCR error on plate crop: {e}")

        if not plate_text:
            plate_text = 'UNREADABLE'

        # Build result in OpenALPR-compatible format
        plate_results.append({
            'coordinates': [
                {'x': x1, 'y': y1},
                {'x': x2, 'y': y1},
                {'x': x2, 'y': y2},
                {'x': x1, 'y': y2},
            ],
            'candidates': [{'plate': plate_text, 'confidence': ocr_conf * 100}],
        })

    if plate_results:
        logger.info(f"Custom detector found {len(plate_results)} plates: "
                    f"{[r['candidates'][0]['plate'] for r in plate_results]}")
    return plate_results


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
                    alpr_instance.set_top_n(7)
                    alpr_instance.set_detect_region(False)
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
        self.plate_texts_history = OrderedDict() # id -> list of (text, conf, crop_info)
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
        self.plate_texts_history[object_id] = []
        if plate_text:
            self.plate_texts_history[object_id].append(plate_text)
        self.saved[object_id] = False
        self.next_object_id += 1
        return object_id

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.bboxes[object_id]
        del self.disappeared[object_id]
        del self.plate_texts[object_id]
        del self.plate_confs[object_id]
        del self.plate_texts_history[object_id]
        del self.saved[object_id]

    def get_consensus_plate(self, object_id):
        """
        Build consensus plate from multiple partial reads.

        Strategy:
        1. Pick the longest reads as the best full-plate candidates.
        2. Use character-position voting across aligned reads.
        3. Apply Indian plate format correction.
        """
        candidates = self.plate_texts_history.get(object_id, [])
        if not candidates:
            return self.plate_texts.get(object_id)

        # Clean all candidates
        clean = []
        for c in candidates:
            c = re.sub(r'[^A-Z0-9]', '', c.upper())
            if len(c) >= 4:
                clean.append(c)

        if not clean:
            return self.plate_texts.get(object_id)

        # Group by length and find the most common full-length reads
        from collections import Counter
        length_counts = Counter(len(c) for c in clean)
        # The true plate length is likely the most common length among longer reads
        # Filter to lengths that appear at least twice, or just take the mode of longer ones
        long_reads = [c for c in clean if len(c) >= max(len(x) for x in clean) - 2]

        if not long_reads:
            long_reads = clean

        # Frequency vote: count exact occurrences, pick the most common
        freq = Counter(long_reads)
        best_text, best_count = freq.most_common(1)[0]

        # If we have a clear winner (seen 2+ times), use it directly
        if best_count >= 2:
            return self._fix_indian_format(best_text)

        # Otherwise, use character-position voting on the longest reads
        # Align reads by trying to find the best alignment offset
        target_len = len(long_reads[0])
        # Use the longest read as anchor
        anchor = max(long_reads, key=len)
        target_len = len(anchor)

        # Build a character vote matrix
        votes = [Counter() for _ in range(target_len)]

        for read in clean:
            # Try to align this read to the anchor via substring match
            offset = self._find_best_alignment(anchor, read)
            if offset is not None:
                for i, ch in enumerate(read):
                    pos = offset + i
                    if 0 <= pos < target_len:
                        votes[pos][ch] += 1

        # Build consensus from votes
        result = []
        for pos_votes in votes:
            if pos_votes:
                result.append(pos_votes.most_common(1)[0][0])
            elif result:
                # Gap in the middle - shouldn't happen normally
                break

        consensus = ''.join(result)
        return self._fix_indian_format(consensus) if consensus else self.plate_texts.get(object_id)

    def _find_best_alignment(self, anchor, read):
        """Find the best offset to align `read` within `anchor`."""
        if read in anchor:
            return anchor.index(read)

        # Try substring matching with 1 char tolerance
        best_offset = None
        best_score = 0
        for offset in range(-(len(read) - 2), len(anchor)):
            score = 0
            count = 0
            for i, ch in enumerate(read):
                pos = offset + i
                if 0 <= pos < len(anchor):
                    count += 1
                    if anchor[pos] == ch:
                        score += 1
            if count >= min(3, len(read)) and score > best_score:
                best_score = score
                best_offset = offset

        # Only accept if at least 50% of overlapping chars match
        if best_offset is not None and best_score >= max(2, len(read) * 0.4):
            return best_offset
        return 0  # Default: align at start

    def _fix_indian_format(self, plate):
        """
        Apply Indian license plate format corrections.
        Indian format: [State 2L][District 2D][Series 1-3L][Number 1-4D]
        e.g., 22BH6517A -> correct is a BH-series plate
        """
        if not plate or len(plate) < 4:
            return plate

        res = list(plate)

        # Common OCR digit<->letter confusions
        digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '5': 'S', '8': 'B', '6': 'G'}
        letter_to_digit = {'O': '0', 'I': '1', 'Z': '2', 'S': '5', 'B': '8', 'G': '6', 'D': '0', 'Q': '0'}

        # Try to detect Indian format by pattern matching
        # Pattern: 2 letters, 2 digits, 1-3 letters, 1-4 digits
        # Also handle BH-series (Bharat series): 2 digits, BH, 4 digits, 1-2 letters

        # Check if this looks like a BH-series plate: ##BH####X
        bh_match = re.match(r'^(\d{2})(BH|8H|6H)(\d{3,4})([A-Z]{1,2})?$', plate)
        if bh_match:
            # BH-series format is correct as-is, just clean it
            return plate

        # Standard Indian format: LL DD LLL DDDD
        # Fix first 2 chars to be letters (state code)
        for i in range(min(2, len(res))):
            if res[i] in digit_to_letter and res[i].isdigit():
                res[i] = digit_to_letter[res[i]]

        # Fix chars at position 2-3 to be digits (district code)
        for i in range(2, min(4, len(res))):
            if res[i] in letter_to_digit and res[i].isalpha():
                res[i] = letter_to_digit[res[i]]

        return ''.join(res)

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
                if input_plates[col]:
                    if obj_id not in self.plate_texts_history:
                        self.plate_texts_history[obj_id] = []
                    
                    self.plate_texts_history[obj_id].append(input_plates[col])
                    # Keep only last 25 candidates for better voting
                    if len(self.plate_texts_history[obj_id]) > 25:
                        self.plate_texts_history[obj_id].pop(0)

                    if input_confs[col] > self.plate_confs.get(obj_id, 0):
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


# ============================================================
# Plate De-Duplication Logic (Fuzzy)
# ============================================================

def get_levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return get_levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]

def get_string_similarity(s1, s2):
    if not s1 or not s2: return 0
    dist = get_levenshtein_distance(s1, s2)
    return 1.0 - (dist / max(len(s1), len(s2)))

class PlateDeDuplicator:
    """
    Prevents counting the same license plate multiple times using FUZZY matching.
    """
    def __init__(self, cooldown_seconds=300, similarity_threshold=0.7):
        self.seen_plates = {} # plate -> last_seen_time
        self.cooldown = cooldown_seconds
        self.similarity_threshold = similarity_threshold

    def is_duplicate(self, plate_text):
        if not plate_text or plate_text == 'UNREADABLE':
            return False
            
        now = time.time()
        
        # Fuzzy check against recently seen plates
        for seen_text, last_seen in list(self.seen_plates.items()):
            if now - last_seen > self.cooldown:
                continue
                
            similarity = get_string_similarity(plate_text, seen_text)
            if similarity >= self.similarity_threshold:
                # Update the record with the longer/better version of the plate
                self.seen_plates[seen_text] = now
                if len(plate_text) > len(seen_text) and get_alpr(): # Check if it's a better read
                     # We don't replace keys in dict easily while iterating, but next 
                     # time this improved text will be the anchor.
                     pass 
                return True
        
        self.seen_plates[plate_text] = now
        self._cleanup()
        return False

    def _cleanup(self):
        now = time.time()
        expired = [p for p, t in self.seen_plates.items() if now - t > self.cooldown * 2]
        for p in expired:
            del self.seen_plates[p]

# Global instances
plate_deduper = PlateDeDuplicator(cooldown_seconds=300, similarity_threshold=0.7)
vehicle_tracker = CentroidTracker(max_disappeared=90, max_distance=150) # Increased persistence for high-speed
total_vehicles_counted = 0
detected_plates_log = []


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


def _merge_plate_results(plate_results, frame_h, frame_w):
    """
    Merge overlapping/nearby plate detections from OpenALPR.
    OpenALPR sometimes returns split reads for the same plate region.
    Keep only the highest-confidence result for each physical plate location.
    """
    if len(plate_results) <= 1:
        return plate_results

    # Extract bounding boxes and sort by confidence (highest first)
    items = []
    for pr in plate_results:
        if not pr.get('candidates'):
            continue
        coords = pr['coordinates']
        px_min = min(c['x'] for c in coords)
        py_min = min(c['y'] for c in coords)
        px_max = max(c['x'] for c in coords)
        py_max = max(c['y'] for c in coords)
        cx = (px_min + px_max) / 2
        cy = (py_min + py_max) / 2
        conf = pr['candidates'][0]['confidence']
        items.append({'result': pr, 'cx': cx, 'cy': cy,
                       'x1': px_min, 'y1': py_min, 'x2': px_max, 'y2': py_max,
                       'conf': conf})

    items.sort(key=lambda x: x['conf'], reverse=True)

    merged = []
    used = set()
    for i, item in enumerate(items):
        if i in used:
            continue
        merged.append(item['result'])
        used.add(i)
        # Suppress any overlapping or nearby detections
        for j in range(i + 1, len(items)):
            if j in used:
                continue
            other = items[j]
            # Check if centers are close (within 1.5x plate width)
            plate_w = item['x2'] - item['x1']
            plate_h = item['y2'] - item['y1']
            dist_x = abs(item['cx'] - other['cx'])
            dist_y = abs(item['cy'] - other['cy'])
            if dist_x < plate_w * 1.5 and dist_y < plate_h * 1.5:
                used.add(j)  # Suppress this lower-confidence duplicate

    return merged


def _associate_plate_to_vehicle(plate_coords, vehicle_boxes, frame_h, frame_w):
    """
    Associate a detected plate with the best matching YOLO vehicle box.
    Returns (vehicle_box_index, vehicle_type, vehicle_conf) or (None, 'car', 0.0).
    """
    px_min = min(c['x'] for c in plate_coords)
    py_min = min(c['y'] for c in plate_coords)
    px_max = max(c['x'] for c in plate_coords)
    py_max = max(c['y'] for c in plate_coords)
    plate_cx = (px_min + px_max) / 2
    plate_cy = (py_min + py_max) / 2

    best_idx = None
    best_score = -1

    for idx, vbox in enumerate(vehicle_boxes):
        vx1, vy1, vx2, vy2, vtype, vconf = vbox
        # Check if plate center is inside or near the vehicle box
        if vx1 <= plate_cx <= vx2 and vy1 <= plate_cy <= vy2:
            # Plate is inside vehicle — prefer smaller (tighter) vehicle boxes
            area = (vx2 - vx1) * (vy2 - vy1)
            score = 1000000.0 / max(area, 1)  # Smaller area = higher score
            if score > best_score:
                best_score = score
                best_idx = idx
        else:
            # Plate is outside — check distance to vehicle center
            vcx = (vx1 + vx2) / 2
            vcy = (vy1 + vy2) / 2
            dist = math.sqrt((plate_cx - vcx) ** 2 + (plate_cy - vcy) ** 2)
            vw = vx2 - vx1
            vh = vy2 - vy1
            # Only consider if plate is reasonably close (within vehicle diagonal)
            max_dist = math.sqrt(vw ** 2 + vh ** 2) * 0.7
            if dist < max_dist:
                score = 1.0 / max(dist, 1)
                if score > best_score:
                    best_score = score
                    best_idx = idx

    if best_idx is not None:
        return best_idx, vehicle_boxes[best_idx][4], vehicle_boxes[best_idx][5]
    return None, 'car', 0.0


def detect_vehicles_in_frame(frame):
    """
    VaxALPR-style detection pipeline (plate-first approach):
      1. Run OpenALPR on the FULL FRAME to find all plates (plate-first)
      2. Run YOLOv8 to find all vehicles (provides vehicle type & bounding box)
      3. Associate each plate with its nearest vehicle
      4. For vehicles without plates, still include them (for counting)

    This approach is superior because:
    - OpenALPR has its own built-in plate detector optimized for finding plates
    - Running on full frame avoids cropping issues that cause split/partial reads
    - YOLO provides vehicle classification context, not plate finding
    """
    mode = getattr(settings, 'ALPR_MODE', 'local')

    if mode == 'online':
        ret, enc = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ret:
            return []
        frame_bytes = enc.tobytes()
        from .online_services import PlateRecognizerService
        detections = PlateRecognizerService.recognize(frame_bytes)
        for det in detections:
            if 'vehicle_crop' not in det:
                x1, y1, x2, y2 = det['bbox']
                if det.get('plate_bbox') == det['bbox']:
                    pw = x2 - x1
                    ph = y2 - y1
                    x1 = max(0, x1 - int(pw * 1.5))
                    y1 = max(0, y1 - int(ph * 4))
                    x2 = min(frame.shape[1], x1 + int(pw * 4))
                    y2 = min(frame.shape[0], y1 + int(ph * 6))
                    det['bbox'] = (x1, y1, x2, y2)
                det['vehicle_crop'] = frame[y1:y2, x1:x2]
                px, py, pw, ph = det['plate_bbox']
                det['plate_img'] = frame[py:py+ph, px:px+pw]
        return detections

    # --- Local Mode: Custom Indian Plate Detector (preferred) or OpenALPR + YOLO ---
    h, w = frame.shape[:2]
    detections = []

    # ===== STEP 1: Detect plates (custom model preferred over OpenALPR) =====
    plate_results = []

    # Try custom-trained Indian plate model first
    custom_detector = get_plate_detector()
    if custom_detector is not None:
        plate_results = _detect_plates_with_custom_model(frame)

    # Fall back to OpenALPR if custom model is unavailable or found nothing
    if not plate_results:
        alpr = get_alpr()
        if alpr is not None:
            try:
                scale = 1.0
                process_frame = frame
                if w < 800:
                    scale = 800.0 / w
                    process_frame = cv2.resize(frame, None, fx=scale, fy=scale,
                                               interpolation=cv2.INTER_CUBIC)

                ret, enc = cv2.imencode('.jpg', process_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ret:
                    alpr_raw = alpr.recognize_array(enc.tobytes())
                    raw_results = alpr_raw.get('results', [])

                    if scale != 1.0:
                        for pr in raw_results:
                            for coord in pr.get('coordinates', []):
                                coord['x'] = int(coord['x'] / scale)
                                coord['y'] = int(coord['y'] / scale)

                    plate_results = _merge_plate_results(raw_results, h, w)
                    if plate_results:
                        plates_found = [pr['candidates'][0]['plate'] for pr in plate_results if pr.get('candidates')]
                        logger.info(f"OpenALPR found {len(plate_results)} plates on full frame: {plates_found}")
            except Exception as e:
                logger.error(f"OpenALPR full-frame error: {e}")

    # ===== STEP 2: Run YOLO for vehicle detection =====
    vehicle_boxes = []  # (x1, y1, x2, y2, type, conf)
    yolo = get_yolo()
    if yolo is not None:
        try:
            results = yolo.predict(frame, conf=0.25, classes=list(VEHICLE_CLASSES.keys()),
                                   verbose=False, imgsz=640)
            boxes = results[0].boxes if results else []
            logger.debug(f"YOLO detected {len(boxes)} vehicles")

            for box in boxes:
                cls_id = int(box.cls[0])
                vehicle_type = VEHICLE_CLASSES.get(cls_id, 'car')
                det_conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                vehicle_boxes.append((x1, y1, x2, y2, vehicle_type, det_conf))
        except Exception as e:
            logger.error(f"YOLOv8 detection error: {e}")

    # ===== STEP 3: Associate plates with vehicles =====
    used_vehicles = set()

    for plate_result in plate_results:
        if not plate_result.get('candidates'):
            continue

        best_candidate = plate_result['candidates'][0]
        plate_text = best_candidate['plate']
        plate_conf = best_candidate['confidence'] / 100.0

        coords = plate_result['coordinates']
        px_min = min(c['x'] for c in coords)
        py_min = min(c['y'] for c in coords)
        px_max = max(c['x'] for c in coords)
        py_max = max(c['y'] for c in coords)
        pw = px_max - px_min
        ph = py_max - py_min

        # Find matching vehicle
        veh_idx, veh_type, veh_conf = _associate_plate_to_vehicle(
            coords, vehicle_boxes, h, w)

        if veh_idx is not None:
            used_vehicles.add(veh_idx)
            vx1, vy1, vx2, vy2 = vehicle_boxes[veh_idx][:4]
        else:
            # No YOLO vehicle found — build a synthetic vehicle bbox around the plate
            vx1 = max(0, px_min - pw)
            vy1 = max(0, py_min - int(ph * 3))
            vx2 = min(w, px_max + pw)
            vy2 = min(h, py_max + int(ph * 1.5))
            veh_type = 'car'
            veh_conf = plate_conf

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        plate_img = frame[max(0, py_min):min(h, py_max), max(0, px_min):min(w, px_max)]

        detections.append({
            'type': veh_type,
            'plate': plate_text,
            'plate_conf': plate_conf,
            'confidence': veh_conf,
            'bbox': (vx1, vy1, vx2, vy2),
            'vehicle_crop': vehicle_crop if vehicle_crop.size > 0 else None,
            'plate_img': plate_img if plate_img.size > 0 else None,
            'plate_bbox': (px_min, py_min, pw, ph),
        })

    # ===== STEP 4: Add vehicles without plates (for counting) =====
    for idx, vbox in enumerate(vehicle_boxes):
        if idx in used_vehicles:
            continue
        vx1, vy1, vx2, vy2, veh_type, veh_conf = vbox
        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        detections.append({
            'type': veh_type,
            'plate': None,
            'plate_conf': 0.0,
            'confidence': veh_conf,
            'bbox': (vx1, vy1, vx2, vy2),
            'vehicle_crop': vehicle_crop if vehicle_crop.size > 0 else None,
            'plate_img': None,
            'plate_bbox': None,
        })

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

    # --- Draw Header HUD ---
    cv2.rectangle(annotated, (0, 0), (w, 60), COLOR_HUD_BG, -1)
    cv2.line(annotated, (0, 60), (w, 60), (100, 100, 100), 1)

    # Mode indicator
    mode = getattr(settings, 'ALPR_MODE', 'local').upper()
    mode_color = (0, 255, 0) if mode == 'LOCAL' else (0, 200, 255)
    cv2.putText(annotated, f"ENGINE: {mode}", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2, cv2.LINE_AA)

    # Counter
    count_str = f"VEHICLES: {total_vehicles_counted}"
    cv2.putText(annotated, count_str, (w - 250, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    
    # Pulse detection indicator
    if len(detections) > 0:
        cv2.circle(annotated, (w - 30, 30), 10, (0, 0, 255), -1)

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
            # Check if we have enough readings for consensus
            candidates = tracker.plate_texts_history.get(obj_id, [])
            best_conf = tracker.plate_confs.get(obj_id, 0.0)

            # Save when: 2+ readings OR 1 high-confidence reading (>70%)
            # Full-frame ALPR gives good full plates, so we don't need many reads
            if len(candidates) >= 2 or (len(candidates) >= 1 and best_conf > 0.70):
                plate_text = tracker.get_consensus_plate(obj_id)

                # Check fuzzy duplicates (is this car already in the log?)
                is_duplicate = False
                if plate_text:
                    if plate_deduper.is_duplicate(plate_text):
                        is_duplicate = True
                        logger.info(f"Fuzzy duplicate suppressed: {plate_text}")

                # Mark as saved: positive ID means new, negative means duplicate
                # Use 'new' or 'duplicate' to distinguish
                tracker.saved[obj_id] = 'duplicate' if is_duplicate else 'new'

                if not is_duplicate:
                    total_vehicles_counted += 1
                    # Log it
                    detected_plates_log.insert(0, {
                        'plate': plate_text or 'UNKNOWN',
                        'type': det_type,
                        'time': datetime.now().strftime('%H:%M:%S'),
                        'id': obj_id,
                        'shots': len(candidates)
                    })
                    if len(detected_plates_log) > 10:
                        detected_plates_log.pop()

                # Store consensus back to tracker for display
                if plate_text:
                    tracker.plate_texts[obj_id] = plate_text

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
                if dets:
                    plates = [d.get('plate') for d in dets if d.get('plate')]
                    logger.info(f"Detection: {len(dets)} vehicles, plates: {plates}")

                # Try to put results in queue without blocking
                try:
                    while not result_queue.empty():
                        result_queue.get_nowait()
                    result_queue.put_nowait(dets)
                except queue.Full:
                    pass
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    # Start background detection thread
    worker_thread = threading.Thread(target=detection_worker, daemon=True)
    worker_thread.start()

    frame_count = 0
    detect_interval = 5    # Send frame to detector every N frames
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
                except queue.Empty:
                    pass # Keep using last_detections

                # Draw HUD on every frame (using last known detections)
                annotated = draw_highway_hud(frame, last_detections, vehicle_tracker)

                # Only save to DB after consensus is reached (tracker.saved == True)
                # This avoids saving raw partial reads like "116517" or "22DH57"
                for obj_id in list(vehicle_tracker.saved.keys()):
                    if vehicle_tracker.saved[obj_id] == 'new' and obj_id not in saved_plates:
                        consensus_plate = vehicle_tracker.plate_texts.get(obj_id)
                        if consensus_plate:
                            saved_plates.add(obj_id)
                            # Build detection dict with best available crop
                            bbox = vehicle_tracker.bboxes.get(obj_id)
                            det_to_save = {
                                'plate': consensus_plate,
                                'type': 'car',
                                'confidence': vehicle_tracker.plate_confs.get(obj_id, 0.0),
                            }
                            # Try to get crop from current detections
                            if bbox:
                                vx1, vy1, vx2, vy2 = bbox
                                det_to_save['vehicle_crop'] = frame[vy1:vy2, vx1:vx2]
                                det_to_save['bbox'] = bbox
                            for det in last_detections:
                                dx1, dy1, dx2, dy2 = det['bbox']
                                if bbox and abs(dx1 - bbox[0]) < 80 and abs(dy1 - bbox[1]) < 80:
                                    det_to_save['type'] = det.get('type', 'car')
                                    if 'plate_img' in det:
                                        det_to_save['plate_img'] = det['plate_img']
                                    if 'plate_bbox' in det:
                                        det_to_save['plate_bbox'] = det['plate_bbox']
                                    break
                            try:
                                save_detection(det_to_save)
                                logger.info(f"Saved merged plate to DB: {consensus_plate}")
                            except Exception as e:
                                logger.error(f"Save error: {e}")
                
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

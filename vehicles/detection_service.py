"""
Vehicle Detection & License Plate Recognition Service
Uses YOLOv8 for vehicle detection and EasyOCR for plate reading.
"""
import cv2
import numpy as np
import threading
import time
import os
import re
import logging
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
            _yolo_model = YOLO('yolov8n.pt')  # Nano model for speed
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
            # Try to interpret source as integer (webcam index)
            src = int(self.source)
        except (ValueError, TypeError):
            src = self.source  # RTSP URL

        self.cap = cv2.VideoCapture(src)
        if not self.cap.isOpened():
            logger.error(f"Cannot open camera source: {src}")
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.running = True

        # Start frame capture thread
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
                # Try to reconnect
                try:
                    src = int(self.source)
                except (ValueError, TypeError):
                    src = self.source
                self.cap.release()
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
            # Encode frame as JPEG
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ret:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        time.sleep(0.033)  # ~30 fps


# COCO class IDs for vehicles
VEHICLE_CLASSES = {
    2: 'car',
    3: 'motorcycle',
    5: 'bus',
    7: 'truck',
}


def clean_plate_text(text):
    """Clean and validate license plate text."""
    # Remove non-alphanumeric characters
    cleaned = re.sub(r'[^A-Za-z0-9]', '', text.upper().strip())
    # Indian plates are typically: XX 00 XX 0000
    # Minimum 4 chars, maximum 12
    if 4 <= len(cleaned) <= 12:
        return cleaned
    return None


def detect_vehicles_in_frame(frame):
    """
    Detect vehicles in a frame and attempt to read license plates.
    Returns list of detections: [{type, plate, confidence, bbox, plate_img}]
    """
    model = get_yolo_model()
    reader = get_ocr_reader()

    if model is None:
        return []

    detections = []

    try:
        results = model(frame, verbose=False, conf=0.4)

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

                # Crop vehicle region
                vehicle_crop = frame[y1:y2, x1:x2]

                # Try to detect license plate in the lower portion of vehicle
                h = y2 - y1
                plate_region = frame[y1 + int(h * 0.5):y2, x1:x2]

                plate_text = None
                plate_img = None

                if reader is not None and plate_region.size > 0:
                    try:
                        # Convert to grayscale for better OCR
                        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
                        # Apply threshold
                        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        ocr_results = reader.readtext(thresh)
                        for (bbox_ocr, text, ocr_conf) in ocr_results:
                            cleaned = clean_plate_text(text)
                            if cleaned and ocr_conf > 0.3:
                                plate_text = cleaned
                                plate_img = plate_region.copy()
                                break
                    except Exception as e:
                        logger.debug(f"OCR error: {e}")

                detections.append({
                    'type': vehicle_type,
                    'plate': plate_text,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'vehicle_crop': vehicle_crop,
                    'plate_img': plate_img,
                })

    except Exception as e:
        logger.error(f"Detection error: {e}")

    return detections


def draw_detections(frame, detections):
    """Draw bounding boxes and labels on frame."""
    annotated = frame.copy()

    colors = {
        'car': (0, 255, 0),
        'truck': (255, 165, 0),
        'bus': (255, 0, 0),
        'motorcycle': (255, 255, 0),
        'auto': (0, 255, 255),
        'unknown': (128, 128, 128),
    }

    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = colors.get(det['type'], (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"{det['type'].upper()} {det['confidence']:.1%}"
        if det['plate']:
            label += f" | {det['plate']}"

        # Draw label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return annotated


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


def generate_detection_feed(source=0):
    """Generator for streaming video with detection overlays."""
    cam = get_camera(source)
    frame_count = 0
    detect_interval = 10  # Detect every N frames for performance

    while True:
        frame = cam.get_frame()
        if frame is not None:
            frame_count += 1

            if frame_count % detect_interval == 0:
                detections = detect_vehicles_in_frame(frame)
                for det in detections:
                    if det.get('plate'):
                        try:
                            save_detection(det)
                        except Exception as e:
                            logger.error(f"Save error: {e}")
                frame = draw_detections(frame, detections)

            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
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

    annotated = draw_detections(frame, detections)
    return annotated, saved_records

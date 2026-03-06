import requests
import json
import logging
import time
from django.conf import settings

logger = logging.getLogger(__name__)

class PlateRecognizerService:
    """
    Service to interact with Plate Recognizer API.
    Docs: https://docs.platerecognizer.com/
    """
    
    API_URL = "https://api.platerecognizer.com/v1/plate-reader/"
    
    @classmethod
    def recognize(cls, frame_bytes):
        """
        Recognize license plate from image bytes using Plate Recognizer API.
        Returns a list of detections compatible with the local format.
        """
        api_key = getattr(settings, 'PLATE_RECOGNIZER_API_KEY', None)
        if not api_key:
            logger.warning("Plate Recognizer API key not found in settings.")
            return []
            
        try:
            response = requests.post(
                cls.API_URL,
                data={'regions': ['in']}, # Optimize for India or leave empty for global
                files={'upload': frame_bytes},
                headers={'Authorization': f'Token {api_key}'},
                timeout=10
            )
            
            if response.status_code != 201:
                logger.error(f"Plate Recognizer API error: {response.status_code} - {response.text}")
                return []
                
            data = response.json()
            results = []
            
            for res in data.get('results', []):
                plate = res.get('plate', '').upper()
                conf = res.get('score', 0.0)
                d_type = res.get('vehicle', {}).get('type', 'car')
                box = res.get('box', {})
                
                # Convert box {top, left, bottom, right} to (x1, y1, x2, y2)
                # Plate Recognizer return absolute coordinates if image size is known
                # but usually it returns coordinates relative to the original image.
                x1, y1 = box.get('xmin', 0), box.get('ymin', 0)
                x2, y2 = box.get('xmax', 0), box.get('ymax', 0)
                
                # Plate Recognizer also returns vehicle box sometimes
                # Here we just use what's available
                results.append({
                    'type': d_type,
                    'plate': plate,
                    'plate_conf': conf,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2), # Use plate box as start if vehicle box not found
                    'plate_bbox': (x1, y1, x2-x1, y2-y1),
                })
            
            return results
        except Exception as e:
            logger.error(f"Plate Recognizer API request failed: {e}")
            return []

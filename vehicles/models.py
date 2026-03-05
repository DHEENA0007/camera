from django.db import models
from django.utils import timezone


class VehicleRecord(models.Model):
    """Individual vehicle detection record."""
    VEHICLE_TYPES = [
        ('car', 'Car'),
        ('truck', 'Truck'),
        ('bus', 'Bus'),
        ('motorcycle', 'Motorcycle'),
        ('auto', 'Auto Rickshaw'),
        ('unknown', 'Unknown'),
    ]

    license_plate = models.CharField(max_length=20, db_index=True)
    vehicle_type = models.CharField(max_length=20, choices=VEHICLE_TYPES, default='unknown')
    confidence = models.FloatField(default=0.0, help_text="Detection confidence score")
    captured_image = models.ImageField(upload_to='captures/%Y/%m/%d/', blank=True, null=True)
    plate_image = models.ImageField(upload_to='plates/%Y/%m/%d/', blank=True, null=True)
    detected_at = models.DateTimeField(default=timezone.now, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-detected_at']
        verbose_name = "Vehicle Record"
        verbose_name_plural = "Vehicle Records"

    def __str__(self):
        return f"{self.license_plate} - {self.get_vehicle_type_display()} ({self.detected_at.strftime('%Y-%m-%d %H:%M')})"


class DailyReport(models.Model):
    """Aggregated daily vehicle count report."""
    date = models.DateField(unique=True, db_index=True)
    total_vehicles = models.IntegerField(default=0)
    total_cars = models.IntegerField(default=0)
    total_trucks = models.IntegerField(default=0)
    total_buses = models.IntegerField(default=0)
    total_motorcycles = models.IntegerField(default=0)
    total_autos = models.IntegerField(default=0)
    total_unknown = models.IntegerField(default=0)
    unique_plates = models.IntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-date']
        verbose_name = "Daily Report"
        verbose_name_plural = "Daily Reports"

    def __str__(self):
        return f"Report: {self.date} - {self.total_vehicles} vehicles"


class CameraConfig(models.Model):
    """Camera configuration settings supporting multiple connection types."""

    CONNECTION_TYPES = [
        ('usb', 'USB / Built-in Webcam'),
        ('wifi', 'WiFi Camera'),
        ('ethernet', 'Ethernet / LAN Camera'),
        ('rtsp', 'RTSP Stream'),
        ('http', 'HTTP / MJPEG Stream'),
        ('onvif', 'ONVIF Camera'),
    ]

    STATUS_CHOICES = [
        ('online', 'Online'),
        ('offline', 'Offline'),
        ('connecting', 'Connecting'),
        ('error', 'Error'),
    ]

    # Basic info
    name = models.CharField(max_length=100, default="Camera")
    location = models.CharField(max_length=200, blank=True, default="",
                                help_text="Physical location e.g. Main Gate, Parking Lot")
    connection_type = models.CharField(max_length=20, choices=CONNECTION_TYPES, default='usb')

    # Connection settings
    source = models.CharField(
        max_length=500, default="0",
        help_text="Camera index (0,1,2..) for USB, or full URL for network cameras"
    )
    ip_address = models.CharField(max_length=100, blank=True, default="",
                                  help_text="IP address for network cameras")
    port = models.IntegerField(blank=True, null=True, help_text="Port number (e.g. 554 for RTSP, 80 for HTTP)")
    username = models.CharField(max_length=100, blank=True, default="",
                                help_text="Camera login username")
    password = models.CharField(max_length=100, blank=True, default="",
                                help_text="Camera login password")
    stream_path = models.CharField(max_length=300, blank=True, default="",
                                   help_text="Stream path e.g. /stream1, /cam/realmonitor")

    # Settings
    resolution = models.CharField(max_length=20, blank=True, default="1280x720",
                                  help_text="Preferred resolution e.g. 1920x1080")
    fps = models.IntegerField(default=15, help_text="Frames per second (5-30)")
    detection_enabled = models.BooleanField(default=True,
                                            help_text="Enable vehicle detection on this camera")

    # Status
    is_active = models.BooleanField(default=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='offline')
    last_seen = models.DateTimeField(blank=True, null=True)
    error_message = models.TextField(blank=True, default="")

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Camera Configuration"
        verbose_name_plural = "Camera Configurations"
        ordering = ['name']

    def __str__(self):
        return f"{self.name} ({self.get_connection_type_display()}) - {'Active' if self.is_active else 'Inactive'}"

    def build_source_url(self):
        """Build the full source URL based on connection type and settings."""
        if self.connection_type == 'usb':
            try:
                return int(self.source)
            except (ValueError, TypeError):
                return 0

        if self.connection_type == 'rtsp':
            auth = ""
            if self.username and self.password:
                auth = f"{self.username}:{self.password}@"
            port = self.port or 554
            path = self.stream_path or "/stream1"
            return f"rtsp://{auth}{self.ip_address}:{port}{path}"

        if self.connection_type == 'http':
            port = self.port or 80
            path = self.stream_path or "/video"
            protocol = "https" if port == 443 else "http"
            auth = ""
            if self.username and self.password:
                auth = f"{self.username}:{self.password}@"
            return f"{protocol}://{auth}{self.ip_address}:{port}{path}"

        if self.connection_type in ('wifi', 'ethernet', 'onvif'):
            # Default to RTSP for WiFi/Ethernet/ONVIF cameras
            auth = ""
            if self.username and self.password:
                auth = f"{self.username}:{self.password}@"
            port = self.port or 554
            path = self.stream_path or "/stream1"
            return f"rtsp://{auth}{self.ip_address}:{port}{path}"

        return self.source

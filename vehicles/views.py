import json
import cv2
import base64
from datetime import date, timedelta
from django.shortcuts import render, redirect, get_object_or_404
from django.http import StreamingHttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.db.models import Count, Sum
from django.db.models.functions import TruncDate
from django.utils import timezone

from .models import VehicleRecord, DailyReport, CameraConfig
from .detection_service import (
    generate_video_feed,
    generate_detection_feed,
    capture_snapshot,
    get_camera,
    stop_camera,
)


def dashboard(request):
    """Main dashboard view."""
    today = date.today()
    today_records = VehicleRecord.objects.filter(detected_at__date=today)

    # Get or create today's report
    report, _ = DailyReport.objects.get_or_create(date=today)

    # Last 7 days reports
    week_ago = today - timedelta(days=7)
    weekly_reports = DailyReport.objects.filter(date__gte=week_ago).order_by('date')

    # Recent detections
    recent_detections = VehicleRecord.objects.all()[:10]

    # Camera configs
    cameras = CameraConfig.objects.filter(is_active=True)

    context = {
        'today_report': report,
        'today_count': today_records.count(),
        'today_unique': today_records.exclude(
            license_plate='UNREADABLE'
        ).values('license_plate').distinct().count(),
        'weekly_reports': weekly_reports,
        'recent_detections': recent_detections,
        'cameras': cameras,
        'today_date': today,
    }
    return render(request, 'vehicles/dashboard.html', context)


def live_feed(request):
    """Live camera feed page."""
    cameras = CameraConfig.objects.filter(is_active=True)
    source = request.GET.get('source', '0')
    context = {
        'cameras': cameras,
        'current_source': source,
    }
    return render(request, 'vehicles/live_feed.html', context)


def video_stream(request):
    """Raw video stream endpoint."""
    source = request.GET.get('source', '0')
    return StreamingHttpResponse(
        generate_video_feed(source),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


def detection_stream(request):
    """Video stream with detection overlays."""
    source = request.GET.get('source', '0')
    return StreamingHttpResponse(
        generate_detection_feed(source),
        content_type='multipart/x-mixed-replace; boundary=frame'
    )


@csrf_exempt
def take_snapshot(request):
    """Capture single frame and detect vehicles."""
    if request.method == 'POST':
        source = request.POST.get('source', '0')
        annotated_frame, records = capture_snapshot(source)

        result = {
            'success': annotated_frame is not None,
            'detections': [],
            'snapshot': None,
        }

        if annotated_frame is not None:
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            img_base64 = base64.b64encode(buffer).decode('utf-8')
            result['snapshot'] = f"data:image/jpeg;base64,{img_base64}"

        for record in records:
            result['detections'].append({
                'id': record.id,
                'plate': record.license_plate,
                'type': record.get_vehicle_type_display(),
                'confidence': f"{record.confidence:.1%}",
                'time': record.detected_at.strftime('%H:%M:%S'),
            })

        return JsonResponse(result)

    return JsonResponse({'error': 'POST required'}, status=405)


def daily_records(request):
    """View daily records and reports."""
    selected_date = request.GET.get('date', date.today().isoformat())
    try:
        selected_date = date.fromisoformat(selected_date)
    except ValueError:
        selected_date = date.today()

    records = VehicleRecord.objects.filter(detected_at__date=selected_date)

    # Vehicle type breakdown
    type_counts = records.values('vehicle_type').annotate(count=Count('id'))

    # Get daily report
    report = DailyReport.objects.filter(date=selected_date).first()

    # All reports for the sidebar calendar
    all_reports = DailyReport.objects.all()[:30]

    context = {
        'selected_date': selected_date,
        'records': records,
        'type_counts': type_counts,
        'report': report,
        'all_reports': all_reports,
        'total_count': records.count(),
        'unique_plates': records.exclude(
            license_plate='UNREADABLE'
        ).values('license_plate').distinct().count(),
    }
    return render(request, 'vehicles/daily_records.html', context)


def vehicle_search(request):
    """Search for vehicles by plate number."""
    query = request.GET.get('q', '').strip().upper()
    results = []

    if query:
        results = VehicleRecord.objects.filter(
            license_plate__icontains=query
        ).order_by('-detected_at')[:50]

    context = {
        'query': query,
        'results': results,
        'result_count': len(results),
    }
    return render(request, 'vehicles/search.html', context)


def api_stats(request):
    """API endpoint for dashboard stats."""
    today = date.today()
    today_records = VehicleRecord.objects.filter(detected_at__date=today)

    # Last 7 days data
    week_ago = today - timedelta(days=7)
    daily_data = (
        VehicleRecord.objects
        .filter(detected_at__date__gte=week_ago)
        .annotate(day=TruncDate('detected_at'))
        .values('day')
        .annotate(count=Count('id'))
        .order_by('day')
    )

    # Recent detections
    recent = VehicleRecord.objects.all()[:5]

    return JsonResponse({
        'today_total': today_records.count(),
        'today_unique': today_records.exclude(
            license_plate='UNREADABLE'
        ).values('license_plate').distinct().count(),
        'daily_chart': [
            {'date': d['day'].isoformat(), 'count': d['count']}
            for d in daily_data
        ],
        'recent': [
            {
                'plate': r.license_plate,
                'type': r.get_vehicle_type_display(),
                'time': r.detected_at.strftime('%H:%M:%S'),
            }
            for r in recent
        ],
    })


@csrf_exempt
def camera_control(request):
    """Start/stop camera."""
    if request.method == 'POST':
        action = request.POST.get('action')
        source = request.POST.get('source', '0')

        if action == 'start':
            cam = get_camera(source)
            return JsonResponse({'status': 'started', 'active': cam.is_active()})
        elif action == 'stop':
            stop_camera()
            return JsonResponse({'status': 'stopped'})

    return JsonResponse({'error': 'Invalid request'}, status=400)


# ===========================
# Camera Management Views
# ===========================

def camera_list(request):
    """Camera integration & management page."""
    cameras = CameraConfig.objects.all()
    connection_types = CameraConfig.CONNECTION_TYPES

    context = {
        'cameras': cameras,
        'connection_types': connection_types,
        'total_cameras': cameras.count(),
        'active_cameras': cameras.filter(is_active=True).count(),
        'online_cameras': cameras.filter(status='online').count(),
    }
    return render(request, 'vehicles/cameras.html', context)


@csrf_exempt
def camera_add(request):
    """Add a new camera."""
    if request.method == 'POST':
        cam = CameraConfig()
        cam.name = request.POST.get('name', 'New Camera')
        cam.location = request.POST.get('location', '')
        cam.connection_type = request.POST.get('connection_type', 'usb')
        cam.source = request.POST.get('source', '0')
        cam.ip_address = request.POST.get('ip_address', '')
        cam.port = request.POST.get('port') or None
        if cam.port:
            cam.port = int(cam.port)
        cam.username = request.POST.get('username', '')
        cam.password = request.POST.get('password', '')
        cam.stream_path = request.POST.get('stream_path', '')
        cam.resolution = request.POST.get('resolution', '1280x720')
        cam.fps = int(request.POST.get('fps', 15))
        cam.detection_enabled = request.POST.get('detection_enabled') == 'on'
        cam.is_active = True
        cam.status = 'offline'
        cam.save()

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'id': cam.id,
                'name': cam.name,
                'message': f'Camera "{cam.name}" added successfully!'
            })
        return redirect('vehicles:cameras')

    return JsonResponse({'error': 'POST required'}, status=405)


@csrf_exempt
def camera_edit(request, camera_id):
    """Edit an existing camera."""
    cam = get_object_or_404(CameraConfig, id=camera_id)

    if request.method == 'POST':
        cam.name = request.POST.get('name', cam.name)
        cam.location = request.POST.get('location', cam.location)
        cam.connection_type = request.POST.get('connection_type', cam.connection_type)
        cam.source = request.POST.get('source', cam.source)
        cam.ip_address = request.POST.get('ip_address', cam.ip_address)
        cam.port = request.POST.get('port') or None
        if cam.port:
            cam.port = int(cam.port)
        cam.username = request.POST.get('username', cam.username)
        cam.password = request.POST.get('password', cam.password)
        cam.stream_path = request.POST.get('stream_path', cam.stream_path)
        cam.resolution = request.POST.get('resolution', cam.resolution)
        cam.fps = int(request.POST.get('fps', cam.fps))
        cam.detection_enabled = request.POST.get('detection_enabled') == 'on'
        cam.save()

        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({
                'success': True,
                'message': f'Camera "{cam.name}" updated successfully!'
            })
        return redirect('vehicles:cameras')

    # GET - return camera data as JSON
    return JsonResponse({
        'id': cam.id,
        'name': cam.name,
        'location': cam.location,
        'connection_type': cam.connection_type,
        'source': cam.source,
        'ip_address': cam.ip_address,
        'port': cam.port,
        'username': cam.username,
        'password': cam.password,
        'stream_path': cam.stream_path,
        'resolution': cam.resolution,
        'fps': cam.fps,
        'detection_enabled': cam.detection_enabled,
        'is_active': cam.is_active,
        'status': cam.status,
    })


@csrf_exempt
def camera_delete(request, camera_id):
    """Delete a camera."""
    cam = get_object_or_404(CameraConfig, id=camera_id)
    if request.method == 'POST':
        name = cam.name
        cam.delete()
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'success': True, 'message': f'Camera "{name}" deleted.'})
        return redirect('vehicles:cameras')
    return JsonResponse({'error': 'POST required'}, status=405)


@csrf_exempt
def camera_toggle(request, camera_id):
    """Toggle camera active status."""
    cam = get_object_or_404(CameraConfig, id=camera_id)
    if request.method == 'POST':
        cam.is_active = not cam.is_active
        if not cam.is_active:
            cam.status = 'offline'
        cam.save()
        return JsonResponse({
            'success': True,
            'is_active': cam.is_active,
            'status': cam.status,
        })
    return JsonResponse({'error': 'POST required'}, status=405)


@csrf_exempt
def camera_test(request, camera_id):
    """Test camera connection and return a snapshot."""
    cam = get_object_or_404(CameraConfig, id=camera_id)

    if request.method == 'POST':
        source = cam.build_source_url()
        success = False
        snapshot = None
        error_msg = ""

        try:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    success = True
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    img_base64 = base64.b64encode(buffer).decode('utf-8')
                    snapshot = f"data:image/jpeg;base64,{img_base64}"

                    cam.status = 'online'
                    cam.last_seen = timezone.now()
                    cam.error_message = ""
                else:
                    error_msg = "Camera opened but failed to read frame"
                    cam.status = 'error'
                    cam.error_message = error_msg
                cap.release()
            else:
                error_msg = f"Cannot open camera source: {source}"
                cam.status = 'error'
                cam.error_message = error_msg
        except Exception as e:
            error_msg = str(e)
            cam.status = 'error'
            cam.error_message = error_msg

        cam.save()

        return JsonResponse({
            'success': success,
            'snapshot': snapshot,
            'status': cam.status,
            'error': error_msg,
            'source_url': str(source) if cam.connection_type != 'usb' else f"USB Camera {source}",
        })

    return JsonResponse({'error': 'POST required'}, status=405)


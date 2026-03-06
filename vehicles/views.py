import json
import cv2
import numpy as np
import base64
import socket
import subprocess
import threading
import ipaddress
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
    detect_vehicles_in_frame,
    draw_highway_hud,
    vehicle_tracker
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
    source = request.GET.get('source', '')
    camera_id = request.GET.get('camera_id', '')
    
    # If camera_id is provided, look up that camera's source URL
    if camera_id:
        try:
            cam = CameraConfig.objects.get(id=int(camera_id))
            source = str(cam.build_source_url())
        except (CameraConfig.DoesNotExist, ValueError):
            source = '0'
    # If no source specified, use the first active camera's source URL
    elif not source and cameras.exists():
        first_cam = cameras.first()
        source = str(first_cam.build_source_url())
    elif not source:
        source = '0'
    
    context = {
        'cameras': cameras,
        'current_source': source,
    }
    return render(request, 'vehicles/live_feed.html', context)


@csrf_exempt
def test_image_upload(request):
    """API endpoint to test OpenALPR with an uploaded static image."""
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            image_file = request.FILES['image']
            
            # Read image to numpy buffer
            file_bytes = np.frombuffer(image_file.read(), np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if frame is None:
                return JsonResponse({"error": "Failed to decode image"}, status=400)
                
            # Run detection
            detections = detect_vehicles_in_frame(frame)
            
            # Just draw the results purely for static visual display
            annotated = frame.copy()
            for det in detections:
                if det.get('plate_bbox'):
                    px, py, pw, ph = det['plate_bbox']
                    cv2.rectangle(annotated, (px, py), (px+pw, py+ph), (0, 255, 255), 2)
                    label = f"{det.get('plate', 'UNKNOWN')} ({det.get('confidence', 0.0):.2f})"
                    cv2.putText(annotated, label, (px, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Encode back to jpeg for the frontend
            ret, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return JsonResponse({"error": "Failed to encode resulting image"}, status=500)
                
            b64_image = base64.b64encode(jpeg.tobytes()).decode('utf-8')
            
            response_data = {
                "success": True,
                "image_b64": b64_image,
                "detections": [
                    {
                        "plate": d.get("plate"),
                        "confidence": d.get("confidence", 0.0)
                    } for d in detections
                ]
            }
            return JsonResponse(response_data)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)
            
    return JsonResponse({"error": "Invalid request"}, status=400)


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
            # Force TCP for RTSP streams to avoid UDP timeout issues
            if isinstance(source, str) and source.startswith('rtsp://'):
                import os
                os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            else:
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


@csrf_exempt
def camera_scan(request):
    """Scan the network for cameras (auto-detect)."""
    if request.method != 'POST':
        return JsonResponse({'error': 'POST required'}, status=405)

    scan_type = request.POST.get('scan_type', 'all')  # 'wifi', 'ethernet', 'usb', 'all'

    if scan_type == 'usb':
        discovered = []
        import platform as _plat
        if _plat.system().lower() == 'windows':
            # On Windows, probe camera indices 0-9 using OpenCV
            for index in range(10):
                try:
                    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
                    if cap.isOpened():
                        name = f"USB Camera {index}"
                        discovered.append({
                            'ip': str(index),
                            'hostname': '',
                            'camera_type': 'USB Camera',
                            'open_ports': [],
                            'suggested_port': '',
                            'stream_path': name,
                            'is_usb': True
                        })
                        cap.release()
                except Exception:
                    pass
        else:
            # On Linux, use /dev/video* devices
            import glob
            video_devices = sorted(glob.glob('/dev/video*'))
            for dev in video_devices:
                try:
                    index = int(dev.replace('/dev/video', ''))
                    name = f"USB Camera {index}"
                    try:
                        res = subprocess.run(['v4l2-ctl', '-d', dev, '--info'], capture_output=True, text=True, timeout=1)
                        if res.returncode == 0:
                            for line in res.stdout.split('\n'):
                                if "Card type" in line:
                                    name = line.split(':', 1)[1].strip()
                                    break
                    except Exception:
                        pass
                    discovered.append({
                        'ip': str(index),
                        'hostname': '',
                        'camera_type': 'USB Camera',
                        'open_ports': [],
                        'suggested_port': '',
                        'stream_path': name,
                        'is_usb': True
                    })
                except ValueError:
                    pass
        
        return JsonResponse({
            'success': True,
            'cameras': discovered,
            'subnets_scanned': [],
            'total_hosts_checked': len(discovered),
            'cameras_found': len(discovered),
        })

    # Common camera ports to probe
    CAMERA_PORTS = [554, 8554, 80, 8080, 443, 37777, 34567, 8000, 8899]

    import platform
    import re
    is_windows = platform.system().lower() == 'windows'

    def get_local_subnets():
        """Get local network subnets (cross-platform: Windows & Linux)."""
        subnets = []
        try:
            if is_windows:
                # Use ipconfig on Windows
                result = subprocess.run(
                    ['ipconfig'],
                    capture_output=True, text=True, timeout=5
                )
                # Parse ipconfig output for IPv4 addresses and subnet masks
                lines = result.stdout.split('\n')
                current_ip = None
                for line in lines:
                    line = line.strip()
                    # Match IPv4 Address line
                    ip_match = re.search(r'IPv4 Address[.\s]*:\s*(\d+\.\d+\.\d+\.\d+)', line)
                    if ip_match:
                        current_ip = ip_match.group(1)
                        continue
                    # Match Subnet Mask line (comes right after IPv4 address)
                    mask_match = re.search(r'Subnet Mask[.\s]*:\s*(\d+\.\d+\.\d+\.\d+)', line)
                    if mask_match and current_ip:
                        mask = mask_match.group(1)
                        if current_ip.startswith('127.'):
                            current_ip = None
                            continue
                        try:
                            network = ipaddress.IPv4Network(f"{current_ip}/{mask}", strict=False)
                            subnets.append({
                                'network': str(network),
                                'ip': current_ip,
                                'interface': 'unknown'
                            })
                        except Exception:
                            pass
                        current_ip = None
            else:
                # Use ip command on Linux
                result = subprocess.run(
                    ['ip', '-4', 'addr', 'show'],
                    capture_output=True, text=True, timeout=5
                )
                for match in re.finditer(r'inet (\d+\.\d+\.\d+\.\d+)/(\d+)', result.stdout):
                    ip_str, prefix = match.group(1), int(match.group(2))
                    if ip_str.startswith('127.'):
                        continue
                    try:
                        network = ipaddress.IPv4Network(f"{ip_str}/{prefix}", strict=False)
                        subnets.append({
                            'network': str(network),
                            'ip': ip_str,
                            'interface': 'unknown'
                        })
                    except Exception:
                        pass
        except Exception:
            pass

        # Fallback: use socket to get local IP if no subnets found
        if not subnets:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.connect(('8.8.8.8', 80))
                local_ip = s.getsockname()[0]
                s.close()
                if not local_ip.startswith('127.'):
                    network = ipaddress.IPv4Network(f"{local_ip}/24", strict=False)
                    subnets.append({
                        'network': str(network),
                        'ip': local_ip,
                        'interface': 'auto-detected'
                    })
            except Exception:
                pass

        return subnets

    def get_arp_hosts():
        """Read ARP table to find known hosts (cross-platform)."""
        hosts = set()
        try:
            if is_windows:
                # Use 'arp -a' on Windows
                result = subprocess.run(
                    ['arp', '-a'],
                    capture_output=True, text=True, timeout=5
                )
                for match in re.finditer(r'(\d+\.\d+\.\d+\.\d+)', result.stdout):
                    ip = match.group(1)
                    if not ip.startswith('127.') and not ip.endswith('.255'):
                        hosts.add(ip)
            else:
                # Read /proc/net/arp on Linux
                with open('/proc/net/arp', 'r') as f:
                    lines = f.readlines()[1:]  # Skip header
                    for line in lines:
                        parts = line.split()
                        if len(parts) >= 4 and parts[2] != '0x0':
                            ip = parts[0]
                            if not ip.startswith('127.'):
                                hosts.add(ip)
        except Exception:
            pass
        return hosts

    def ping_sweep(subnet_str, timeout=1):
        """Quick ping sweep to populate ARP table (cross-platform)."""
        try:
            network = ipaddress.IPv4Network(subnet_str, strict=False)
            # Only sweep /24 or smaller to avoid huge scans
            if network.prefixlen < 24:
                network = ipaddress.IPv4Network(
                    f"{str(network.network_address).rsplit('.', 1)[0]}.0/24",
                    strict=False
                )

            if is_windows:
                ping_cmd = lambda h: ['ping', '-n', '1', '-w', '1000', str(h)]
            else:
                ping_cmd = lambda h: ['ping', '-c', '1', '-W', '1', str(h)]

            threads = []
            for host in list(network.hosts())[:254]:
                t = threading.Thread(
                    target=lambda h: subprocess.run(
                        ping_cmd(h),
                        capture_output=True, timeout=3
                    ),
                    args=(host,)
                )
                t.daemon = True
                threads.append(t)

            # Run in batches of 50
            for i in range(0, len(threads), 50):
                batch = threads[i:i+50]
                for t in batch:
                    t.start()
                for t in batch:
                    t.join(timeout=3)
        except Exception:
            pass

    def probe_port(ip, port, timeout=1.5):
        """Check if a port is open on the given IP."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((ip, port))
            sock.close()
            return result == 0
        except Exception:
            return False

    def identify_camera(ip, open_ports):
        """Try to identify camera type based on open ports."""
        camera_type = 'IP Camera'
        stream_path = '/stream1'
        suggested_port = 554

        if 554 in open_ports or 8554 in open_ports:
            camera_type = 'RTSP Camera'
            suggested_port = 554 if 554 in open_ports else 8554
            stream_path = '/stream1'
        elif 80 in open_ports or 8080 in open_ports:
            camera_type = 'HTTP/Web Camera'
            suggested_port = 80 if 80 in open_ports else 8080
            stream_path = '/video'
        elif 37777 in open_ports:
            camera_type = 'Dahua Camera'
            suggested_port = 554
            stream_path = '/cam/realmonitor?channel=1&subtype=0'
        elif 34567 in open_ports:
            camera_type = 'XMEye Camera'
            suggested_port = 554
            stream_path = '/user=admin&password=&channel=1&stream=0.sdp'

        # Try to get hostname
        hostname = ''
        try:
            hostname = socket.gethostbyaddr(ip)[0]
        except Exception:
            pass

        return {
            'ip': ip,
            'hostname': hostname,
            'camera_type': camera_type,
            'open_ports': sorted(open_ports),
            'suggested_port': suggested_port,
            'stream_path': stream_path,
        }

    # --- Main scan logic ---
    discovered = []
    subnets = get_local_subnets()

    if not subnets:
        return JsonResponse({
            'success': False,
            'error': 'No network interfaces found.',
            'cameras': []
        })

    # Filter subnets based on scan type
    if scan_type == 'ethernet':
        # Prefer 192.168.x.x subnets on ethernet
        eth_subnets = [s for s in subnets if '192.168.' in s['ip'] or '10.' in s['ip']]
        if eth_subnets:
            subnets = eth_subnets
    elif scan_type == 'wifi':
        pass  # Scan all subnets

    # Step 1: Ping sweep to populate ARP table
    for subnet_info in subnets[:3]:  # Limit to 3 subnets max
        ping_sweep(subnet_info['network'])

    # Step 2: Get all known hosts from ARP
    all_hosts = get_arp_hosts()

    # Also add common default camera IPs
    common_camera_ips = [
        '192.168.1.1', '192.168.1.10', '192.168.1.64', '192.168.1.100',
        '192.168.1.108', '192.168.1.168', '192.168.0.10', '192.168.0.64',
        '192.168.0.100', '192.168.0.108',
    ]
    for ip in common_camera_ips:
        all_hosts.add(ip)

    # Get own IPs to exclude
    own_ips = {s['ip'] for s in subnets}

    # Step 3: Probe camera ports on discovered hosts
    scan_results = {}
    probe_threads = []

    def probe_host(ip):
        open_ports = []
        for port in CAMERA_PORTS:
            if probe_port(ip, port):
                open_ports.append(port)
        if open_ports:
            scan_results[ip] = open_ports

    hosts_to_scan = [h for h in all_hosts if h not in own_ips]

    for ip in hosts_to_scan:
        t = threading.Thread(target=probe_host, args=(ip,))
        t.daemon = True
        probe_threads.append(t)

    # Run port probes in batches
    for i in range(0, len(probe_threads), 30):
        batch = probe_threads[i:i+30]
        for t in batch:
            t.start()
        for t in batch:
            t.join(timeout=5)

    # Step 4: Build results
    for ip, open_ports in scan_results.items():
        camera_info = identify_camera(ip, open_ports)
        discovered.append(camera_info)

    # Sort by IP
    discovered.sort(key=lambda x: [int(p) for p in x['ip'].split('.')])

    return JsonResponse({
        'success': True,
        'cameras': discovered,
        'subnets_scanned': [s['network'] for s in subnets[:3]],
        'total_hosts_checked': len(hosts_to_scan),
        'cameras_found': len(discovered),
    })


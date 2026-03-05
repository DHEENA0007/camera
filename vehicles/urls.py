from django.urls import path
from . import views

app_name = 'vehicles'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    path('live/', views.live_feed, name='live_feed'),
    path('records/', views.daily_records, name='daily_records'),
    path('search/', views.vehicle_search, name='search'),

    # Camera management
    path('cameras/', views.camera_list, name='cameras'),
    path('cameras/add/', views.camera_add, name='camera_add'),
    path('cameras/<int:camera_id>/edit/', views.camera_edit, name='camera_edit'),
    path('cameras/<int:camera_id>/delete/', views.camera_delete, name='camera_delete'),
    path('cameras/<int:camera_id>/toggle/', views.camera_toggle, name='camera_toggle'),
    path('cameras/<int:camera_id>/test/', views.camera_test, name='camera_test'),
    path('cameras/scan/', views.camera_scan, name='camera_scan'),

    # Streaming endpoints
    path('stream/video/', views.video_stream, name='video_stream'),
    path('stream/detection/', views.detection_stream, name='detection_stream'),

    # API endpoints
    path('api/snapshot/', views.take_snapshot, name='snapshot'),
    path('api/stats/', views.api_stats, name='api_stats'),
    path('api/camera/', views.camera_control, name='camera_control'),
]

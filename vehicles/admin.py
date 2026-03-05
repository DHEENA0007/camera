from django.contrib import admin
from .models import VehicleRecord, DailyReport, CameraConfig


@admin.register(VehicleRecord)
class VehicleRecordAdmin(admin.ModelAdmin):
    list_display = ('license_plate', 'vehicle_type', 'confidence', 'detected_at')
    list_filter = ('vehicle_type', 'detected_at')
    search_fields = ('license_plate',)
    readonly_fields = ('created_at',)
    date_hierarchy = 'detected_at'


@admin.register(DailyReport)
class DailyReportAdmin(admin.ModelAdmin):
    list_display = ('date', 'total_vehicles', 'unique_plates', 'total_cars', 'total_trucks', 'total_buses')
    list_filter = ('date',)
    readonly_fields = ('created_at', 'updated_at')


@admin.register(CameraConfig)
class CameraConfigAdmin(admin.ModelAdmin):
    list_display = ('name', 'source', 'is_active')
    list_filter = ('is_active',)

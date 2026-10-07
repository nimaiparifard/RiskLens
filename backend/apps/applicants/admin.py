from django.contrib import admin

from apps.applicants.models import Applicant, Application


@admin.register(Applicant)
class ApplicantAdmin(admin.ModelAdmin):
    list_display = ("full_name", "external_id", "email", "created_at")
    search_fields = ("full_name", "external_id", "email")


@admin.register(Application)
class ApplicationAdmin(admin.ModelAdmin):
    list_display = (
        "reference",
        "applicant",
        "amt_credit",
        "status",
        "latest_grade",
        "latest_probability",
        "created_at",
    )
    list_filter = ("status", "latest_grade", "is_override")
    search_fields = ("reference", "applicant__full_name")

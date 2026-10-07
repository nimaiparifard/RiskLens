from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from apps.users.models import User


@admin.register(User)
class RiskLensUserAdmin(UserAdmin):
    fieldsets = (*UserAdmin.fieldsets, ("RiskLens", {"fields": ("role",)}))
    list_display = ("username", "email", "role", "is_active", "last_login")
    list_filter = ("role", "is_active")

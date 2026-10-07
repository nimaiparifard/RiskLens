from rest_framework import serializers, viewsets

from apps.audit.models import AuditLog
from apps.core.permissions import IsOfficer


class AuditLogSerializer(serializers.ModelSerializer):
    actor = serializers.CharField(source="actor.username", default=None, read_only=True)

    class Meta:
        model = AuditLog
        fields = [
            "id",
            "created_at",
            "actor",
            "action",
            "target_type",
            "target_id",
            "payload",
            "request_id",
        ]


class AuditLogViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = AuditLog.objects.select_related("actor")
    serializer_class = AuditLogSerializer
    permission_classes = [IsOfficer]
    filterset_fields = ["action", "target_type", "target_id"]

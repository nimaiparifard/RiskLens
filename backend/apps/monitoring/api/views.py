from drf_spectacular.utils import extend_schema
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.core.permissions import IsAdminRole
from apps.monitoring import services
from apps.monitoring.models import DriftReport


class DriftReportSerializer(serializers.ModelSerializer):
    report_url = serializers.SerializerMethodField()

    class Meta:
        model = DriftReport
        fields = [
            "id",
            "created_at",
            "task",
            "model_version",
            "window_start",
            "window_end",
            "n_samples",
            "feature_psi",
            "prediction_psi",
            "max_psi",
            "drifted_features",
            "status",
            "retrain_triggered",
            "report_url",
            "error",
        ]

    def get_report_url(self, obj) -> str | None:
        return obj.report_file.url if obj.report_file else None


class RunDriftSerializer(serializers.Serializer):
    task = serializers.ChoiceField(choices=["credit", "fraud"], default="credit")
    hours = serializers.IntegerField(required=False, min_value=1, max_value=24 * 90)


class DriftReportViewSet(viewsets.ReadOnlyModelViewSet):
    """GET /api/monitoring/drift/?task=credit   POST /api/monitoring/drift/run/ (admin)"""

    queryset = DriftReport.objects.all()
    serializer_class = DriftReportSerializer
    filterset_fields = ["task", "status"]

    @extend_schema(request=RunDriftSerializer, responses=DriftReportSerializer)
    @action(detail=False, methods=["post"], permission_classes=[IsAdminRole])
    def run(self, request):
        ser = RunDriftSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        report = services.run_drift_report(**ser.validated_data)
        return Response(DriftReportSerializer(report).data, status=status.HTTP_201_CREATED)

from drf_spectacular.utils import extend_schema
from rest_framework import mixins, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.applicants import selectors, services
from apps.applicants.api.filters import ApplicationFilter
from apps.applicants.api.serializers import (
    ApplicationCreateSerializer,
    ApplicationDetailSerializer,
    ApplicationListSerializer,
    DecisionSerializer,
)
from apps.audit.api.views import AuditLogSerializer
from apps.audit.models import AuditLog
from apps.core.permissions import IsOfficer, ReadOnlyOrOfficer
from apps.scoring import services as scoring
from apps.scoring.api.serializers import PredictionSerializer


class ApplicationViewSet(
    mixins.ListModelMixin,
    mixins.RetrieveModelMixin,
    mixins.CreateModelMixin,
    viewsets.GenericViewSet,
):
    """Loan applications.

    GET  /api/applications/?status=pending&grade=C
    POST /api/applications/
    POST /api/applications/{id}/score/       -> calls the model server, stores a Prediction
    POST /api/applications/{id}/decision/    -> officer approve / reject / override
    GET  /api/applications/{id}/audit/       -> audit trail
    GET  /api/applications/stats/            -> dashboard numbers
    """

    permission_classes = [ReadOnlyOrOfficer]
    throttle_scope = None  # set per action (expensive model calls)
    filterset_class = ApplicationFilter
    search_fields = ["reference", "applicant__full_name", "applicant__external_id"]
    ordering_fields = ["created_at", "latest_probability", "amt_credit"]

    def get_queryset(self):
        return selectors.application_list()

    def get_serializer_class(self):
        if self.action == "list":
            return ApplicationListSerializer
        if self.action == "create":
            return ApplicationCreateSerializer
        if self.action == "decision":
            return DecisionSerializer
        return ApplicationDetailSerializer

    def create(self, request, *args, **kwargs):
        ser = ApplicationCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        app = services.create_application(user=request.user, **ser.validated_data)
        return Response(ApplicationDetailSerializer(app).data, status=status.HTTP_201_CREATED)

    @extend_schema(request=None, responses=PredictionSerializer)
    @action(detail=True, methods=["post"], permission_classes=[IsOfficer], throttle_scope="scoring")
    def score(self, request, pk=None):
        pred = scoring.score_application(self.get_object(), request.user)
        return Response(PredictionSerializer(pred).data, status=status.HTTP_201_CREATED)

    @extend_schema(request=DecisionSerializer, responses=ApplicationDetailSerializer)
    @action(detail=True, methods=["post"], permission_classes=[IsOfficer])
    def decision(self, request, pk=None):
        ser = DecisionSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        app = scoring.decide(self.get_object(), request.user, **ser.validated_data)
        return Response(ApplicationDetailSerializer(app).data)

    @extend_schema(responses=AuditLogSerializer(many=True))
    @action(detail=True, methods=["get"])
    def audit(self, request, pk=None):
        app = self.get_object()
        logs = AuditLog.objects.filter(
            target_type="application", target_id=str(app.pk)
        ).select_related("actor")
        return Response(AuditLogSerializer(logs, many=True).data)

    @extend_schema(responses=PredictionSerializer(many=True))
    @action(detail=True, methods=["get"])
    def predictions(self, request, pk=None):
        return Response(PredictionSerializer(self.get_object().predictions.all(), many=True).data)

    @action(detail=False, methods=["get"])
    def stats(self, request):
        return Response(selectors.application_stats())

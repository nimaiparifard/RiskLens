from django.conf import settings
from drf_spectacular.types import OpenApiTypes
from drf_spectacular.utils import extend_schema
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.audit.models import AuditLog
from apps.audit.services import log_action
from apps.core.exceptions import DomainError
from apps.core.permissions import IsAdminRole
from apps.mlmodels.clients import MlflowRegistryClient
from apps.scoring.clients import get_serving_client


class ModelsView(APIView):
    """GET /api/models/ - registry versions + metrics (champion vs challenger) and what serving runs."""

    @extend_schema(responses=OpenApiTypes.OBJECT)
    def get(self, request):
        registry = MlflowRegistryClient()
        out = {}
        for task, name in (
            ("credit", settings.CREDIT_MODEL_NAME),
            ("fraud", settings.FRAUD_MODEL_NAME),
        ):
            try:
                out[task] = registry.model_summary(name)
            except DomainError as exc:
                out[task] = {"name": name, "error": exc.message, "versions": [], "aliases": {}}
        try:
            serving = get_serving_client().model_info()
        except DomainError as exc:
            serving = {"error": exc.message}
        return Response({"registry": out, "serving": serving})


class ReloadView(APIView):
    """POST /api/models/reload/ - ask serving to reload the champion now (admin)."""

    permission_classes = [IsAdminRole]

    @extend_schema(request=OpenApiTypes.OBJECT, responses=OpenApiTypes.OBJECT)
    def post(self, request):
        task = request.data.get("task")
        result = get_serving_client().reload(task)
        log_action(request.user, AuditLog.Action.MODEL_RELOADED, f"model:{task or 'all'}", result)
        return Response(result)

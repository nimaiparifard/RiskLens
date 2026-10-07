from rest_framework import viewsets

from apps.scoring.api.serializers import PredictionSerializer
from apps.scoring.models import Prediction


class PredictionViewSet(viewsets.ReadOnlyModelViewSet):
    queryset = Prediction.objects.all()
    serializer_class = PredictionSerializer
    filterset_fields = ["application", "grade", "decision", "model_version"]
    ordering_fields = ["created_at", "probability"]

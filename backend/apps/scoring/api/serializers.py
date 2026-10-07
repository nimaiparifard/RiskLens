from rest_framework import serializers

from apps.scoring.models import Prediction


class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = [
            "id",
            "application",
            "created_at",
            "model_name",
            "model_version",
            "probability",
            "score_points",
            "grade",
            "decision",
            "threshold",
            "shap_reasons",
            "latency_ms",
            "shadow",
        ]
        read_only_fields = fields

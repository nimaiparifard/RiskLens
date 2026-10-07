from rest_framework import serializers

from apps.applicants.models import Applicant, Application
from apps.scoring.api.serializers import PredictionSerializer

REQUIRED_FEATURES = ("AMT_INCOME_TOTAL", "AMT_CREDIT", "DAYS_BIRTH")


class ApplicantSerializer(serializers.ModelSerializer):
    class Meta:
        model = Applicant
        fields = ["id", "external_id", "full_name", "email", "phone"]


class ApplicationListSerializer(serializers.ModelSerializer):
    applicant_name = serializers.CharField(source="applicant.full_name", read_only=True)

    class Meta:
        model = Application
        fields = [
            "id",
            "reference",
            "applicant_name",
            "contract_type",
            "amt_credit",
            "amt_income_total",
            "status",
            "latest_grade",
            "latest_probability",
            "recommendation",
            "is_override",
            "created_at",
        ]


class ApplicationDetailSerializer(serializers.ModelSerializer):
    applicant = ApplicantSerializer(read_only=True)
    latest_prediction = serializers.SerializerMethodField()
    decided_by = serializers.CharField(source="decided_by.username", default=None, read_only=True)

    class Meta:
        model = Application
        fields = [
            "id",
            "reference",
            "applicant",
            "contract_type",
            "amt_credit",
            "amt_annuity",
            "amt_income_total",
            "features",
            "history",
            "status",
            "latest_grade",
            "latest_probability",
            "recommendation",
            "decided_by",
            "decided_at",
            "decision_comment",
            "is_override",
            "created_at",
            "updated_at",
            "latest_prediction",
        ]

    def get_latest_prediction(self, obj):
        pred = obj.predictions.order_by("-created_at").first()
        return PredictionSerializer(pred).data if pred else None


class ApplicationCreateSerializer(serializers.Serializer):
    applicant = ApplicantSerializer()
    features = serializers.DictField(
        help_text="Home Credit application fields (AMT_CREDIT, DAYS_BIRTH, ...)"
    )
    history = serializers.DictField(
        required=False, default=dict, help_text="pre-computed history aggregates"
    )

    def validate_features(self, value):
        missing = [k for k in REQUIRED_FEATURES if value.get(k) in (None, "")]
        if missing:
            raise serializers.ValidationError(f"missing required fields: {', '.join(missing)}")
        for k in ("AMT_INCOME_TOTAL", "AMT_CREDIT"):
            try:
                if float(value[k]) <= 0:
                    raise serializers.ValidationError(f"{k} must be > 0")
            except (TypeError, ValueError) as exc:
                raise serializers.ValidationError(f"{k} must be a number") from exc
        if int(value["DAYS_BIRTH"]) >= 0:
            raise serializers.ValidationError(
                "DAYS_BIRTH must be negative (days before application)"
            )
        return value


class DecisionSerializer(serializers.Serializer):
    decision = serializers.ChoiceField(choices=["approve", "reject"])
    comment = serializers.CharField(required=False, allow_blank=True, default="", max_length=2000)

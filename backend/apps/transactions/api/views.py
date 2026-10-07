from datetime import timedelta

import django_filters
from django.db.models import Avg, Count, Q
from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import serializers, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.core.permissions import IsOfficer
from apps.transactions import services
from apps.transactions.models import Transaction

LATENCY_WINDOW_MINUTES = 60  # latency KPI = current behaviour, not the all-time mean
LATENCY_MAX_ROWS = 500


class TransactionSerializer(serializers.ModelSerializer):
    reviewed_by = serializers.CharField(source="reviewed_by.username", default=None, read_only=True)

    class Meta:
        model = Transaction
        fields = [
            "id",
            "transaction_id",
            "card1",
            "amount",
            "product_cd",
            "p_emaildomain",
            "device_type",
            "transaction_dt",
            "probability",
            "grade",
            "decision",
            "reasons",
            "model_version",
            "latency_ms",
            "status",
            "is_flagged",
            "reviewed_by",
            "reviewed_at",
            "created_at",
        ]
        read_only_fields = fields


class ScoreTransactionSerializer(serializers.Serializer):
    TransactionID = serializers.IntegerField()
    TransactionDT = serializers.IntegerField(min_value=0)
    TransactionAmt = serializers.FloatField(min_value=0.01)
    card1 = serializers.IntegerField()
    ProductCD = serializers.ChoiceField(choices=["W", "C", "R", "H", "S"], default="W")

    def to_internal_value(self, data):
        validated = super().to_internal_value(data)
        return {**data, **validated}  # keep the extra raw fields (card2.., C*, D*, V*, id_*)


class ReviewSerializer(serializers.Serializer):
    label = serializers.ChoiceField(choices=["fraud", "legit"])
    comment = serializers.CharField(required=False, allow_blank=True, default="")


class TransactionFilter(django_filters.FilterSet):
    since_id = django_filters.NumberFilter(field_name="id", lookup_expr="gt")
    flagged = django_filters.BooleanFilter(field_name="is_flagged")

    class Meta:
        model = Transaction
        fields = ["status", "decision", "card1", "flagged", "since_id"]


class TransactionViewSet(viewsets.ReadOnlyModelViewSet):
    """GET /api/transactions/?flagged=true&since_id=123   (live feed polling)
    POST /api/transactions/score/                         (real-time fraud check)
    POST /api/transactions/{id}/review/                   (analyst label)"""

    queryset = Transaction.objects.select_related("reviewed_by")
    serializer_class = TransactionSerializer
    throttle_scope = None  # set per action
    filterset_class = TransactionFilter
    ordering_fields = ["created_at", "probability", "amount"]

    @extend_schema(request=ScoreTransactionSerializer, responses=TransactionSerializer)
    @action(detail=False, methods=["post"], permission_classes=[IsOfficer], throttle_scope="fraud")
    def score(self, request):
        ser = ScoreTransactionSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        tx = services.score_transaction(ser.validated_data)
        return Response(TransactionSerializer(tx).data, status=status.HTTP_201_CREATED)

    @extend_schema(request=ReviewSerializer, responses=TransactionSerializer)
    @action(detail=True, methods=["post"], permission_classes=[IsOfficer])
    def review(self, request, pk=None):
        ser = ReviewSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        tx = services.review_transaction(self.get_object(), request.user, **ser.validated_data)
        return Response(TransactionSerializer(tx).data)

    @action(detail=False, methods=["get"])
    def stats(self, request):
        agg = Transaction.objects.aggregate(
            total=Count("id"),
            flagged=Count("id", filter=Q(is_flagged=True)),
            blocked=Count("id", filter=Q(status=Transaction.Status.BLOCKED)),
            confirmed=Count("id", filter=Q(status=Transaction.Status.CONFIRMED_FRAUD)),
        )
        # latency is an operational KPI: report current behaviour, not the all-time mean
        since = timezone.now() - timedelta(minutes=LATENCY_WINDOW_MINUTES)
        recent = (
            Transaction.objects.filter(created_at__gte=since)
            .order_by("-id")
            .values_list("id", flat=True)[:LATENCY_MAX_ROWS]
        )
        agg["avg_latency_ms"] = Transaction.objects.filter(id__in=list(recent)).aggregate(
            v=Avg("latency_ms")
        )["v"]
        agg["latency_window_minutes"] = LATENCY_WINDOW_MINUTES
        return Response(agg)

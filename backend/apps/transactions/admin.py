from django.contrib import admin

from apps.transactions.models import Transaction


@admin.register(Transaction)
class TransactionAdmin(admin.ModelAdmin):
    list_display = (
        "transaction_id",
        "card1",
        "amount",
        "probability",
        "decision",
        "status",
        "created_at",
    )
    list_filter = ("status", "decision", "is_flagged")
    search_fields = ("transaction_id", "card1")

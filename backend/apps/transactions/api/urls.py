from rest_framework.routers import SimpleRouter

from apps.transactions.api.views import TransactionViewSet

router = SimpleRouter()
router.register("", TransactionViewSet, basename="transaction")
urlpatterns = router.urls

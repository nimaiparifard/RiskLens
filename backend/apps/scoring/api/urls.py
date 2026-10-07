from rest_framework.routers import SimpleRouter

from apps.scoring.api.views import PredictionViewSet

router = SimpleRouter()
router.register("", PredictionViewSet, basename="prediction")
urlpatterns = router.urls

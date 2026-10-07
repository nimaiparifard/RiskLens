from rest_framework.routers import SimpleRouter

from apps.monitoring.api.views import DriftReportViewSet

router = SimpleRouter()
router.register("drift", DriftReportViewSet, basename="drift")
urlpatterns = router.urls

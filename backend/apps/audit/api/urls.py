from rest_framework.routers import SimpleRouter

from apps.audit.api.views import AuditLogViewSet

router = SimpleRouter()
router.register("", AuditLogViewSet, basename="audit")
urlpatterns = router.urls

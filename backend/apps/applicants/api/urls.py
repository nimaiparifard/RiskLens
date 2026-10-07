from rest_framework.routers import SimpleRouter

from apps.applicants.api.views import ApplicationViewSet

router = SimpleRouter()
router.register("", ApplicationViewSet, basename="application")
urlpatterns = router.urls

from django.contrib import admin
from django.urls import include, path
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from apps.core.views import health

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/health/", health, name="health"),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="docs"),
    path("api/auth/", include("apps.users.api.urls")),
    path("api/applications/", include("apps.applicants.api.urls")),
    path("api/predictions/", include("apps.scoring.api.urls")),
    path("api/transactions/", include("apps.transactions.api.urls")),
    path("api/monitoring/", include("apps.monitoring.api.urls")),
    path("api/models/", include("apps.mlmodels.api.urls")),
    path("api/audit/", include("apps.audit.api.urls")),
]

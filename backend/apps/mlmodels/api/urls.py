from django.urls import path

from apps.mlmodels.api.views import ModelsView, ReloadView

urlpatterns = [
    path("", ModelsView.as_view(), name="models"),
    path("reload/", ReloadView.as_view(), name="models-reload"),
]

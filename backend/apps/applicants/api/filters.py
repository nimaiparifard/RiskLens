import django_filters

from apps.applicants.models import Application


class ApplicationFilter(django_filters.FilterSet):
    status = django_filters.MultipleChoiceFilter(choices=Application.Status.choices)
    grade = django_filters.CharFilter(field_name="latest_grade")
    created_after = django_filters.IsoDateTimeFilter(field_name="created_at", lookup_expr="gte")
    created_before = django_filters.IsoDateTimeFilter(field_name="created_at", lookup_expr="lte")
    min_probability = django_filters.NumberFilter(
        field_name="latest_probability", lookup_expr="gte"
    )

    class Meta:
        model = Application
        fields = ["status", "grade", "is_override"]

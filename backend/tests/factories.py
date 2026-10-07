import factory

from apps.applicants.models import Applicant, Application
from apps.users.models import User


class UserFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = User
        django_get_or_create = ("username",)

    username = factory.Sequence(lambda n: f"user{n}")
    role = "officer"
    password = factory.PostGenerationMethodCall("set_password", "pass12345!")


class ApplicantFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Applicant

    full_name = factory.Faker("name")
    email = factory.Faker("email")


class ApplicationFactory(factory.django.DjangoModelFactory):
    class Meta:
        model = Application

    applicant = factory.SubFactory(ApplicantFactory)
    amt_credit = 400000
    amt_income_total = 150000
    amt_annuity = 20000
    features = factory.LazyFunction(
        lambda: {
            "AMT_INCOME_TOTAL": 150000,
            "AMT_CREDIT": 400000,
            "AMT_ANNUITY": 20000,
            "DAYS_BIRTH": -12000,
            "EXT_SOURCE_2": 0.4,
        }
    )
    history = factory.LazyFunction(lambda: {"BURO_COUNT": 2})

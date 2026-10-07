from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    class Role(models.TextChoices):
        VIEWER = "viewer", "Viewer"
        OFFICER = "officer", "Credit officer"
        ADMIN = "admin", "Admin"

    role = models.CharField(max_length=16, choices=Role.choices, default=Role.VIEWER)

    @property
    def is_officer(self) -> bool:
        return self.role in (self.Role.OFFICER, self.Role.ADMIN)

    def __str__(self) -> str:
        return f"{self.username} ({self.role})"

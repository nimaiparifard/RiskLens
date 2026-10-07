from rest_framework import permissions
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.views import TokenObtainPairView

from apps.users.api.serializers import RoleTokenSerializer, UserSerializer


class LoginView(TokenObtainPairView):
    serializer_class = RoleTokenSerializer
    throttle_scope = "scoring"


class MeView(APIView):
    permission_classes = [permissions.IsAuthenticated]
    serializer_class = UserSerializer

    def get(self, request):
        return Response(UserSerializer(request.user).data)

from django.db import connection
from django.http import JsonResponse


def health(request):
    """Liveness + DB check (used by docker-compose / k8s probes)."""
    try:
        with connection.cursor() as cur:
            cur.execute("SELECT 1")
        db = "ok"
    except Exception as exc:  # pragma: no cover
        return JsonResponse({"status": "error", "db": str(exc)}, status=503)
    return JsonResponse({"status": "ok", "db": db})

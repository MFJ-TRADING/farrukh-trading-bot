import json

from main import run_once


def handler(request=None, response=None):
    try:
        ok = run_once()
        payload = {"ok": ok, "status": "done"}
        if response is not None:
            response.status = 200
            response.headers["Content-Type"] = "application/json"
            response.text = json.dumps(payload)
            return response
        return payload
    except Exception as exc:
        payload = {"ok": False, "status": "error", "error": str(exc)}
        if response is not None:
            response.status = 500
            response.headers["Content-Type"] = "application/json"
            response.text = json.dumps(payload)
            return response
        return payload

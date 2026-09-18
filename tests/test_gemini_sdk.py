import sys
import types

import main


def test_get_google_model_prefers_google_genai(monkeypatch):
    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

    fake_google_genai = types.SimpleNamespace(Client=lambda api_key: FakeClient(api_key))
    monkeypatch.setattr(main, "API_KEY", "test-key")
    monkeypatch.setattr(main, "genai", None)
    monkeypatch.setattr(main, "google_genai_module", fake_google_genai)

    model = main.get_google_model("gemini-2.0-flash")

    assert model["provider"] == "google-genai"
    assert model["model"] == "gemini-2.0-flash"
    assert model["client"].api_key == "test-key"


def test_generate_google_content_builds_new_sdk_request(monkeypatch):
    class FakeResponse:
        def __init__(self, text):
            self.text = text

    fake_client = types.SimpleNamespace()
    fake_client.models = types.SimpleNamespace()
    fake_client.models.generate_content = lambda **kwargs: FakeResponse('{"direction":"bullish"}')

    fake_google = types.ModuleType("google")
    fake_google_genai = types.ModuleType("google.genai")
    fake_google_genai.Client = lambda api_key: fake_client
    fake_google_genai.types = types.SimpleNamespace(
        GenerateContentConfig=lambda **kwargs: kwargs
    )

    fake_google.genai = fake_google_genai
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_google_genai)

    monkeypatch.setattr(main, "API_KEY", "test-key")
    monkeypatch.setattr(main, "genai", None)
    monkeypatch.setattr(main, "google_genai_module", fake_google_genai)
    monkeypatch.setattr(main, "MODEL_CANDIDATES", ["gemini-2.0-flash"])

    response = main.generate_google_content("gemini-2.0-flash", "system", "user")

    assert response.text == '{"direction":"bullish"}'

"""Проверки HTTP-интерфейса без загрузки сайтов и платных API."""

import unittest
from unittest.mock import MagicMock, patch

import requests
from fastapi.testclient import TestClient
from openai import APIStatusError

from backend import web


REPORT = {
    "strengths": ["Сильная сторона"] * 3,
    "weaknesses": ["Слабая сторона"] * 3,
    "recommendations": ["Рекомендация"] * 5,
}


class WebTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(web.app)
        # Даже при ошибке в моке тест не сможет обратиться к внешним сервисам.
        self.addCleanup(self.client.close)
        for target in ("backend.site_parser.requests.get",
                       "backend.openai_module.OpenAI"):
            guard = patch(target, side_effect=AssertionError("Network forbidden"))
            guard.start()
            self.addCleanup(guard.stop)

    def test_page_and_local_assets(self):
        with patch.object(web.agent, "run") as run:
            page = self.client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn('lang="ru"', page.text)
            self.assertIn('type="url"', page.text)
            for asset in ("styles.css", "app.js"):
                response = self.client.get("/static/" + asset)
                self.assertEqual(response.status_code, 200)
            run.assert_not_called()

    def test_success_delegates_to_existing_agent(self):
        with patch.object(web.agent, "run", return_value=REPORT) as run:
            response = self.client.post("/api/analyze", json={"url": "https://example.com/page"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), REPORT)
            run.assert_called_once_with("https://example.com/page")

    def test_invalid_requests_do_not_start_analysis(self):
        with patch.object(web.agent, "run") as run:
            for payload in ({}, {"url": "example.com"}, {"url": "file:///tmp/test"},
                            {"url": "https://"}, {"url": None}):
                with self.subTest(payload=payload):
                    response = self.client.post("/api/analyze", json=payload)
                    self.assertEqual(response.status_code, 422)
            self.assertEqual(self.client.post("/api/analyze", content="{").status_code, 422)
            run.assert_not_called()

    def test_safe_errors_and_recovery(self):
        api_error = APIStatusError("secret-marker", response=MagicMock(status_code=401), body=None)
        for failure, status in ((requests.Timeout("secret-marker"), 502),
                                (api_error, 502), (ValueError("secret-marker"), 422),
                                (RuntimeError("secret-marker"), 500)):
            with self.subTest(failure=type(failure)):
                with patch.object(web.agent, "run", side_effect=[failure, REPORT]):
                    response = self.client.post("/api/analyze", json={"url": "https://example.com"})
                    self.assertEqual(response.status_code, status)
                    self.assertNotIn("secret-marker", response.text)
                    self.assertIsInstance(response.json()["detail"], str)
                    self.assertEqual(self.client.post("/api/analyze", json={"url": "https://example.com"}).json(), REPORT)

    def test_private_files_not_served(self):
        for path in ("/.env", "/backend/agent.py", "/static/.env", "/static/%2e%2e/.env"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path).status_code, 404)


if __name__ == "__main__":
    unittest.main()

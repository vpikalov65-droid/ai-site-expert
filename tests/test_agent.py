"""Локальные тесты: все сетевые границы заменены моками."""

import copy
import io
import json
import os
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import requests
from openai import APIConnectionError, APIStatusError

from backend import agent, openai_module, site_parser


REPORT = {
    "strengths": ["Понятный заголовок", "Контакты", "Описание услуги"],
    "weaknesses": ["Нет цены", "Нет сроков", "Нет примеров"],
    "recommendations": [
        "Добавьте стоимость услуги",
        "Укажите сроки выполнения",
        "Покажите примеры работ",
        "Добавьте кнопку заявки",
        "Опишите этапы работы",
    ],
}


def api_error(status: int) -> APIStatusError:
    response = MagicMock(status_code=status, headers={})
    return APIStatusError("secret-marker", response=response, body=None)


class ParserTests(unittest.TestCase):
    def test_html_cleanup_and_limit(self) -> None:
        with patch.object(site_parser, "_download") as download:
            download.return_value = (
                '<meta charset="utf-8"><h1>Привет</h1>'
                '<script>bad</script><style>bad</style>'
                '<nav>bad</nav><footer>bad</footer><p>Мир</p>'
            )
            self.assertEqual(
                site_parser.extract_text("https://example.com"), "Привет Мир"
            )
            download.return_value = "<p>" + "x" * 16000 + "</p>"
            self.assertEqual(
                len(site_parser.extract_text("https://example.com")), 15000
            )
            download.return_value = "<script>only script</script>"
            with self.assertRaises(ValueError):
                site_parser.extract_text("https://example.com")

    def test_invalid_url_never_downloaded(self) -> None:
        with patch.object(site_parser, "_download") as download:
            for url in ("example.com", "file:///tmp/a", "https://"):
                with self.subTest(url=url), self.assertRaises(ValueError):
                    site_parser.extract_text(url)
            download.assert_not_called()

    def test_network_retry_limit_and_delays(self) -> None:
        with patch.object(site_parser.requests, "get") as get:
            with patch.object(site_parser._download.retry, "sleep") as sleep:
                get.side_effect = requests.Timeout("secret-marker")
                with self.assertRaises(requests.Timeout):
                    site_parser._download("https://example.com")
                self.assertEqual(get.call_count, 3)
                self.assertEqual(sleep.call_args_list, [call(1.0), call(2.0)])

    def test_http_retry_and_recovery(self) -> None:
        for status, attempts in ((404, 1), (429, 3), (503, 3)):
            with self.subTest(status=status):
                response = MagicMock(status_code=status)
                response.raise_for_status.side_effect = requests.HTTPError(
                    response=response
                )
                with patch.object(site_parser.requests, "get") as get:
                    get.return_value.__enter__.return_value = response
                    with patch.object(site_parser._download.retry, "sleep"):
                        with self.assertRaises(requests.HTTPError):
                            site_parser._download("https://example.com")
                    self.assertEqual(get.call_count, attempts)
        with patch.object(site_parser.requests, "get") as get:
            response = MagicMock(text="Привет", encoding="utf-8")
            context = MagicMock()
            context.__enter__.return_value = response
            get.side_effect = [requests.ConnectionError(), context]
            with patch.object(site_parser._download.retry, "sleep"):
                self.assertEqual(
                    site_parser._download("https://example.com"), "Привет"
                )
            get.assert_called_with("https://example.com", timeout=15)


class OpenAITests(unittest.TestCase):
    def test_json_and_text_requests(self) -> None:
        with patch.object(openai_module, "load_dotenv") as load:
            with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
                with patch.object(openai_module, "OpenAI") as factory:
                    client = factory.return_value.__enter__.return_value
                    create = client.chat.completions.create
                    create.return_value = SimpleNamespace(choices=[
                        SimpleNamespace(
                            finish_reason="stop",
                            message=SimpleNamespace(
                                content=json.dumps(REPORT)
                            ),
                        )
                    ])
                    self.assertEqual(openai_module.ask_llm_json("UX"), REPORT)
                    self.assertEqual(
                        create.call_args.kwargs["response_format"],
                        {"type": "json_object"},
                    )
                    factory.assert_called_with(
                        api_key="test-key", max_retries=0, timeout=30.0
                    )
                    self.assertEqual(load.call_args.args[0].name, ".env")
                    openai_module.ask_llm("hello")
                    self.assertNotIn(
                        "response_format", create.call_args.kwargs
                    )

    def test_empty_and_incomplete_answers(self) -> None:
        for content, reason in ((None, "stop"), (" ", "stop"),
                                ("{}", "length"), ("{}", "content_filter")):
            with self.subTest(content=content, reason=reason):
                with (
                    patch.object(openai_module, "load_dotenv"),
                    patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}),
                    patch.object(openai_module, "OpenAI") as factory,
                ):
                    client = factory.return_value.__enter__.return_value
                    create = client.chat.completions.create
                    create.return_value = SimpleNamespace(choices=[
                        SimpleNamespace(
                            finish_reason=reason,
                            message=SimpleNamespace(content=content),
                        )
                    ])
                    with self.assertRaises(ValueError):
                        openai_module.ask_llm_json("UX")
                    self.assertEqual(create.call_count, 1)

    def test_missing_key(self) -> None:
        with patch.object(openai_module, "load_dotenv"):
            with patch.dict(os.environ, {}, clear=True):
                with patch.object(openai_module, "OpenAI") as factory:
                    with self.assertRaises(ValueError):
                        openai_module.ask_llm_json("UX")
                    factory.assert_not_called()

    def test_bad_json_and_non_object(self) -> None:
        for content in ("not JSON", "[]", "null"):
            with self.subTest(content=content):
                with patch.object(
                    openai_module, "_request", return_value=content
                ):
                    with self.assertRaises(ValueError):
                        openai_module.ask_llm_json("UX")

    def test_api_retry_policy(self) -> None:
        errors = [
            (APIConnectionError(request=MagicMock()), 3),
            (api_error(408), 3), (api_error(409), 3),
            (api_error(429), 3), (api_error(500), 3),
            (api_error(401), 1), (api_error(400), 1),
        ]
        quota = api_error(429)
        quota.code = "insufficient_quota"
        errors.append((quota, 1))
        for error, attempts in errors:
            with self.subTest(error=type(error), attempts=attempts):
                client = MagicMock()
                client.chat.completions.create.side_effect = error
                with patch.object(openai_module._create_completion.retry,
                                  "sleep") as sleep:
                    with self.assertRaises(type(error)):
                        openai_module._create_completion(client, model="test")
                self.assertEqual(
                    client.chat.completions.create.call_count, attempts
                )
                self.assertEqual(sleep.call_count, attempts - 1)
                if attempts == 3:
                    self.assertEqual(
                        sleep.call_args_list, [call(1.0), call(2.0)]
                    )

    def test_api_recovers(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = [api_error(503), "ok"]
        with patch.object(openai_module._create_completion.retry, "sleep"):
            self.assertEqual(openai_module._create_completion(client), "ok")


class AgentTests(unittest.TestCase):
    def test_run(self) -> None:
        with patch.object(
            agent, "extract_text", return_value="Текст"
        ) as parse:
            with patch.object(agent, "ask_llm_json", return_value=REPORT):
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(agent.run("https://example.com"), REPORT)
            parse.assert_called_once_with("https://example.com")

    def test_validation(self) -> None:
        self.assertTrue(agent.validate_report(REPORT))
        for invalid in (
            None, [], {}, {**REPORT, "recommendations": ["x"] * 4}
        ):
            self.assertFalse(agent.validate_report(invalid))
        for field in REPORT:
            for item in (" ", 1, None):
                invalid = copy.deepcopy(REPORT)
                invalid[field][0] = item
                self.assertFalse(agent.validate_report(invalid))

    def test_run_rejects_invalid_report(self) -> None:
        with patch.object(agent, "extract_text", return_value="Текст"):
            with patch.object(agent, "ask_llm_json", return_value={}):
                with (
                    redirect_stderr(io.StringIO()),
                    self.assertRaises(ValueError),
                ):
                    agent.run("https://example.com")

    def test_cli_json_and_url(self) -> None:
        with patch("sys.argv", ["agent.py", "https://example.org"]):
            with patch.object(agent, "run", return_value=REPORT) as run:
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(agent.main(), 0)
                self.assertEqual(json.loads(output.getvalue()), REPORT)
                run.assert_called_once_with("https://example.org")

    def test_cli_errors_do_not_leak_details(self) -> None:
        for error in (requests.Timeout("secret-marker"), api_error(401),
                      ValueError("secret-marker")):
            with self.subTest(error=type(error)):
                with patch("sys.argv", ["agent.py", "https://example.com"]):
                    with patch.object(agent, "run", side_effect=error):
                        output, errors = io.StringIO(), io.StringIO()
                        with redirect_stdout(output), redirect_stderr(errors):
                            self.assertEqual(agent.main(), 1)
                        self.assertEqual(output.getvalue(), "")
                        self.assertNotIn("secret-marker", errors.getvalue())
                        self.assertIn("Ошибка", errors.getvalue())


if __name__ == "__main__":
    unittest.main()

import argparse
import json
import sys
from typing import Any

import requests
from openai import OpenAIError

if __package__:
    from .site_parser import extract_text
    from .openai_module import ask_llm_json
else:
    from site_parser import extract_text
    from openai_module import ask_llm_json


def validate_report(report: dict[str, Any]) -> bool:
    """Проверяет структуру UX-отчёта."""

    if not isinstance(report, dict):
        return False

    required_fields = {
        "strengths": 3,
        "weaknesses": 3,
        "recommendations": 5,
    }

    for field, expected_count in required_fields.items():
        items = report.get(field)

        if not isinstance(items, list):
            return False

        if len(items) != expected_count:
            return False

        if not all(
            isinstance(item, str) and item.strip()
            for item in items
        ):
            return False

    return True


def run(url: str) -> dict[str, Any]:
    """Анализирует сайт и возвращает UX-отчёт."""

    print("Шаг 1. Загружаем сайт...", file=sys.stderr)
    site_text = extract_text(url)

    print("Шаг 2. Передаём текст нейросети...", file=sys.stderr)

    prompt = f"""
    Ты опытный UX-аналитик и маркетолог.

    Проанализируй текст веб-страницы:

    {site_text}

    Верни JSON-объект со следующими ключами:

    "strengths": список из 3 сильных сторон сайта.
    "weaknesses": список из 3 слабых сторон сайта.
    "recommendations": список из 5 конкретных улучшений.

    Все элементы списков должны быть строками на русском языке.

    Не придумывай факты, отсутствующие в тексте.
    Не оценивай визуальный дизайн без скриншота.
    """

    report = ask_llm_json(prompt)

    if not validate_report(report):
        raise ValueError(
            "Отчёт не прошёл проверку структуры"
        )

    print("Шаг 3. Анализ завершён!", file=sys.stderr)

    return report


def main() -> int:
    """CLI: JSON в stdout, сообщения и ошибки в stderr."""
    parser = argparse.ArgumentParser(description="UX-рецензент сайта")
    parser.add_argument("url", help="URL страницы с http:// или https://")
    args = parser.parse_args()
    try:
        result = run(args.url)
    except requests.RequestException:
        print("Ошибка загрузки сайта: проверьте адрес, сеть и HTTP-доступ.",
              file=sys.stderr)
        return 1
    except OpenAIError:
        print("Ошибка OpenAI API: проверьте ключ, доступ и лимиты.",
              file=sys.stderr)
        return 1
    except ValueError:
        print("Ошибка данных: проверьте URL, ключ в .env и структуру отчёта.",
              file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Локальный веб-интерфейс над существующим UX-агентом."""

from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAIError
from pydantic import BaseModel, HttpUrl

from . import agent


FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
app = FastAPI(title="AI Site Expert", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=FRONTEND), name="static")


class AnalysisRequest(BaseModel):
    url: HttpUrl


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(FRONTEND / "index.html")


@app.post("/api/analyze")
def analyze(payload: AnalysisRequest):
    # Синхронный маршрут выполняется FastAPI в пуле потоков.
    try:
        return agent.run(str(payload.url))
    except requests.RequestException:
        raise HTTPException(502, "Не удалось загрузить сайт. Проверьте адрес и доступность страницы.") from None
    except OpenAIError:
        raise HTTPException(502, "Сервис анализа недоступен. Проверьте настройки OpenAI и лимиты API.") from None
    except ValueError:
        raise HTTPException(422, "Не удалось получить отчёт. Проверьте настройки API и наличие текста на странице.") from None
    except Exception:
        raise HTTPException(500, "Произошла ошибка анализа. Попробуйте позже.") from None

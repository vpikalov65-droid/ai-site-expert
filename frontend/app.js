"use strict";

const form = document.getElementById("analysis-form");
const input = document.getElementById("site-url");
const button = document.getElementById("analyze-button");
const progress = document.getElementById("progress");
const error = document.getElementById("error");
const results = document.getElementById("results");
const completion = document.getElementById("completion");
const reportUrl = document.getElementById("report-url");
const counts = { strengths: 3, weaknesses: 3, recommendations: 5 };
let busy = false;

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  error.hidden = true;
  const url = input.value.trim();
  try {
    const parsed = new URL(url);
    if (!["http:", "https:"].includes(parsed.protocol)) throw new Error();
  } catch {
    error.textContent = "Укажите корректный адрес с http:// или https://";
    error.hidden = false;
    input.focus();
    return;
  }

  busy = true;
  button.disabled = true;
  input.disabled = true;
  progress.hidden = false;
  results.hidden = true;
  results.setAttribute("aria-busy", "true");
  completion.textContent = "";
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    if (!response.ok) {
      // Показываем только известные сообщения, без отражения ответа сервера.
      const messages = {
        422: "Не удалось выполнить анализ. Проверьте URL, настройки API и наличие текста на странице.",
        502: "Не удалось загрузить страницу или обратиться к OpenAI. Проверьте доступность сайта, настройки API и лимиты.",
      };
      throw new Error(messages[response.status] || "Сервис временно недоступен. Попробуйте позже.");
    }
    const report = await response.json();
    for (const [key, count] of Object.entries(counts)) {
      if (!Array.isArray(report[key]) || report[key].length !== count ||
          !report[key].every((item) => typeof item === "string" && item.trim())) {
        throw new Error("Получен некорректный отчёт. Попробуйте выполнить анализ ещё раз.");
      }
    }
    for (const key of Object.keys(counts)) {
      const items = report[key].map((text) => {
        const li = document.createElement("li");
        li.textContent = text;
        return li;
      });
      document.getElementById(key).replaceChildren(...items);
    }
    reportUrl.textContent = url;
    reportUrl.hidden = false;
    document.getElementById("report-status").textContent = "Анализ завершён";
    results.hidden = false;
    completion.textContent = "Отчёт готов: 3 сильные стороны, 3 слабые стороны и 5 рекомендаций.";
  } catch (failure) {
    error.textContent = failure instanceof TypeError
      ? "Нет связи с локальным сервером. Проверьте, что он запущен, и повторите попытку."
      : failure.message || "Не удалось выполнить анализ. Попробуйте позже.";
    error.hidden = false;
  } finally {
    busy = false;
    button.disabled = false;
    input.disabled = false;
    progress.hidden = true;
    results.setAttribute("aria-busy", "false");
  }
});

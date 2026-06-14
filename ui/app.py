from __future__ import annotations

import json
import os
import time
from typing import Any

import gradio as gr
import requests

DEFAULT_ENDPOINT = "http://localhost:8082/serve/text-sentiment"
ENDPOINT = os.getenv("SERVING_ENDPOINT", DEFAULT_ENDPOINT)
TIMEOUT_SECONDS = float(os.getenv("SERVING_TIMEOUT", "10"))


def extract_label(payload: dict[str, Any]) -> str:
    if "label" in payload:
        return str(payload["label"])
    if "predictions" in payload and payload["predictions"]:
        first = payload["predictions"][0]
        if isinstance(first, dict) and "label" in first:
            return str(first["label"])
    if "result" in payload and isinstance(payload["result"], dict):
        return extract_label(payload["result"])
    return "unknown"


def predict(text: str, endpoint: str) -> tuple[str, str, str, str]:
    endpoint = (endpoint or ENDPOINT).strip()
    if not text or not text.strip():
        return "", "0 ms", "Enter a non-empty text.", "{}"

    started = time.perf_counter()
    try:
        response = requests.post(endpoint, json={"text": text.strip()}, timeout=TIMEOUT_SECONDS)
        latency_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        label = extract_label(payload)
        return label, f"{latency_ms:.1f} ms", "OK", json.dumps(payload, ensure_ascii=False, indent=2)
    except requests.exceptions.RequestException as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        return "", f"{latency_ms:.1f} ms", f"Endpoint error: {exc}", "{}"
    except ValueError as exc:
        latency_ms = (time.perf_counter() - started) * 1000
        raw = response.text if "response" in locals() else ""
        return "", f"{latency_ms:.1f} ms", f"Invalid JSON response: {exc}", raw


with gr.Blocks(title="TextFlow Sentiment UI") as demo:
    gr.Markdown("# TextFlow Sentiment")
    gr.Markdown("UI sends HTTP requests to ClearML Serving. The model is not loaded by the UI process.")

    endpoint_input = gr.Textbox(label="Serving endpoint", value=ENDPOINT)
    text_input = gr.Textbox(
        label="Text",
        lines=5,
        placeholder="Example: The delivery arrived early and the service was helpful.",
    )
    predict_button = gr.Button("Predict", variant="primary")

    with gr.Row():
        label_output = gr.Textbox(label="Label")
        latency_output = gr.Textbox(label="Latency")
    status_output = gr.Textbox(label="Status")
    raw_output = gr.Code(label="Raw response", language="json")

    predict_button.click(
        predict,
        inputs=[text_input, endpoint_input],
        outputs=[label_output, latency_output, status_output, raw_output],
    )
    text_input.submit(
        predict,
        inputs=[text_input, endpoint_input],
        outputs=[label_output, latency_output, status_output, raw_output],
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.getenv("UI_PORT", "7860")))
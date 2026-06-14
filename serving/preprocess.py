from __future__ import annotations

import re
from typing import Any

import numpy as np


class Preprocess:
    labels = {
        0: "negative",
        1: "positive",
    }

    def preprocess(self, body: dict[str, Any], state: dict[str, Any], collect_custom_statistics_fn=None) -> list[str]:
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")

        raw_text = body.get("text", body.get("texts"))
        if raw_text is None:
            raise ValueError("Request must contain 'text' or 'texts'")

        if isinstance(raw_text, str):
            texts = [raw_text]
            state["single_request"] = True
        elif isinstance(raw_text, list):
            texts = [str(item) for item in raw_text]
            state["single_request"] = False
        else:
            raise ValueError("'text' must be a string; 'texts' must be a list")

        cleaned = [self._clean(text) for text in texts]
        if any(not text for text in cleaned):
            raise ValueError("Input text must be non-empty")

        if collect_custom_statistics_fn:
            collect_custom_statistics_fn({"batch_size": len(cleaned)})
        return cleaned

    def postprocess(self, data: Any, state: dict[str, Any], collect_custom_statistics_fn=None) -> dict[str, Any]:
        values = np.asarray(data).reshape(-1).tolist()
        predictions = []
        for value in values:
            try:
                label_id = int(value)
            except (TypeError, ValueError):
                label_id = value
            predictions.append(
                {
                    "label": self.labels.get(label_id, str(label_id)),
                    "label_id": label_id,
                }
            )

        if state.get("single_request", False):
            return predictions[0] if predictions else {"label": "unknown", "label_id": None}
        return {"predictions": predictions}

    @staticmethod
    def _clean(text: str) -> str:
        text = re.sub(r"<[^>]+>", " ", str(text))
        text = re.sub(r"\s+", " ", text).strip().lower()
        return text
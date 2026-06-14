from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from clearml import Task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish the best completed ClearML experiment model.")
    parser.add_argument("--project", default="MLOps-Course", help="ClearML project containing training tasks.")
    parser.add_argument("--task-name", default="textflow-train", help="Training task name to compare.")
    parser.add_argument("--metric-title", default="validation", help="Scalar metric title.")
    parser.add_argument("--metric-series", default="f1", help="Scalar metric series used for model selection.")
    return parser.parse_args()


def metric_value(task: Task, title: str, series: str) -> float | None:
    metrics: dict[str, Any] = task.get_last_scalar_metrics() or {}
    entry = metrics.get(title, {}).get(series)
    if isinstance(entry, dict):
        for key in ("last", "value", "y"):
            if key in entry:
                return float(entry[key])
    if entry is not None:
        return float(entry)
    return None


def choose_best_task(project: str, task_name: str, title: str, series: str) -> tuple[Task, float]:
    tasks = Task.get_tasks(
        project_name=project,
        task_name=task_name,
        task_filter={"status": ["completed"]},
    )
    if not tasks:
        raise RuntimeError(f"No completed tasks named '{task_name}' found in project '{project}'.")

    ranked: list[tuple[float, Task]] = []
    for task in tasks:
        value = metric_value(task, title, series)
        if value is not None:
            ranked.append((value, task))
            print(f"candidate task={task.id} {title}/{series}={value:.4f}")
        else:
            print(f"skip task={task.id}: metric {title}/{series} not found")

    if not ranked:
        raise RuntimeError(f"No tasks have metric {title}/{series}.")
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1], ranked[0][0]


def publish_output_model(task: Task, score: float) -> str:
    models = task.get_models().get("output", [])
    if not models:
        raise RuntimeError(f"Task {task.id} has no OutputModel. Check train/train.py execution.")

    model = models[-1]
    model.publish()
    try:
        model.set_tags(["course_project", "sentiment", "registry", f"selected_f1_{score:.4f}"])
    except Exception:
        pass

    model_id = model.id
    output_path = Path(__file__).with_name("model_id.txt")
    output_path.write_text(model_id, encoding="utf-8")
    print(f"Published model ID: {model_id}")
    print(f"Best task ID: {task.id}")
    print(f"Saved model id to: {output_path}")
    return model_id


def main() -> None:
    args = parse_args()
    best_task, best_score = choose_best_task(args.project, args.task_name, args.metric_title, args.metric_series)
    publish_output_model(best_task, best_score)


if __name__ == "__main__":
    main()
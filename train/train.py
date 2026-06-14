from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from clearml import Dataset, OutputModel, Task

try:
    from clearml.config import running_remotely
except Exception:  # pragma: no cover - compatibility fallback
    def running_remotely() -> bool:
        return False

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABELS = [0, 1]
LABEL_NAMES = ["negative", "positive"]


def read_text_if_exists(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return ""


def detect_git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "not_available"


def model_upload_uri() -> str:
    return (
        os.getenv("CLEARML_MODEL_UPLOAD_URI")
        or os.getenv("CLEARML_FILES_HOST")
        or "http://host.docker.internal:8081"
    )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sentiment classifier through ClearML Agent.")
    parser.add_argument("--project", default=os.getenv("CLEARML_PROJECT", "MLOps-Course"), help="ClearML project.")
    parser.add_argument("--task-name", default="textflow-train", help="ClearML task name for experiments.")
    parser.add_argument("--queue", default=os.getenv("CLEARML_QUEUE", "students"), help="ClearML Agent queue.")
    parser.add_argument("--dataset-id", default=os.getenv("CLEARML_DATASET_ID", ""), help="ClearML Dataset ID.")
    parser.add_argument("--remote", action="store_true", help="Send this task to ClearML Agent and exit locally.")
    parser.add_argument("--model-name", default="textflow-sentiment-pipeline", help="Output model name.")
    parser.add_argument("--max-features", type=int, default=1200, help="TF-IDF max_features.")
    parser.add_argument("--ngram-max", type=int, default=1, choices=[1, 2], help="Upper n-gram range for TF-IDF.")
    parser.add_argument("--min-df", type=int, default=1, help="TF-IDF min_df.")
    parser.add_argument("--c", type=float, default=1.0, help="LogisticRegression regularization strength.")
    parser.add_argument("--solver", default="liblinear", choices=["liblinear", "lbfgs"], help="LogisticRegression solver.")
    parser.add_argument("--random-state", type=int, default=2026, help="Model random state.")
    return parser.parse_args()


def load_clearml_dataset(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    dataset = Dataset.get(dataset_id=dataset_id)
    dataset_path = Path(dataset.get_local_copy())
    train_path = dataset_path / "train.csv"
    test_path = dataset_path / "test.csv"
    if not train_path.exists() or not test_path.exists():
        available = ", ".join(path.name for path in dataset_path.glob("*.csv"))
        raise FileNotFoundError(f"Dataset must contain train.csv and test.csv. Available CSV files: {available}")
    return pd.read_csv(train_path), pd.read_csv(test_path), dataset.id


def build_model(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    max_features=args.max_features,
                    min_df=args.min_df,
                    ngram_range=(1, args.ngram_max),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=args.c,
                    solver=args.solver,
                    max_iter=1000,
                    random_state=args.random_state,
                ),
            ),
        ]
    )


def log_confusion_matrix(task: Task, y_true: list[int], y_pred: list[int]) -> None:
    logger = task.get_logger()
    matrix = confusion_matrix(y_true, y_pred, labels=LABELS)
    logger.report_confusion_matrix(
        title="confusion_matrix",
        series="validation",
        matrix=matrix.tolist(),
        iteration=0,
        xaxis=LABEL_NAMES,
        yaxis=LABEL_NAMES,
    )

    figure, axis = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=LABEL_NAMES,
        cmap="Blues",
        ax=axis,
        colorbar=False,
    )
    axis.set_title("Validation confusion matrix")
    figure.tight_layout()
    logger.report_matplotlib_figure(
        title="confusion_matrix_image",
        series="validation",
        figure=figure,
        iteration=0,
        report_image=True,
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if not args.dataset_id:
        args.dataset_id = read_text_if_exists(PROJECT_ROOT / "dataset" / "dataset_id.txt")
    if not args.dataset_id:
        raise ValueError("Dataset ID is required. Run dataset/create_dataset.py or pass --dataset-id.")

    task = Task.init(
        project_name=args.project,
        task_name=args.task_name,
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
    )
    task.add_tags(["course_project", "sentiment", "sklearn", f"ngram_{args.ngram_max}"])
    task.connect(vars(args), name="train_args")
    task.set_parameter("source/git_commit", detect_git_commit())

    if args.remote and not running_remotely():
        print(f"Sending task {task.id} to ClearML queue '{args.queue}'")
        task.execute_remotely(queue_name=args.queue, clone=False, exit_process=True)

    train_df, test_df, resolved_dataset_id = load_clearml_dataset(args.dataset_id)
    task.set_parameter("dataset/id", resolved_dataset_id)
    task.set_parameter("dataset/train_rows", int(len(train_df)))
    task.set_parameter("dataset/test_rows", int(len(test_df)))

    model = build_model(args)
    y_train = train_df["label"].astype(int)
    y_test = test_df["label"].astype(int)
    model.fit(train_df["text"].astype(str), y_train)
    predictions = model.predict(test_df["text"].astype(str))

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions, average="binary", pos_label=1))
    f1_weighted = float(f1_score(y_test, predictions, average="weighted"))

    logger = task.get_logger()
    logger.report_scalar("validation", "accuracy", iteration=0, value=accuracy)
    logger.report_scalar("validation", "f1", iteration=0, value=f1)
    logger.report_scalar("validation", "f1_weighted", iteration=0, value=f1_weighted)
    log_confusion_matrix(task, y_test.tolist(), [int(v) for v in predictions])

    model_dir = PROJECT_ROOT / "models" / task.id
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "sentiment_pipeline.joblib"
    joblib.dump(model, model_path)

    task.upload_artifact("model_joblib", artifact_object=str(model_path))
    task.upload_artifact("label_map", artifact_object={"0": "negative", "1": "positive"})

    output_model = OutputModel(task=task, name=args.model_name, framework="scikit-learn")
    output_model.update_weights(
        weights_filename=str(model_path),
        upload_uri=model_upload_uri(),
        auto_delete_file=False,
    )

    Path(__file__).with_name("last_task_id.txt").write_text(task.id, encoding="utf-8")
    Path(__file__).with_name("last_model_id.txt").write_text(output_model.id or "", encoding="utf-8")

    logger.report_text(
        f"accuracy={accuracy:.4f}, f1={f1:.4f}, f1_weighted={f1_weighted:.4f}, "
        f"model_id={output_model.id}, dataset_id={resolved_dataset_id}"
    )
    print(f"Task ID: {task.id}")
    print(f"Output model ID: {output_model.id}")
    print(f"accuracy={accuracy:.4f} f1={f1:.4f} f1_weighted={f1_weighted:.4f}")
    print(f"Model file: {model_path}")
    task.close()


if __name__ == "__main__":
    main()
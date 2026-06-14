from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from clearml import Dataset, Task

PROJECT_ROOT = Path(__file__).resolve().parents[1]

POSITIVE_REVIEWS = [
    "The delivery arrived early and the product worked exactly as promised.",
    "I enjoyed the service because support answered quickly and politely.",
    "The app feels fast, simple, and reliable during daily use.",
    "This purchase was worth the price and I would recommend it.",
    "The interface is clean and the instructions are easy to follow.",
    "Customer service solved my issue on the first attempt.",
    "The packaging was careful and everything looked professional.",
    "The result exceeded my expectations and saved me time.",
    "I like the stable performance and clear feedback messages.",
    "The new version is much better than the previous one.",
    "Setup was quick and the default settings worked well.",
    "The quality is consistent and the product feels dependable.",
    "The response was accurate, friendly, and genuinely helpful.",
    "I had a smooth experience from registration to checkout.",
    "The documentation is practical and easy to understand.",
    "Everything loaded quickly and the workflow felt comfortable.",
    "The tool handled my request without errors or delays.",
    "The final output was clear, useful, and nicely formatted.",
    "I trust this service after several successful orders.",
    "The recommendation matched my needs very well.",
    "The model gave a confident and useful prediction.",
    "The dashboard made it easy to compare the results.",
    "The endpoint responded quickly during the whole test.",
    "The system recovered cleanly after a small input mistake.",
]

NEGATIVE_REVIEWS = [
    "The delivery was late and the product did not work properly.",
    "Support ignored my request and gave an unhelpful answer.",
    "The app feels slow, confusing, and unreliable under load.",
    "This purchase was disappointing and not worth the money.",
    "The interface is cluttered and the instructions are unclear.",
    "Customer service failed to solve the issue after several attempts.",
    "The packaging was damaged and the item looked careless.",
    "The result was worse than expected and wasted my time.",
    "I dislike the unstable performance and vague error messages.",
    "The new version is worse than the previous one.",
    "Setup was painful and the default settings failed immediately.",
    "The quality is inconsistent and the product feels fragile.",
    "The response was inaccurate, cold, and not useful.",
    "I had a frustrating experience from registration to checkout.",
    "The documentation is incomplete and hard to understand.",
    "Everything loaded slowly and the workflow felt uncomfortable.",
    "The tool failed on my request with repeated errors.",
    "The final output was confusing, noisy, and poorly formatted.",
    "I do not trust this service after several failed orders.",
    "The recommendation did not match my needs at all.",
    "The model returned a wrong and unhelpful prediction.",
    "The dashboard made it hard to compare the results.",
    "The endpoint responded slowly during the whole test.",
    "The system crashed after a small input mistake.",
]

CONTEXTS = [
    "Short review: {text}",
    "After a week of use, {text}",
    "My feedback is simple: {text}",
    "From a customer perspective, {text}",
]


def resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def expand_reviews(texts: list[str], label: int, label_name: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for text in texts:
        rows.append({"text": text, "label": label, "label_name": label_name})
        for template in CONTEXTS[:2]:
            rows.append({"text": template.format(text=text), "label": label, "label_name": label_name})
    return rows


def stratified_split(rows: list[dict[str, object]], seed: int, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    positives = [row for row in rows if row["label"] == 1]
    negatives = [row for row in rows if row["label"] == 0]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    def split_class(items: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        cut = max(1, int(len(items) * train_ratio))
        return items[:cut], items[cut:]

    pos_train, pos_test = split_class(positives)
    neg_train, neg_test = split_class(negatives)
    train_rows = pos_train + neg_train
    test_rows = pos_test + neg_test
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def build_dataset(seed: int, train_ratio: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = expand_reviews(POSITIVE_REVIEWS, 1, "positive") + expand_reviews(NEGATIVE_REVIEWS, 0, "negative")
    return stratified_split(rows, seed=seed, train_ratio=train_ratio)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a versioned ClearML Dataset for the course project.")
    parser.add_argument("--project", default="MLOps-Course", help="ClearML project for the data preparation task.")
    parser.add_argument("--dataset-project", default="MLOps-Course/Datasets", help="ClearML Dataset project.")
    parser.add_argument("--dataset-name", default="textflow-sentiment-reviews", help="ClearML Dataset name.")
    parser.add_argument("--version", default="1.0", help="Dataset version.")
    parser.add_argument("--output-dir", default="data/sentiment_reviews", help="Local directory for generated CSV files.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed for deterministic split.")
    parser.add_argument("--train-ratio", type=float, default=0.75, help="Train split ratio.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = Task.init(
        project_name=args.project,
        task_name="textflow-create-dataset",
        task_type=Task.TaskTypes.data_processing,
        reuse_last_task_id=False,
    )
    task.connect(vars(args), name="dataset_args")
    task.add_tags(["course_project", "dataset", "sentiment"])

    train_df, test_df = build_dataset(seed=args.seed, train_ratio=args.train_ratio)
    train_path = output_dir / "train.csv"
    test_path = output_dir / "test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    dataset = Dataset.create(
        dataset_name=args.dataset_name,
        dataset_project=args.dataset_project,
        dataset_version=args.version,
    )
    dataset.add_files(path=str(train_path))
    dataset.add_files(path=str(test_path))
    dataset.add_tags(["sentiment", "text-classification", "course-project"])
    dataset.upload(show_progress=True)
    dataset.finalize(auto_upload=False)

    id_path = Path(__file__).with_name("dataset_id.txt")
    id_path.write_text(dataset.id, encoding="utf-8")
    task.upload_artifact("train_preview", train_df.head(12))
    task.upload_artifact("test_preview", test_df.head(12))
    task.get_logger().report_text(
        f"Created dataset_id={dataset.id}, version={dataset.version}, "
        f"train_rows={len(train_df)}, test_rows={len(test_df)}"
    )

    print(f"Dataset ID: {dataset.id}")
    print(f"Dataset version: {dataset.version}")
    print(f"Local files: {train_path} | {test_path}")
    print(f"Saved dataset id to: {id_path}")
    task.close()


if __name__ == "__main__":
    main()
"""Train and evaluate a sklearn model on board-aware Lichess move features.

Splits by game ID (not by row) to prevent leakage across snapshots from the
same game. Supports logistic regression, calibrated SVM, histogram gradient
boosting, XGBoost, and several MLP configurations via --model.

    python scripts/train_baseline_model.py \
        --input data/dev_board_features_10000_games.csv \
        --model logreg --output-json artifacts/logreg_10k.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


NUMERIC_FEATURES = [
    "white_elo",
    "black_elo",
    "elo_diff_white_minus_black",
    "ply_index",
    "is_capture",
    "is_check",
    "is_checkmate",
    "is_castle",
    "is_promotion",
    "san_length",
    "white_time_seconds",
    "black_time_seconds",
    "mover_time_seconds",
    "opponent_time_seconds",
    "mover_time_spent_seconds",
    "white_time_ratio",
    "black_time_ratio",
    "clock_diff_seconds_white_minus_black",
    "legal_moves_count",
    "halfmove_clock",
    "white_material",
    "black_material",
    "material_diff_white_minus_black",
    "white_pawns",
    "black_pawns",
    "white_knights",
    "black_knights",
    "white_bishops",
    "black_bishops",
    "white_rooks",
    "black_rooks",
    "white_queens",
    "black_queens",
    "white_has_bishop_pair",
    "black_has_bishop_pair",
    "white_can_castle_kingside",
    "white_can_castle_queenside",
    "black_can_castle_kingside",
    "black_can_castle_queenside",
    "is_insufficient_material",
]

CATEGORICAL_FEATURES = [
    "mover",
    "side_to_move",
]

EXCLUDED_INFERENCE_FEATURES = {
    "fullmove_number",
}


def get_available_feature_columns(
    frame: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Return the numeric and categorical features available in the dataset."""
    numeric = [
        column
        for column in NUMERIC_FEATURES
        if column in frame.columns and column not in EXCLUDED_INFERENCE_FEATURES
    ]
    categorical = [
        column
        for column in CATEGORICAL_FEATURES
        if column in frame.columns and column not in EXCLUDED_INFERENCE_FEATURES
    ]
    return numeric, categorical


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model training.

    Returns:
        argparse.Namespace: Parsed arguments.

    Example:
        >>> isinstance(parse_args, object)
        True
    """
    parser = argparse.ArgumentParser(
        description="Train a baseline result predictor on extracted move features."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="CSV file produced by extract_lichess_features.py.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of games reserved for the test split.",
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.1,
        help="Fraction of remaining non-test games reserved for validation.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on the number of rows loaded from the CSV file.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=None,
        help="Optional JSON path where summary metrics will be written.",
    )
    parser.add_argument(
        "--model",
        choices=[
            "logreg",
            "hgbt",
            "mlp",
            "mlp_deep",
            "mlp_wide",
            "mlp_balanced",
            "svm",
            "xgb",
        ],
        default="logreg",
        help=(
            "Model to train: logistic regression, histogram gradient boosting, "
            "MLP variants, calibrated linear SVM, or XGBoost."
        ),
    )
    parser.add_argument(
        "--oversample-draw-factor",
        type=float,
        default=1.0,
        help=(
            "Training-only oversampling multiplier for draw rows. "
            "Use values > 1.0 to upweight the draw class."
        ),
    )
    return parser.parse_args()


def load_dataset(path: Path, max_rows: int | None) -> pd.DataFrame:
    """Load the extracted move-level dataset.

    Args:
        path: Path to the CSV feature file.
        max_rows: Optional row limit.

    Returns:
        pd.DataFrame: Loaded dataset.

    Example:
        >>> callable(load_dataset)
        True
    """
    return pd.read_csv(path, nrows=max_rows)


def oversample_draw_rows(
    features: pd.DataFrame,
    target: pd.Series,
    factor: float,
) -> tuple[pd.DataFrame, pd.Series]:
    """Oversample draw rows in the training set only.

    Args:
        features: Training feature matrix.
        target: Training labels.
        factor: Multiplicative factor for draw rows.

    Returns:
        tuple[pd.DataFrame, pd.Series]: Oversampled features and labels.
    """
    if factor <= 1.0:
        return features, target

    train_frame = features.copy()
    train_frame["target"] = target.values
    draw_rows = train_frame[train_frame["target"] == "draw"]

    if draw_rows.empty:
        return features, target

    extra_count = int(len(draw_rows) * (factor - 1.0))
    if extra_count <= 0:
        return features, target

    extra_rows = draw_rows.sample(
        n=extra_count,
        replace=True,
        random_state=42,
    )
    oversampled = pd.concat([train_frame, extra_rows], ignore_index=True)
    oversampled = oversampled.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return oversampled.drop(columns=["target"]), oversampled["target"]


def prepare_target(frame: pd.DataFrame) -> pd.Series:
    """Build a multiclass target column from the PGN result.

    Args:
        frame: Input feature table.

    Returns:
        pd.Series: Target labels in string form.

    Example:
        >>> prepare_target(pd.DataFrame({"result": ["1-0", "0-1"]})).tolist()
        ['white_win', 'black_win']
    """
    mapping = {
        "1-0": "white_win",
        "0-1": "black_win",
        "1/2-1/2": "draw",
    }
    return frame["result"].map(mapping)


def split_by_game(
    frame: pd.DataFrame, test_size: float, val_size: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the dataset by game ID to avoid leakage across snapshots.

    Args:
        frame: Full move-level dataset.
        test_size: Fraction of games for the test split.
        val_size: Fraction of post-test remaining games for the validation split.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            Train, validation, and test subsets.

    Example:
        >>> df = pd.DataFrame({"game_id": ["a", "a", "b", "b"], "x": [1, 2, 3, 4]})
        >>> train_df, val_df, test_df = split_by_game(df, 0.5, 0.5)
        >>> set(train_df["game_id"]).isdisjoint(set(test_df["game_id"]))
        True
    """
    game_targets = (
        frame[["game_id", "target"]]
        .drop_duplicates(subset=["game_id"])
        .dropna(subset=["game_id", "target"])
    )
    stratify_labels = None
    class_counts = game_targets["target"].value_counts()
    if not class_counts.empty and int(class_counts.min()) >= 2:
        stratify_labels = game_targets["target"]

    train_val_games, test_games = train_test_split(
        game_targets["game_id"],
        test_size=test_size,
        random_state=42,
        shuffle=True,
        stratify=stratify_labels,
    )

    remaining_targets = game_targets[game_targets["game_id"].isin(train_val_games)]
    remaining_stratify = None
    remaining_counts = remaining_targets["target"].value_counts()
    if not remaining_counts.empty and int(remaining_counts.min()) >= 2:
        remaining_stratify = remaining_targets["target"]

    train_games, val_games = train_test_split(
        remaining_targets["game_id"],
        test_size=val_size,
        random_state=42,
        shuffle=True,
        stratify=remaining_stratify,
    )

    train_frame = frame[frame["game_id"].isin(train_games)].copy()
    val_frame = frame[frame["game_id"].isin(val_games)].copy()
    test_frame = frame[frame["game_id"].isin(test_games)].copy()
    return train_frame, val_frame, test_frame


def build_model(
    model_name: str, numeric_features: list[str], categorical_features: list[str]
) -> Pipeline:
    """Create the preprocessing and training pipeline.

    Returns:
        Pipeline: Scikit-learn pipeline for baseline training.

    Example:
        >>> isinstance(build_model("logreg", ["white_elo"], ["mover"]), Pipeline)
        True
    """
    if model_name == "logreg":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )
        classifier = LogisticRegression(max_iter=400, random_state=42)
    elif model_name == "hgbt":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )
        classifier = HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.1,
            max_iter=200,
            min_samples_leaf=40,
            random_state=42,
        )
    elif model_name == "mlp":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )
        classifier = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42,
        )
    elif model_name == "mlp_deep":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )
        classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=5e-5,
            batch_size=512,
            learning_rate_init=7e-4,
            max_iter=150,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=12,
            random_state=42,
            verbose=False,
        )
    elif model_name == "mlp_wide":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )
        classifier = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=512,
            learning_rate_init=5e-4,
            max_iter=180,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=42,
            verbose=False,
        )
    elif model_name == "mlp_balanced":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )
        classifier = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=3e-4,
            batch_size=512,
            learning_rate_init=4e-4,
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=18,
            random_state=42,
            verbose=False,
        )
    elif model_name == "svm":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
            ]
        )
        classifier = CalibratedClassifierCV(
            estimator=LinearSVC(
                C=1.5,
                class_weight="balanced",
                dual="auto",
                max_iter=6000,
                random_state=42,
            ),
            method="sigmoid",
            cv=3,
        )
    elif model_name == "xgb":
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_features),
                (
                    "cat",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    categorical_features,
                ),
            ]
        )
        classifier = XGBClassifier(
            objective="multi:softprob",
            num_class=3,
            n_estimators=300,
            max_depth=8,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=2,
            tree_method="hist",
            random_state=42,
            eval_metric="mlogloss",
            n_jobs=0,
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def main() -> None:
    """Train the baseline model and report evaluation metrics."""
    args = parse_args()
    frame = load_dataset(args.input, args.max_rows)
    frame = frame.dropna(subset=["game_id", "result"]).copy()
    frame["target"] = prepare_target(frame)
    frame = frame.dropna(subset=["target"]).copy()
    numeric_features, categorical_features = get_available_feature_columns(frame)

    train_frame, val_frame, test_frame = split_by_game(
        frame,
        test_size=args.test_size,
        val_size=args.val_size,
    )
    train_class_count = train_frame["target"].nunique()
    val_class_count = val_frame["target"].nunique()
    test_class_count = test_frame["target"].nunique()

    if train_class_count < 2:
        raise ValueError(
            "Training split contains fewer than 2 classes. "
            "Use a larger dataset or adjust the split."
        )

    selected_features = numeric_features + categorical_features
    x_train = train_frame[selected_features]
    y_train = train_frame["target"]
    x_val = val_frame[selected_features]
    y_val = val_frame["target"]
    x_test = test_frame[selected_features]
    y_test = test_frame["target"]
    x_train, y_train = oversample_draw_rows(
        x_train,
        y_train,
        factor=args.oversample_draw_factor,
    )

    label_mapping = None
    if args.model == "xgb":
        ordered_labels = ["white_win", "black_win", "draw"]
        label_mapping = {label: idx for idx, label in enumerate(ordered_labels)}
        y_train = y_train.map(label_mapping)
        y_val = y_val.map(label_mapping)
        y_test = y_test.map(label_mapping)

    model = build_model(args.model, numeric_features, categorical_features)
    model.fit(x_train, y_train)

    y_val_pred = model.predict(x_val)
    y_val_proba = model.predict_proba(x_val)
    y_test_pred = model.predict(x_test)
    y_test_proba = model.predict_proba(x_test)

    metrics = {
        "train_rows": int(len(train_frame)),
        "val_rows": int(len(val_frame)),
        "test_rows": int(len(test_frame)),
        "train_games": int(train_frame["game_id"].nunique()),
        "val_games": int(val_frame["game_id"].nunique()),
        "test_games": int(test_frame["game_id"].nunique()),
        "train_classes": int(train_class_count),
        "val_classes": int(val_class_count),
        "test_classes": int(test_class_count),
        "model": args.model,
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "oversample_draw_factor": float(args.oversample_draw_factor),
        "val_accuracy": float(accuracy_score(y_val, y_val_pred)),
        "val_log_loss": float(log_loss(y_val, y_val_proba, labels=model.classes_)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "test_log_loss": float(log_loss(y_test, y_test_proba, labels=model.classes_)),
    }

    print("Baseline metrics")
    print(json.dumps(metrics, indent=2))
    print("\nValidation classification report")
    if label_mapping is not None:
        target_names = ["white_win", "black_win", "draw"]
        print(
            classification_report(
                y_val,
                y_val_pred,
                labels=[0, 1, 2],
                target_names=target_names,
                zero_division=0,
            )
        )
    else:
        print(classification_report(y_val, y_val_pred, zero_division=0))
    print("\nTest classification report")
    if label_mapping is not None:
        target_names = ["white_win", "black_win", "draw"]
        print(
            classification_report(
                y_test,
                y_test_pred,
                labels=[0, 1, 2],
                target_names=target_names,
                zero_division=0,
            )
        )
    else:
        print(classification_report(y_test, y_test_pred, zero_division=0))

    if args.metrics_output is not None:
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics written to: {args.metrics_output}")


if __name__ == "__main__":
    main()

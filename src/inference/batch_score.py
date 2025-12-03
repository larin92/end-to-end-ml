"""
Batch scoring pipeline for scoring inference datasets.

This script loads the champion model and scores all rows in the inference dataset,
saving predictions to a file and logging the run to MLflow. It automatically reads
the target column name from the training config and removes it if present in the
inference data.

Usage:
    python batch_score.py
    python batch_score.py --config_path path/to/training-config.yml
    python batch_score.py --inference_data_path path/to/inference.parquet
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import get_console_logger
from src.utils.path import ARTIFACTS_DIR, DATA_DIR, PARENT_DIR

load_dotenv()

logger = get_console_logger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Dictionary containing configuration parameters.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_champion_model(model_path: Path) -> object:
    """Load the champion model from artifacts.

    Args:
        model_path: Path to the champion model file.

    Returns:
        The loaded champion model.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Champion model not found at {model_path}")

    logger.info(f"Loading champion model from {model_path}")
    model = joblib.load(model_path)
    return model


def load_inference_data(data_path: Path) -> pd.DataFrame:
    """Load the inference dataset.

    Args:
        data_path: Path to the inference parquet file.

    Returns:
        DataFrame containing the inference data.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Inference data not found at {data_path}")

    logger.info(f"Loading inference data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows for scoring")
    return df


def filter_unknown_categories(
    model: object, features: pd.DataFrame, data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter out rows with unknown categorical values (hotfix for encoder mismatch / data drift).

    As this is a hotfix, use with caution:
    - is ok for testing/analysis
    - is not ok for production, at least in current implementation

    Args:
        model: The trained model pipeline.
        features: DataFrame with feature columns only (no target).
        data: Original DataFrame including target column.

    Returns:
        Tuple of (filtered_features, filtered_data).
    """
    initial_row_count = len(features)

    try:
        # Extract OneHotEncoder from the pipeline to get known categories
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            preprocessor = model.named_steps["preprocessor"]

            if hasattr(preprocessor, "named_transformers_"):
                # Try both 'cat' and 'categorical' as keys
                cat_transformer = preprocessor.named_transformers_.get(
                    "cat"
                ) or preprocessor.named_transformers_.get("categorical")

                if cat_transformer:
                    # Try different ways to access the encoder
                    encoder = None
                    if hasattr(cat_transformer, "named_steps"):
                        # Try different encoder names
                        encoder = (
                            cat_transformer.named_steps.get("onehotencoder")
                            or cat_transformer.named_steps.get("onehot_encoder")
                            or cat_transformer.named_steps.get("encoder")
                        )
                    elif hasattr(cat_transformer, "steps"):
                        # Pipeline with steps attribute
                        for name, step in cat_transformer.steps:
                            if "onehotencoder" in name.lower():
                                encoder = step
                                break
                    elif (
                        hasattr(cat_transformer, "__class__")
                        and "OneHotEncoder" in cat_transformer.__class__.__name__
                    ):
                        encoder = cat_transformer

                    if encoder and hasattr(encoder, "categories_"):
                        # Get categorical column names
                        cat_cols = preprocessor.transformers_[1][2]

                        # Filter rows where all categorical values are in known categories
                        valid_mask = pd.Series(
                            [True] * len(features), index=features.index
                        )
                        for i, col in enumerate(cat_cols):
                            known_cats = encoder.categories_[i]
                            col_mask = features[col].isin(known_cats)
                            valid_mask &= col_mask

                        features = features[valid_mask].copy()
                        data = data[valid_mask].copy()

                        filtered_count = initial_row_count - len(features)
                        if filtered_count > 0:
                            logger.warning(
                                f"Filtered out {filtered_count} rows with unknown categorical values "
                                f"({filtered_count/initial_row_count*100:.1f}% of data)"
                            )
    except Exception as e:
        logger.warning(
            f"Could not filter unknown categories: {e}. Proceeding with all data."
        )

    if len(features) == 0:
        logger.error(
            "All rows were filtered out due to unknown categories. "
            "This indicates inference data is completely incompatible with training data. "
            "The model was trained on different data than what you're trying to score."
        )
        raise ValueError(
            "All rows filtered out - inference data incompatible with model. "
            "Please retrain model with handle_unknown='ignore' or use compatible data."
        )

    return features, data


def score_data(
    model: object,
    data: pd.DataFrame,
    target_col: str = None,
    filter_unknowns: bool = False,
) -> pd.DataFrame:
    """Score the data using the model.

    Args:
        model: The trained model.
        data: DataFrame containing features.
        target_col: Optional name of target column to drop before scoring.
        filter_unknowns: Whether to filter rows with unknown categorical values.

    Returns:
        DataFrame with predictions added.
    """
    logger.info("Scoring data...")

    # Create a copy to avoid modifying original data
    data_copy = data.copy()

    # Check if target column exists (for metrics computation later)
    has_labels = target_col and target_col in data_copy.columns

    # Drop target column if it exists in the data
    if has_labels:
        logger.info(f"Removing target column '{target_col}' from inference data")
        features = data_copy.drop(columns=[target_col])
    else:
        features = data_copy
        logger.warning(
            "True labels not available - will not be able to compute metrics"
        )

    # Filter out rows with unknown categories (hotfix for encoder mismatch / data drift)
    # TODO: see mismatch between:
    # - schemas.py:187, cat_features_ohe_handle_unknown: str = "error"
    # - training-config.yml:75 , cat_features_ohe_handle_unknown: "infrequent_if_exist"
    # TODO: current default inference dataset (src/feature/feature_repo/data/inference.parquet)
    # is incompatible with default champion model (src/training/artifacts/champion_model.pkl)
    if filter_unknowns:
        features, data = filter_unknown_categories(model, features, data)

    # Get predicted probabilities and classes with timing and error handling
    try:
        start_time = time.time()
        predictions_proba = model.predict_proba(features)
        predictions = model.predict(features)
        prediction_time = time.time() - start_time

        logger.info(
            f"Predictions (predict + predict_proba) completed in {prediction_time:.3f}s for {len(features)} rows"
        )
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

    # Create results dataframe with original data
    results = data.copy()
    results["predicted_class"] = predictions
    results["predicted_probability_class_0"] = predictions_proba[:, 0]
    results["predicted_probability_class_1"] = predictions_proba[:, 1]

    logger.info(f"Scored {len(results)} rows")
    return results


def log_to_mlflow(
    predictions: pd.DataFrame,
    output_path: Path,
    model_path: Path,
    target_col: str,
    timestamp: str,
    mlflow_tracking_uri: str = None,
) -> None:
    """Log the batch scoring run to MLflow.

    Args:
        predictions: DataFrame containing predictions.
        output_path: Path where predictions were saved.
        model_path: Path to the champion model used for scoring.
        target_col: Name of target column (for metrics computation).
        timestamp: Timestamp string for run name.
        mlflow_tracking_uri: Optional MLflow tracking URI.
    """
    # Set tracking URI if provided
    if mlflow_tracking_uri:
        mlflow.set_tracking_uri(mlflow_tracking_uri)

    # Set experiment
    mlflow.set_experiment("batch-scoring")

    # Start MLflow run
    with mlflow.start_run(run_name=f"batch_score_{timestamp}"):
        # Log parameters
        num_rows_scored = len(predictions)
        mlflow.log_param("num_rows_scored", num_rows_scored)

        # Log basic prediction statistics
        num_positive_predictions = (predictions["predicted_class"] == 1).sum()
        num_negative_predictions = (predictions["predicted_class"] == 0).sum()
        avg_probability_class_0 = predictions["predicted_probability_class_0"].mean()
        avg_probability_class_1 = predictions["predicted_probability_class_1"].mean()

        mlflow.log_metric("num_positive_predictions", num_positive_predictions)
        mlflow.log_metric("num_negative_predictions", num_negative_predictions)
        mlflow.log_metric("avg_probability_class_0", avg_probability_class_0)
        mlflow.log_metric("avg_probability_class_1", avg_probability_class_1)

        # Compute and log classification metrics if true labels are available
        if target_col not in predictions.columns:
            logger.info(
                "True labels not available - skipping classification metrics computation"
            )
        else:
            logger.info("Computing classification metrics...")
            y_true = predictions[target_col]
            y_pred = predictions["predicted_class"]

            # Compute metrics
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            # Compute confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("true_negatives", tn)
            mlflow.log_metric("true_positives", tp)
            mlflow.log_metric("false_negatives", fn)
            mlflow.log_metric("false_positives", fp)

            logger.info(
                f"Metrics: Accuracy={accuracy:.3f}, F1-score={f1:.3f}, "
                f"Precision={precision:.3f}, Recall={recall:.3f}"
            )

        # Log artifacts
        mlflow.log_artifact(str(output_path), "predictions")
        mlflow.log_artifact(str(model_path), "model")


def main(
    config_path: Path = None,
    inference_data_path: Path = None,
    output_path: Path = None,
    mlflow_tracking_uri: str = None,
    filter_unknown_categories_flag: bool = False,
) -> None:
    """Main function to run the batch scoring pipeline.

    Args:
        config_path: Path to the training config YAML file.
        inference_data_path: Path to the inference dataset.
        output_path: Path to save predictions.
        mlflow_tracking_uri: Optional MLflow tracking URI.
        filter_unknown_categories_flag: Whether to filter rows with unknown categorical values.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Set default paths
    if config_path is None:
        config_path = PARENT_DIR / "config" / "training-config.yml"
    if inference_data_path is None:
        inference_data_path = DATA_DIR / "inference.parquet"
    if output_path is None:
        output_path = DATA_DIR / f"predictions_{timestamp}.parquet"

    # Load config to get target column name and model path
    config = load_config(config_path)
    target_col = config.get("data", {}).get("class_col_name")
    champion_model_name = config.get("modelregistry", {}).get("champion_model_name")
    champion_model_path = ARTIFACTS_DIR / f"{champion_model_name}.pkl"
    logger.debug(f"Target column name from config: {target_col}")
    logger.debug(f"Champion model name from config: {champion_model_name}")
    logger.debug(f"Champion model path: {champion_model_path}")

    # Load model
    model = load_champion_model(champion_model_path)

    # Load inference data
    inference_data = load_inference_data(inference_data_path)

    # Score data and save predictions
    predictions = score_data(
        model,
        inference_data,
        target_col=target_col,
        filter_unknowns=filter_unknown_categories_flag,
    )
    predictions.to_parquet(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")

    # Log to MLflow
    log_to_mlflow(
        predictions,
        output_path,
        champion_model_path,
        target_col,
        timestamp,
        mlflow_tracking_uri,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch scoring pipeline for inference datasets"
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to training config YAML file (default: src/config/training-config.yml)",
    )
    parser.add_argument(
        "--inference_data_path",
        type=str,
        default=None,
        help="Path to the inference dataset (default: data/inference.parquet)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save predictions (default: data/predictions_<timestamp>.parquet)",
    )
    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=None,
        help="MLflow tracking URI (optional)",
    )
    parser.add_argument(
        "--filter_unknown_categories_flag",
        action="store_true",
        help="Filter out rows with unknown categorical values (default: False)",
    )

    args = parser.parse_args()

    # Convert string paths to Path objects if provided
    _config_path = Path(args.config_path) if args.config_path else None
    _inference_data_path = (
        Path(args.inference_data_path) if args.inference_data_path else None
    )
    _output_path = Path(args.output_path) if args.output_path else None

    main(
        config_path=_config_path,
        inference_data_path=_inference_data_path,
        output_path=_output_path,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        filter_unknown_categories_flag=args.filter_unknown_categories_flag,
    )

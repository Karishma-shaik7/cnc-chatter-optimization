"""
Machine Learning Classifier for CNC Chatter State Detection.
Trains a Scikit-Learn Random Forest Classifier on machining and vibration features.
"""

from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from .synthetic_data import (
    LABEL_STABLE,
    LABEL_WARNING,
    LABEL_HIGH_RISK,
    VALID_LABELS,
    generate_synthetic_dataset,
)

FEATURE_COLUMNS = [
    "spindle_speed_rpm",
    "feed_rate_mm_min",
    "depth_of_cut_mm",
    "vibration_rms_g",
    "vibration_peak_g",
    "dominant_frequency_hz",
]


class ChatterClassifier:
    """
    Random Forest Classifier for predicting machining chatter states:
    - Stable (Normal operation)
    - Warning (Incipient chatter / parameter boundary)
    - High Risk (Severe regenerative chatter)
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 12, random_state: int = 42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.model: Optional[RandomForestClassifier] = None
        self.classes_: List[str] = VALID_LABELS
        self.feature_names: List[str] = FEATURE_COLUMNS
        self.feature_importances_: Dict[str, float] = {}
        self.metrics_: Dict[str, Any] = {}
        self.is_trained: bool = False

    def fit(self, df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the classifier on the provided DataFrame.
        Evaluates performance on a hold-out test set and stores metrics.
        """
        for col in self.feature_names:
            if col not in df.columns:
                raise ValueError(f"Missing required feature column: {col}")
        if "chatter_state" not in df.columns:
            raise ValueError("Missing target column: chatter_state")

        X = np.asarray(df[self.feature_names].values, dtype=np.float64)
        y = np.asarray(df["chatter_state"].to_numpy(), dtype=str)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=self.random_state,
            class_weight="balanced",
        )
        self.model.fit(X_train, y_train)
        self.classes_ = list(self.model.classes_)

        # Calculate evaluation metrics on hold-out set
        y_pred = self.model.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        precision, recall, f1, support = precision_recall_fscore_support(
            y_test, y_pred, labels=self.classes_, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=self.classes_)

        # Compute feature importances
        importances = self.model.feature_importances_
        self.feature_importances_ = {
            name: float(imp) for name, imp in zip(self.feature_names, importances)
        }

        # Store detailed metrics
        class_metrics = {}
        for idx, cls in enumerate(self.classes_):
            class_metrics[cls] = {
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1_score": float(f1[idx]),
                "support": int(support[idx]),
            }

        self.metrics_ = {
            "accuracy": acc,
            "test_samples": len(y_test),
            "train_samples": len(y_train),
            "class_metrics": class_metrics,
            "confusion_matrix": cm.tolist(),
            "classes": self.classes_,
        }
        self.is_trained = True
        return self.metrics_

    def predict(self, feature_dict_or_df: Any) -> Dict[str, Any]:
        """
        Predict chatter state and probabilities for a single sample or batch.
        Accepts dict with feature keys or pandas DataFrame.
        Returns prediction details including risk scores.
        """
        if not self.is_trained or self.model is None:
            raise RuntimeError("Model has not been trained yet. Call .fit() first.")

        if isinstance(feature_dict_or_df, dict):
            # Single sample dictionary
            input_vector = np.array([[feature_dict_or_df[col] for col in self.feature_names]])
        elif isinstance(feature_dict_or_df, pd.DataFrame):
            input_vector = feature_dict_or_df[self.feature_names].values
        else:
            input_vector = np.asarray(feature_dict_or_df)
            if input_vector.ndim == 1:
                input_vector = input_vector.reshape(1, -1)

        predicted_classes = self.model.predict(input_vector)
        probabilities = self.model.predict_proba(input_vector)

        results = []
        for i in range(len(predicted_classes)):
            prob_dict = {
                cls: float(probabilities[i][idx])
                for idx, cls in enumerate(self.classes_)
            }
            p_high = prob_dict.get(LABEL_HIGH_RISK, 0.0)
            p_warn = prob_dict.get(LABEL_WARNING, 0.0)
            p_stable = prob_dict.get(LABEL_STABLE, 0.0)

            # Composite chatter risk score (0 to 100)
            # High risk has full weight (1.0), Warning has 0.40 weight
            risk_score = min(100.0, max(0.0, (p_high * 1.0 + p_warn * 0.40) * 100.0))

            results.append({
                "predicted_state": str(predicted_classes[i]),
                "probabilities": prob_dict,
                "risk_score": round(risk_score, 1),
                "confidence": round(float(np.max(probabilities[i])) * 100.0, 1),
                "is_stable": predicted_classes[i] == LABEL_STABLE,
            })

        if isinstance(feature_dict_or_df, dict) or (
            isinstance(feature_dict_or_df, np.ndarray) and feature_dict_or_df.ndim == 1
        ):
            return results[0]
        return results[0] if len(results) == 1 else {"batch_results": results}


def get_trained_default_model(n_samples: int = 1500, random_state: int = 42) -> Tuple[ChatterClassifier, pd.DataFrame]:
    """
    Helper function to generate a synthetic dataset and train a default ChatterClassifier.
    Useful for rapid prototype initialization.
    """
    df = generate_synthetic_dataset(n_samples=n_samples, random_state=random_state)
    clf = ChatterClassifier(random_state=random_state)
    clf.fit(df)
    return clf, df

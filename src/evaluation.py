# evaluation.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
import torch
from src.models.trainers import (
    TrainDNN,
    TrainXGBoost,
    TrainRandomForest,
    compute_minority_weighted_recall,
)
import os
import xgboost as xgb
import joblib


class ModelEvaluator:
    """Manages model evaluation and result visualization"""

    def __init__(self, config):
        self.config = config
        self.dnn_models_dir = (
            "models/DNN/only_regular"
            if config.ONLY_REGULAR
            else "models/DNN/all_clusters"
        )
        self.xgboost_models_dir = (
            "models/XGBoost/only_regular"
            if config.ONLY_REGULAR
            else "models/XGBoost/all_clusters"
        )
        self.randomforest_models_dir = (
            "models/RandomForest/only_regular"
            if config.ONLY_REGULAR
            else "models/RandomForest/all_clusters"
        )

    def evaluate_binary_classifier(self, y_true, y_pred, title="Binary Classifier"):
        """Evaluates a binary classifier and shows results"""
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_per_class = precision_score(
            y_true, y_pred, average=None, zero_division=0
        )
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # Print results
        print(f"{title.upper()} RESULTS:")
        print("Accuracy = {:.4f}".format(accuracy))
        print("Precision = {:.4f}".format(precision_score(y_true, y_pred)))
        print("Recall = {:.4f}".format(recall_score(y_true, y_pred)))
        print("F1-score = {:.4f}".format(f1_score(y_true, y_pred)))
        print(
            "Weighted-recall = {:.4f}".format(
                compute_minority_weighted_recall(y_true, y_pred)
            )
        )
        print("Confusion Matrix:\n{}".format(cm))

        # Create enhanced visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        fig.suptitle(f"{title} - Confusion Matrix", fontsize=16, fontweight="bold")

        # Plot confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{title} (Accuracy: {accuracy:.3f})")

        # Add detailed metrics text below the confusion matrix
        metrics_text = ""
        for class_idx in range(len(precision_per_class)):
            metrics_text += f"Class {class_idx}: Precision={precision_per_class[class_idx]:.3f}, Recall={recall_per_class[class_idx]:.3f}, F1={f1_per_class[class_idx]:.3f}\n"

        # Add overall metrics
        metrics_text += f"\nOverall: Precision={precision_score(y_true, y_pred):.3f}, Recall={recall_score(y_true, y_pred):.3f}, F1={f1_score(y_true, y_pred):.3f}"

        ax.text(
            0.5,
            -0.20,
            metrics_text,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.8),
        )

        plt.tight_layout()
        plt.show()

    def train_and_evaluate_dnn(self, X, y):
        """Trains and evaluates DNN model with cross-validation"""
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.SEED,
            shuffle=True,
            stratify=y,
        )

        n_splits = 4
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.config.SEED
        )

        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.LongTensor(y_test)

        # Check if models need to be trained
        models_exist = all(
            os.path.exists(
                os.path.join(self.dnn_models_dir, f"dnn_model_fold_{fold}.pth")
            )
            for fold in range(4)
        )

        if not models_exist:
            print("Training DNN models...")
            val_recalls = []
            fold = 0
            for train_idx, val_idx in skf.split(X_tensor, y_tensor):
                print(f"\n--- Fold {fold + 1}/{n_splits} ---")
                X_train_fold, X_val_fold = X_tensor[train_idx], X_tensor[val_idx]
                y_train_fold, y_val_fold = y_tensor[train_idx], y_tensor[val_idx]

                dnn_trainer = TrainDNN(
                    X_train_fold, X_val_fold, y_train_fold, y_val_fold
                )
                history, val_recall = dnn_trainer.train_neural_network(
                    self.config.LEARNING_RATE_DNN,
                    self.config.N_EPOCHS_DNN,
                    self.dnn_models_dir,
                    self.config.BATCH_SIZE,
                    self.config.WEIGHT_DECAY,
                    fold,
                )
                val_recalls.append(val_recall)
                fold += 1

            print(f"\nCross-validation recall scores: {val_recalls}")
            print(
                f"Mean CV recall: {np.mean(val_recalls):.4f} (+/- {np.std(val_recalls) * 2:.4f})"
            )
        else:
            print("DNN models already exist, loading...")

        # Test phase
        print("\n=== TESTING DNN PHASE ===")
        test_predictions = []
        test_probabilities = []

        for fold in range(4):
            model_path = os.path.join(self.dnn_models_dir, f"dnn_model_fold_{fold}.pth")
            dnn_trainer = TrainDNN(X_tensor, None, y_tensor, None)
            y_pred, probabilities = dnn_trainer.test_dnn(
                X_test_tensor, y_test_tensor, model_path
            )
            test_predictions.append(y_pred.numpy())
            test_probabilities.append(probabilities.numpy())

        # ensamble

        test_predictions = np.array(test_predictions)
        test_probabilities = np.array(test_probabilities)
        hard_voting_pred = []
        soft_voting_pred = []

        for i in range(y_test.shape[0]):
            predictions = test_predictions[:, i]
            proba = test_probabilities[:, i]

            hard_voting_pred.append(np.argmax(np.bincount(predictions)))
            soft_voting_pred.append(np.argmax(np.sum(proba, axis=0) / 4))

        return hard_voting_pred, soft_voting_pred, y_test

    def train_and_evaluate_xgboost(self, X, y, max_depth, learning_rate):
        """Trains and evaluates XGBoost model with cross-validation"""
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.SEED,
            shuffle=True,
            stratify=y,
        )

        n_splits = 4
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.config.SEED
        )

        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)

        # Check if models need to be trained
        models_exist = all(
            os.path.exists(os.path.join(self.xgboost_models_dir, f"fold_{fold}.json"))
            for fold in range(4)
        )

        if not models_exist:
            print("Training XGBoost models...")
            val_recalls = []
            xgb_models = []
            fold = 0
            for train_idx, val_idx in skf.split(X_tensor, y_tensor):
                print(f"\n--- Fold {fold + 1}/{n_splits} ---")
                X_train_fold, X_val_fold = X_tensor[train_idx], X_tensor[val_idx]
                y_train_fold, y_val_fold = y_tensor[train_idx], y_tensor[val_idx]

                xgb_trainer = TrainXGBoost(
                    X_train_fold, X_val_fold, y_train_fold, y_val_fold
                )
                xgb_model, val_recall = xgb_trainer.train_xgboost(
                    self.xgboost_models_dir,
                    max_depth=max_depth,
                    lr=learning_rate,
                    fold=fold,
                )
                val_recalls.append(val_recall)
                xgb_models.append(xgb_model)
                fold += 1

            print(f"\nCross-validation XGBoost recall scores: {val_recalls}")
            print(
                f"Mean XGBoost CV recall: {np.mean(val_recalls):.4f} (+/- {np.std(val_recalls) * 2:.4f})"
            )
        else:
            print("XGBoost models already exist, loading...")

        # Load models and make predictions
        xgb_models = []
        for fold in range(4):
            model_path = os.path.join(self.xgboost_models_dir, f"fold_{fold}.json")
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model(model_path)
            xgb_models.append(xgb_model)

        # Test phase
        print("\n=== TESTING XGBOOST PHASE ===")
        test_predictions = []
        test_probabilities = []
        for xgb_model in xgb_models:
            xgb_trainer = TrainXGBoost(X_tensor, None, y_tensor, None)
            y_pred, y_pred_proba = xgb_trainer.test_xgboost(X_test_tensor, xgb_model)
            test_predictions.append(y_pred)
            test_probabilities.append(y_pred_proba)

        # ensamble
        test_predictions = np.array(test_predictions)
        test_probabilities = np.array(test_probabilities)
        hard_voting_pred = []
        soft_voting_pred = []

        for i in range(y_test.shape[0]):
            predictions = test_predictions[:, i]
            proba = test_probabilities[:, i]

            hard_voting_pred.append(np.argmax(np.bincount(predictions)))
            soft_voting_pred.append(np.argmax(np.sum(proba, axis=0) / 4))

        return hard_voting_pred, soft_voting_pred, y_test

    def train_and_evaluate_randomforest(self, X, y):
        """Trains and evaluates Random Forest model with cross-validation"""

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.TEST_SIZE,
            random_state=self.config.SEED,
            shuffle=True,
            stratify=y,
        )

        n_splits = 4
        skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=self.config.SEED
        )

        X_tensor = torch.FloatTensor(X_train)
        y_tensor = torch.LongTensor(y_train)
        X_test_tensor = torch.FloatTensor(X_test)

        # Check if models need to be trained
        models_exist = all(
            os.path.exists(
                os.path.join(self.randomforest_models_dir, f"fold_{fold}.pkl")
            )
            for fold in range(4)
        )

        if not models_exist:
            print("Training Random Forest models...")
            val_recalls = []
            rf_models = []
            fold = 0
            for train_idx, val_idx in skf.split(X_tensor, y_tensor):
                print(f"\n--- Fold {fold + 1}/{n_splits} ---")
                X_train_fold, X_val_fold = X_tensor[train_idx], X_tensor[val_idx]
                y_train_fold, y_val_fold = y_tensor[train_idx], y_tensor[val_idx]

                rf_trainer = TrainRandomForest(
                    X_train_fold, X_val_fold, y_train_fold, y_val_fold
                )
                rf_model, val_recall = rf_trainer.train_randomforest(
                    self.randomforest_models_dir,
                    n_estimators=100,
                    max_depth=5,
                    fold=fold,
                )
                val_recalls.append(val_recall)
                rf_models.append(rf_model)
                fold += 1

            print(f"\nCross-validation Random Forest recall scores: {val_recalls}")
            print(
                f"Mean Random Forest CV recall: {np.mean(val_recalls):.4f} (+/- {np.std(val_recalls) * 2:.4f})"
            )
        else:
            print("Random Forest models already exist, loading...")

        # Load models and make predictions
        rf_models = []
        for fold in range(4):
            model_path = os.path.join(self.randomforest_models_dir, f"fold_{fold}.pkl")
            rf_model = joblib.load(model_path)
            rf_models.append(rf_model)

        # Test phase
        print("\n=== TESTING RANDOM FOREST PHASE ===")
        test_predictions = []
        test_probabilities = []
        for rf_model in rf_models:
            rf_trainer = TrainRandomForest(X_tensor, None, y_tensor, None)
            y_pred, y_pred_proba = rf_trainer.test_randomforest(X_test_tensor, rf_model)
            test_predictions.append(y_pred)
            test_probabilities.append(y_pred_proba)

        # Ensemble
        test_predictions = np.array(test_predictions)
        test_probabilities = np.array(test_probabilities)
        hard_voting_pred = []
        soft_voting_pred = []

        for i in range(y_test.shape[0]):
            predictions = test_predictions[:, i]
            proba = test_probabilities[:, i]

            hard_voting_pred.append(np.argmax(np.bincount(predictions)))
            soft_voting_pred.append(np.argmax(np.sum(proba, axis=0) / 4))

        return hard_voting_pred, soft_voting_pred, y_test

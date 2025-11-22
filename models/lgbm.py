import os
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from lightgbm import Dataset, train
import joblib

class LightGBMTrainer:
    def __init__(self, model_path="model_ckpts/lightgbm", image_path="images/models/lightgbm"):
        self.model_path = model_path
        self.image_path = image_path
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)
        self.model = None
        self.metric_history = {'precision': [], 'recall': [], 'f1': []}

    def load_data(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.features = [c for c in train_df.columns if c != "target"]
        self.target_col = "target"
        logging.info(f"Loaded train size: {len(train_df)}, test size: {len(test_df)}")

    def train(self, params=None, num_boost_round=500, early_stopping_rounds=50):
        if params is None:
            params = {
                "objective": "binary",
                "metric": "auc",
                "learning_rate": 0.1,
                "num_leaves": 256,
                "feature_fraction": 0.9,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "max_depth": -1,
                "verbosity": -1
            }

        logging.info("Start training LightGBM model")

        train_data = Dataset(self.train_df[self.features], label=self.train_df[self.target_col])
        test_data = Dataset(self.test_df[self.features], label=self.test_df[self.target_col])

        from lightgbm import early_stopping, log_evaluation

        # Custom callback to record precision, recall, F1 over iterations
        def record_metrics(env):
            y_prob = env.model.predict(self.test_df[self.features])
            y_pred = (y_prob >= 0.5).astype(int)
            self.metric_history['precision'].append(precision_score(self.test_df[self.target_col], y_pred))
            self.metric_history['recall'].append(recall_score(self.test_df[self.target_col], y_pred))
            self.metric_history['f1'].append(f1_score(self.test_df[self.target_col], y_pred))

        self.model = train(
            params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, test_data],
            valid_names=["train", "test"],
            callbacks=[
                early_stopping(stopping_rounds=early_stopping_rounds),
                log_evaluation(period=50),
                record_metrics
            ]
        )

        model_file = os.path.join(self.model_path, "lightgbm_model.pkl")
        joblib.dump(self.model, model_file)
        logging.info(f"Model saved to {model_file}")

    def load_model(self, model_file):
        logging.info(f"Loading model from {model_file}")
        self.model = joblib.load(model_file)

    def evaluate(self):
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")

        logging.info("Start evaluation")

        y_true = self.test_df[self.target_col].values
        y_prob = self.model.predict(self.test_df[self.features])

        auc_score = roc_auc_score(y_true, y_prob)
        logging.info(f"AUC: {auc_score}")

        y_pred = (y_prob >= 0.5).astype(int)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        logging.info(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

        self._plot_roc_curve(y_true, y_prob)
        self._plot_prf_curve()

    def _plot_roc_curve(self, y_true, y_prob):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc_score(y_true, y_prob):.4f})")
        plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()

        save_path = os.path.join(self.image_path, "roc_curve.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"ROC curve saved to {save_path}")

    def _plot_prf_curve(self):
        plt.figure(figsize=(8, 6))
        plt.plot(self.metric_history['precision'], label='Precision')
        plt.plot(self.metric_history['recall'], label='Recall')
        plt.plot(self.metric_history['f1'], label='F1')
        plt.xlabel('Iteration')
        plt.ylabel('Score')
        plt.title('Precision, Recall, F1 over Training Rounds')
        plt.legend()
        plt.grid(True)

        save_path = os.path.join(self.image_path, "precision_recall_f1_over_rounds.png")
        plt.savefig(save_path)
        plt.close()
        logging.info(f"Precision/Recall/F1 over rounds chart saved to {save_path}")

    def tune(self):
        logging.info("Hyperparameter tuning placeholder.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    train_df = pd.read_csv("data/processed_data/train_processed_data.csv")
    test_df = pd.read_csv("data/processed_data/test_processed_data.csv")

    trainer = LightGBMTrainer(model_path="model_ckpts/lightgbm", image_path="images/models/lightgbm")
    trainer.load_data(train_df, test_df)
    trainer.train()
    trainer.evaluate()
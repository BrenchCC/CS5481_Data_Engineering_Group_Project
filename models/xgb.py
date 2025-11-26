import os
import json
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve, auc, log_loss
import xgboost as xgb
import joblib
from tqdm import tqdm


# custom tqdm + logging callback
class LoggingCallback(xgb.callback.TrainingCallback):
    def __init__(self, predictor, dtrain, y_train, dval, y_val, total_rounds):
        self.predictor = predictor
        self.dtrain = dtrain
        self.y_train = y_train
        self.dval = dval
        self.y_val = y_val
        self.total_rounds = total_rounds
        self.pbar = tqdm(total=total_rounds, desc="Training XGBoost", ncols=80)

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)

        y_train_pred_prob = model.predict(self.dtrain, iteration_range=(0, epoch + 1))
        y_val_pred_prob = model.predict(self.dval, iteration_range=(0, epoch + 1))

        self.predictor._update_history(self.y_train, y_train_pred_prob, self.y_val, y_val_pred_prob)

        current_val_auc = self.predictor.history["test_auc"][-1]

        # logging at interval
        if (epoch + 1) % self.predictor.log_interval == 0:
            if current_val_auc > self.predictor.best_val_auc:
                self.predictor.best_val_auc = current_val_auc
                self.predictor.best_iteration = epoch
                self.predictor._save_model(best=True)

            y_train_pred = (y_train_pred_prob >= self.predictor.threshold).astype(int)
            y_val_pred = (y_val_pred_prob >= self.predictor.threshold).astype(int)
            train_auc = roc_auc_score(self.y_train, y_train_pred_prob)
            val_auc = current_val_auc
            train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
                self.y_train, y_train_pred, average='binary'
            )
            val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
                self.y_val, y_val_pred, average='binary'
            )

            logging.info(
                f"Round {epoch + 1}: "
                f"Train AUC={train_auc:.4f}, Precision={train_prec:.4f}, Recall={train_rec:.4f}, F1={train_f1:.4f} | "
                f"Val AUC={val_auc:.4f}, Precision={val_prec:.4f}, Recall={val_rec:.4f}, F1={val_f1:.4f}"
            )

        return False

    def after_training(self, model):
        self.pbar.close()
        return model


# XGB Predictor
class XGBMusicPredictor:
    def __init__(
            self,
            model_path: str = "model_ckpts/xgboost",
            image_path: str = "images/models/xgboost",
            threshold: float = 0.5,
            xgb_params: dict = None,
            n_estimators: int = 200,
            early_stopping_rounds: int = 20,
            random_state: int = 42,
            log_interval: int = 50
    ):
        self.model_path = model_path
        self.image_path = image_path
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)

        self.threshold = threshold
        self.n_estimators = n_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self.log_interval = log_interval

        self.xgb_params = xgb_params or {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "learning_rate": 0.1,
            "max_depth": 8,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "auto",
            "random_state": self.random_state,
            "n_jobs": -1
        }

        self.model = None
        self.evals_result = {}

        self.history = {
            "train_loss": [],
            "test_loss": [],
            "train_auc": [],
            "test_auc": [],
            "precision": [],
            "recall": [],
            "f1": []
        }
        self.image_dir = self.image_path

        self.best_val_auc = 0.0
        self.best_iteration = 0

    def train(self, train_df: pd.DataFrame, val_df: pd.DataFrame):
        logging.info("Preparing data for XGBoost training...")
        X_train = train_df.drop(columns=['target'])
        y_train = train_df['target'].values
        X_val = val_df.drop(columns=['target'])
        y_val = val_df['target'].values

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)

        logging.info("Start training XGBoost model...")

        callback = LoggingCallback(
            predictor=self,
            dtrain=dtrain,
            y_train=y_train,
            dval=dval,
            y_val=y_val,
            total_rounds=self.n_estimators
        )

        self.model = xgb.train(
            params=self.xgb_params,
            dtrain=dtrain,
            num_boost_round=self.n_estimators,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=self.evals_result,
            verbose_eval=False,
            callbacks=[callback]
        )

        logging.info(
            f"Training completed. Best iteration: {self.best_iteration}, Best Val AUC: {self.best_val_auc:.4f}"
        )
        # self._save_model()
        self._save_training_configs()

    def predict(self, df: pd.DataFrame):
        X = df.drop(columns=['target'], errors='ignore')
        dmatrix = xgb.DMatrix(X)
        y_pred_prob = self.model.predict(dmatrix, iteration_range=(0, self.model.best_iteration + 1))
        y_pred = (y_pred_prob >= self.threshold).astype(int)
        return y_pred, y_pred_prob

    def evaluate(self, df: pd.DataFrame):
        logging.info("Evaluating model...")
        y_true = df['target'].values
        y_pred, y_pred_prob = self.predict(df)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
        auc_score = roc_auc_score(y_true, y_pred_prob)
        logging.info(
            f"Final Evaluation - Precision: {precision:.4f}, Recall: {recall:.4f}, F1={f1:.4f}, AUC={auc_score:.4f}"
        )
        return precision, recall, f1, auc_score

    def _update_history(self, y_train, y_train_pred_prob, y_val, y_val_pred_prob):
        self.history["train_auc"].append(roc_auc_score(y_train, y_train_pred_prob))
        self.history["test_auc"].append(roc_auc_score(y_val, y_val_pred_prob))
        self.history["train_loss"].append(log_loss(y_train, y_train_pred_prob))
        self.history["test_loss"].append(log_loss(y_val, y_val_pred_prob))
        y_val_pred = (y_val_pred_prob >= self.threshold).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_val_pred, average="binary")
        self.history["precision"].append(prec)
        self.history["recall"].append(rec)
        self.history["f1"].append(f1)

    # plotting
    def plot_metrics(self, save_plot: bool = True):
        if not self.history["train_loss"]:
            logging.warning("No training history available. Train the model first.")
            return

        num_epochs = len(self.history["train_loss"])
        epochs = range(1, num_epochs + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        ax1.plot(epochs, self.history["train_loss"], label="Train Loss", linewidth=2)
        ax1.plot(epochs, self.history["test_loss"], label="Test Loss", linewidth=2)
        ax1.set_title("Training and Test Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Log Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2.plot(epochs, self.history["train_auc"], label="Train AUC", linewidth=2)
        ax2.plot(epochs, self.history["test_auc"], label="Test AUC", linewidth=2)
        ax2.set_title("Training and Test AUC")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("AUC")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3.plot(epochs, self.history["precision"], label="Precision", linewidth=2)
        ax3.plot(epochs, self.history["recall"], label="Recall", linewidth=2)
        ax3.plot(epochs, self.history["f1"], label="F1-Score", linewidth=2)
        ax3.set_title("Precision, Recall, and F1-Score")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Score")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4.plot(epochs, self.history["test_auc"], label="AUC", linewidth=2)
        ax4.plot(epochs, self.history["precision"], label="Precision", linewidth=2)
        ax4.plot(epochs, self.history["recall"], label="Recall", linewidth=2)
        ax4.plot(epochs, self.history["f1"], label="F1-Score", linewidth=2)
        ax4.set_title("All Test Metrics")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Score")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        if save_plot:
            plot_path = os.path.join(self.image_dir, "training_metrics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logging.info(f"Metrics plot saved to: {plot_path}")
        # plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, save_plot: bool = True) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_plot:
            plot_path = os.path.join(self.image_dir, "roc_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            logging.info(f"ROC curve plot saved to: {plot_path}")
        # plt.show()
        return roc_auc

    def _save_model(self, best: bool = False):
        filename = "best_model.pkl" if best else "xgb_model.pkl"
        path = os.path.join(self.model_path, filename)
        joblib.dump(self.model, path)
        logging.info(f"{'Best' if best else 'Current'} model saved to {path}")

    def load_model(self, model_file: str = None):
        model_file = model_file or os.path.join(self.model_path, "best_model.pkl")
        self.model = joblib.load(model_file)
        logging.info(f"Model loaded from {model_file}")

    def _save_training_configs(self):
        configs = {
            "xgb_params": self.xgb_params,
            "threshold": self.threshold,
            "n_estimators": self.n_estimators,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_state": self.random_state,
            "log_interval": self.log_interval
        }
        config_path = os.path.join(self.model_path, "training_configs.json")
        with open(config_path, "w") as f:
            json.dump(configs, f, indent=4)
        logging.info(f"Training configs saved to {config_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="model_ckpts/xgboost")
    parser.add_argument("--image_path", type=str, default="images/models/xgboost")

    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--early_stopping_rounds", type=int, default=50)
    parser.add_argument("--log_interval", type=int, default=50)

    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--max_depth", type=int, default=8)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample_bytree", type=float, default=0.8)
    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)

    return parser.parse_args()


def build_xgboost_trainer_params(args):
    return {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "tree_method": args.tree_method,
        "random_state": args.random_state,
        "n_jobs": args.n_jobs
    }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    logging.info("Loading processed data...")
    train_df = pd.read_csv("data/processed_data/train_processed_data.csv")
    test_df = pd.read_csv("data/processed_data/test_processed_data.csv")

    args = parse_args()
    train_params = build_xgboost_trainer_params(args)
    logging.info("Building xgboost model...")

    predictor = XGBMusicPredictor(
        model_path=args.model_path,
        image_path=args.image_path,
        threshold=args.threshold,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        log_interval=args.log_interval,
        xgb_params=train_params
    )

    predictor.train(train_df, test_df)
    predictor.evaluate(test_df)

    predictor.plot_metrics()
    y_pred, y_pred_prob = predictor.predict(test_df)
    predictor.plot_roc_curve(test_df["target"].values, y_pred_prob)
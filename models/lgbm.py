import os
import copy
import json
import warnings
import logging
import argparse

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score, auc, log_loss
from lightgbm import Dataset, train as lgb_train, record_evaluation
import lightgbm as lgb
from tqdm import tqdm

warnings.filterwarnings('ignore')


class LightGBMTrainer:
    def __init__(self, model_path="model_ckpts/lightgbm", image_path="images/models/lightgbm"):
        self.model_path = model_path
        self.image_path = image_path
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs(self.image_path, exist_ok=True)
        self.model = None
        self.evals_result = {}
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'train_auc': [],
            'test_auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        self.booster_history = []

    def load_data(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df
        self.features = [c for c in train_df.columns if c != "target"]
        self.target_col = "target"
        logging.info(f"Loaded train size: {len(train_df)}, test size: {len(test_df)}")

    def params_config_save(self, params, num_boost_round, early_stopping_rounds):
        params_record = params.copy()
        params_record["num_boost_round"] = num_boost_round
        params_record["early_stopping_rounds"] = early_stopping_rounds
        with open(os.path.join(self.model_path, "training_configs.json"), "w") as f:
            f.write(json.dumps(params_record, indent=2))

    ##################################################################
    #                   ⭐ tqdm + step-by-step training
    ##################################################################
    def train(self, params=None, num_boost_round=1000, early_stopping_rounds=100):
        if params is None:
            params = {
                "objective": "binary",
                "boosting": "gbdt",
                "metric": ["auc", "binary_logloss"],
                "learning_rate": 0.1,
                "num_leaves": 100,
                "min_data_in_leaf": 1000,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 1,
                "bagging_seed": 42,
                "max_depth": -1,
                "lambda_l1": 0.5,
                "lambda_l2": 2.0,
                "min_split_gain": 0.01,
                "max_bin": 255,
                "verbosity": 1,
            }

        logging.info("Start training LightGBM model")
        logging.info(f"params: {params}")
        logging.info(f"training_configs saving path: {os.path.join(self.model_path, 'training_configs.json')}")
        self.params_config_save(params, num_boost_round, early_stopping_rounds)

        train_data = lgb.Dataset(
            self.train_df[self.features],
            label=self.train_df[self.target_col],
            free_raw_data=False
        )
        valid_data = lgb.Dataset(
            self.test_df[self.features],
            label=self.test_df[self.target_col],
            free_raw_data=False
        )

        self.booster_history = []
        self.evals_result = {}

        # -------------------- callback for recording --------------------
        def record_metrics(model, iteration):
            y_true = self.test_df[self.target_col].values
            y_prob = model.predict(self.test_df[self.features])
            y_pred = (y_prob >= 0.5).astype(int)

            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)

            self.history['precision'].append(precision)
            self.history['recall'].append(recall)
            self.history['f1'].append(f1)

            if iteration % 100 == 0 or iteration == num_boost_round - 1:
                try:
                    self.booster_history.append((iteration, copy.deepcopy(model)))
                except Exception:
                    pass

        # -------------------- manual incremental training --------------------
        model = None
        best_iteration = None
        best_score = -1
        not_improved = 0

        pbar = tqdm(range(num_boost_round), desc="Training", ncols=120)

        for i in pbar:
            model = lgb_train(
                params,
                train_data,
                num_boost_round=1,                  # 每次只训练一轮
                valid_sets=[train_data, valid_data],
                valid_names=["train", "valid"],
                init_model=model,                   # 接上一次训练
                keep_training_booster=True,         # 允许继续训练
                callbacks=[record_evaluation(self.evals_result)]
            )

            # 记录分类指标（precision/recall/f1）
            record_metrics(model, i)

            # Early stopping 手工实现
            current_score = self.evals_result["valid"]["auc"][-1]
            if current_score > best_score:
                best_score = current_score
                best_iteration = i
                not_improved = 0
            else:
                not_improved += 1

            pbar.set_postfix({
                "AUC": f"{current_score:.4f}",
                "Best": f"{best_score:.4f}",
                "Patience": f"{not_improved}/{early_stopping_rounds}"
            })

            if not_improved >= early_stopping_rounds:
                logging.info(f"Early stopped at iteration {i}. Best iteration = {best_iteration}")
                break

        self.model = model

        # -------------------- save model --------------------
        model_file = os.path.join(self.model_path, "lightgbm_model.pkl")
        joblib.dump(self.model, model_file)
        logging.info(f"Model saved to {model_file}")

        self._record_training_history()
        self.plot_metrics()

    ##################################################################

    def _record_training_history(self):
        temp_classification = {
            'precision': self.history['precision'].copy(),
            'recall': self.history['recall'].copy(),
            'f1': self.history['f1'].copy()
        }

        self.history = {key: [] for key in self.history}

        if "train" in self.evals_result and "binary_logloss" in self.evals_result["train"]:
            self.history['train_loss'] = self.evals_result["train"]["binary_logloss"]
        if "valid" in self.evals_result and "binary_logloss" in self.evals_result["valid"]:
            self.history['test_loss'] = self.evals_result["valid"]["binary_logloss"]

        if "train" in self.evals_result and "auc" in self.evals_result["train"]:
            self.history['train_auc'] = self.evals_result["train"]["auc"]
        if "valid" in self.evals_result and "auc" in self.evals_result["valid"]:
            self.history['test_auc'] = self.evals_result["valid"]["auc"]

        self.history['precision'] = temp_classification['precision']
        self.history['recall'] = temp_classification['recall']
        self.history['f1'] = temp_classification['f1']

    def load_model(self, model_file):
        logging.info(f"Loading model from {model_file}")
        self.model = joblib.load(model_file)

    def evaluate(self):
        if self.model is None:
            raise ValueError("Model is not loaded or trained.")

        logging.info("Start evaluation")

        y_true = self.test_df[self.target_col].values
        y_prob = self.model.predict(self.test_df[self.features])

        auc_score_val = roc_auc_score(y_true, y_prob)
        logging.info(f"AUC: {auc_score_val}")

        y_pred = (y_prob >= 0.5).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        logging.info(f"Precision: {precision}, Recall: {recall}, F1: {f1}")

        test_loss = log_loss(y_true, y_prob)
        logging.info(f"Test Log Loss: {test_loss}")

        self.plot_roc_curve(y_true, y_prob)

    def plot_metrics(self, save_plot: bool = True) -> None:
        has_loss_data = bool(self.history['train_loss'] and self.history['test_loss'])
        has_auc_data = bool(self.history['train_auc'] and self.history['test_auc'])
        has_classification_data = bool(self.history['precision'])

        if not (has_loss_data or has_auc_data or has_classification_data):
            logging.warning("No training history available. Train the model first.")
            return

        lengths = [
            len(self.history.get('train_loss', [])),
            len(self.history.get('test_loss', [])),
            len(self.history.get('train_auc', [])),
            len(self.history.get('test_auc', [])),
            len(self.history.get('precision', []))
        ]
        min_length = min(length for length in lengths if length > 0)

        if min_length == 0:
            logging.warning("No valid training history data.")
            return

        if has_loss_data:
            epochs = range(1, min_length + 1)
        else:
            epochs = range(1, len(self.history['precision']) + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        if has_loss_data and len(self.history['train_loss']) >= min_length:
            ax1.plot(epochs, self.history['train_loss'][:min_length], label='Train Loss', linewidth=2)
            ax1.plot(epochs, self.history['test_loss'][:min_length], label='Test Loss', linewidth=2)
            ax1.set_title('Training and Test Loss')
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Log Loss')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        if has_auc_data and len(self.history['train_auc']) >= min_length:
            ax2.plot(epochs, self.history['train_auc'][:min_length], label='Train AUC', linewidth=2)
            ax2.plot(epochs, self.history['test_auc'][:min_length], label='Test AUC', linewidth=2)
            ax2.set_title('Training and Test AUC')
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('AUC')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        if has_classification_data and len(self.history['precision']) >= min_length:
            classification_epochs = range(1, len(self.history['precision'][:min_length]) + 1)
            ax3.plot(classification_epochs, self.history['precision'][:min_length], label='Precision', linewidth=2)
            ax3.plot(classification_epochs, self.history['recall'][:min_length], label='Recall', linewidth=2)
            ax3.plot(classification_epochs, self.history['f1'][:min_length], label='F1-Score', linewidth=2)
            ax3.set_title('Precision, Recall, and F1-Score')
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Score')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        if has_auc_data and has_classification_data:
            min_len = min(len(self.history['test_auc']), len(self.history['precision']))
            comparison_epochs = range(1, min_len + 1)

            ax4.plot(comparison_epochs, self.history['test_auc'][:min_len], label='AUC', linewidth=2)
            ax4.plot(comparison_epochs, self.history['precision'][:min_len], label='Precision', linewidth=2)
            ax4.plot(comparison_epochs, self.history['recall'][:min_len], label='Recall', linewidth=2)
            ax4.plot(comparison_epochs, self.history['f1'][:min_len], label='F1-Score', linewidth=2)
            ax4.set_title('All Test Metrics')
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Score')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.image_path, "training_metrics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info("Metrics plot saved to: %s", plot_path)

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, save_plot: bool = True) -> float:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}", linewidth=2)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_plot:
            plot_path = os.path.join(self.image_path, "roc_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info("ROC curve plot saved to: %s", plot_path)

        return roc_auc


##################################################################
#                        argparse + params
##################################################################
def parse_args():
    parser = argparse.ArgumentParser(
        description="LightGBM binary classification training"
    )
    parser.add_argument("--model_path", type=str, default="model_ckpts/lightgbm_3000rounds_256leaves")
    parser.add_argument("--image_path", type=str, default="images/models/lightgbm_3000rounds_256leaves")
    parser.add_argument("--learning_rate", type=float, default=0.05)
    parser.add_argument("--num_leaves", type=int, default=256)
    parser.add_argument("--min_data_in_leaf", type=int, default=1000)
    parser.add_argument("--feature_fraction", type=float, default=0.8)
    parser.add_argument("--bagging_fraction", type=float, default=0.8)
    parser.add_argument("--bagging_freq", type=int, default=1)
    parser.add_argument("--bagging_seed", type=int, default=42)
    parser.add_argument("--max_depth", type=int, default=-1)
    parser.add_argument("--lambda_l1", type=float, default=0.5)
    parser.add_argument("--lambda_l2", type=float, default=2.0)
    parser.add_argument("--min_split_gain", type=float, default=0.01)
    parser.add_argument("--max_bin", type=int, default=255)
    parser.add_argument("--verbosity", type=int, default=1)
    parser.add_argument("--num_boost_round", type=int, default=3000)
    parser.add_argument("--early_stopping_rounds", type=int, default=100)
    return parser.parse_args()


def build_lgb_params(args):
    return {
        "objective": "binary",
        "boosting": "gbdt",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "bagging_seed": args.bagging_seed,
        "max_depth": args.max_depth,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "min_split_gain": args.min_split_gain,
        "max_bin": args.max_bin,
        "verbosity": args.verbosity,
    }


##################################################################
#                           main
##################################################################
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    args = parse_args()
    params = build_lgb_params(args)

    train_df = pd.read_csv("data/processed_data/train_processed_data.csv")
    test_df = pd.read_csv("data/processed_data/test_processed_data.csv")

    trainer = LightGBMTrainer(
        model_path=args.model_path,
        image_path=args.image_path,
    )

    trainer.load_data(train_df, test_df)
    trainer.train(
        params=params,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds
    )

    trainer.evaluate()

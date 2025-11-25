import os
import json
import logging
import argparse
import pickle
from typing import Dict, List, Tuple, Optional, Union

import numpy as np
import pandas as pd
from lightfm import LightFM
from sklearn.metrics import log_loss, roc_auc_score, precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.sparse import coo_matrix
import warnings

warnings.filterwarnings('ignore')


class LightFMRecommender:
    """
    A recommender system class using LightFM for collaborative filtering.
    """

    def __init__(self, model_dir: str = "model_ckpts/lightfm",
                 image_dir: str = "images/models/lightfm",
                 threshold: float = 0.5,
                 **model_params):
        """
        Initialize the LightFM recommender.

        Args:
            model_dir: Directory to save model checkpoints and mappings
            image_dir: Directory to save evaluation plots
            threshold: Classification threshold for binary metrics
            **model_params: LightFM model parameters
        """
        self.model_dir = model_dir
        self.image_dir = image_dir
        self.threshold = threshold
        self.model_params = model_params or {
            'loss': 'logistic',
            'no_components': 30,
            'learning_rate': 0.05,
            'random_state': 42
        }

        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)

        # Initialize attributes
        self.model = None
        self.user_id_map = None
        self.song_id_map = None
        self.num_users = 0
        self.num_items = 0
        self.is_trained = False

        self.history = {
            'train_loss': [],
            'test_loss': [],
            'train_auc': [],
            'test_auc': [],
            'precision': [],
            'recall': [],
            'f1': []
        }

        logging.info("LightFMRecommender initialized with parameters: %s", self.model_params)

    def _create_mappings(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """Create user and item mappings from training and test data."""
        logging.info("Creating user and item mappings...")

        all_msno = np.union1d(train_df['msno'].unique(), test_df['msno'].unique())
        all_song_id = np.union1d(train_df['song_id'].unique(), test_df['song_id'].unique())

        self.user_id_map = {user: idx for idx, user in enumerate(all_msno)}
        self.song_id_map = {song: idx for idx, song in enumerate(all_song_id)}
        self.num_users = len(self.user_id_map)
        self.num_items = len(self.song_id_map)

        logging.info("Mappings created: %d users, %d items", self.num_users, self.num_items)

    def _create_interaction_matrix(self, df: pd.DataFrame, target_col: str = 'target') -> coo_matrix:
        """Create interaction matrix from DataFrame."""
        user_indices = df['msno'].map(self.user_id_map).astype(int)
        song_indices = df['song_id'].map(self.song_id_map).astype(int)
        targets = df[target_col].values

        return coo_matrix((targets, (user_indices, song_indices)),
                          shape=(self.num_users, self.num_items))

    def _calculate_metrics(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        auc = roc_auc_score(y_true, y_pred_proba)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )

        return {
            'auc': auc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def train(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
              num_epochs: int = 20, save_model: bool = True) -> Dict[str, List[float]]:
        """
        Train the LightFM model.

        Args:
            train_df: Training data DataFrame
            test_df: Test data DataFrame
            num_epochs: Number of training epochs
            save_model: Whether to save the trained model

        Returns:
            Training history dictionary
        """
        logging.info("Starting model training for %d epochs", num_epochs)

        # Create mappings
        self._create_mappings(train_df, test_df)

        # Create interaction matrices
        train_coo = self._create_interaction_matrix(train_df)
        test_coo = self._create_interaction_matrix(test_df)

        # Get indices for metric calculation
        train_user_indices = train_df['msno'].map(self.user_id_map).astype(int)
        train_song_indices = train_df['song_id'].map(self.song_id_map).astype(int)
        test_user_indices = test_df['msno'].map(self.user_id_map).astype(int)
        test_song_indices = test_df['song_id'].map(self.song_id_map).astype(int)

        train_target = train_df['target'].values
        test_target = test_df['target'].values

        # Initialize model
        self.model = LightFM(**self.model_params)

        # Training loop
        progress_bar = tqdm(total=num_epochs, desc="Training Progress")

        for epoch in range(num_epochs):
            self.model.fit_partial(interactions=train_coo, epochs=1, num_threads=4)

            train_preds = self.model.predict(train_user_indices.values, train_song_indices.values)
            train_prob = 1 / (1 + np.exp(-train_preds))

            test_preds = self.model.predict(test_user_indices.values, test_song_indices.values)
            test_prob = 1 / (1 + np.exp(-test_preds))

            # Calculate losses
            train_loss = log_loss(train_target, train_prob.clip(1e-15, 1 - 1e-15))
            test_loss = log_loss(test_target, test_prob.clip(1e-15, 1 - 1e-15))

            # Calculate metrics
            train_metrics = self._calculate_metrics(train_target, train_prob)
            test_metrics = self._calculate_metrics(test_target, test_prob)

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['test_loss'].append(test_loss)
            self.history['train_auc'].append(train_metrics['auc'])
            self.history['test_auc'].append(test_metrics['auc'])
            self.history['precision'].append(test_metrics['precision'])
            self.history['recall'].append(test_metrics['recall'])
            self.history['f1'].append(test_metrics['f1'])

            # Log progress
            progress_bar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Test Loss': f'{test_loss:.4f}',
                'Test AUC': f'{test_metrics["auc"]:.4f}'
            })
            progress_bar.update(1)

            logging.info(
                "Epoch %d/%d - Train Loss: %.4f, Test Loss: %.4f, Test AUC: %.4f, "
                "Precision: %.4f, Recall: %.4f, F1: %.4f",
                epoch + 1, num_epochs, train_loss, test_loss, test_metrics['auc'],
                test_metrics['precision'], test_metrics['recall'], test_metrics['f1']
            )

        progress_bar.close()
        self.is_trained = True

        # Save model and configurations
        if save_model:
            self.save_model()

        logging.info("Model training completed successfully")
        return self.history

    def predict(self, user_ids: List[str], item_ids: List[str]) -> np.ndarray:
        """
        Predict scores for user-item pairs.

        Args:
            user_ids: List of user IDs
            item_ids: List of item IDs

        Returns:
            Array of prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained or loaded before prediction")

        user_indices = [self.user_id_map[user] for user in user_ids]
        item_indices = [self.song_id_map[item] for item in item_ids]

        preds = self.model.predict(user_indices, item_indices)
        return 1 / (1 + np.exp(-preds))  # Sigmoid transformation

    def plot_metrics(self, save_plot: bool = True) -> None:
        """Plot training metrics including loss, AUC, precision, recall, and F1."""
        if not self.history['train_loss']:
            logging.warning("No training history available. Train the model first.")
            return

        num_epochs = len(self.history['train_loss'])
        epochs = range(1, num_epochs + 1)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Plot loss
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', linewidth=2)
        ax1.plot(epochs, self.history['test_loss'], label='Test Loss', linewidth=2)
        ax1.set_title('Training and Test Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Log Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot AUC
        ax2.plot(epochs, self.history['train_auc'], label='Train AUC', linewidth=2)
        ax2.plot(epochs, self.history['test_auc'], label='Test AUC', linewidth=2)
        ax2.set_title('Training and Test AUC')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('AUC')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Plot precision, recall, F1
        ax3.plot(epochs, self.history['precision'], label='Precision', linewidth=2)
        ax3.plot(epochs, self.history['recall'], label='Recall', linewidth=2)
        ax3.plot(epochs, self.history['f1'], label='F1-Score', linewidth=2)
        ax3.set_title('Precision, Recall, and F1-Score')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Plot all metrics together for comparison
        ax4.plot(epochs, self.history['test_auc'], label='AUC', linewidth=2)
        ax4.plot(epochs, self.history['precision'], label='Precision', linewidth=2)
        ax4.plot(epochs, self.history['recall'], label='Recall', linewidth=2)
        ax4.plot(epochs, self.history['f1'], label='F1-Score', linewidth=2)
        ax4.set_title('All Test Metrics')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Score')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plot_path = os.path.join(self.image_dir, "training_metrics.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info("Metrics plot saved to: %s", plot_path)

        plt.show()

    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, save_plot: bool = True) -> float:
        """
        Plot ROC curve and calculate AUC.

        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            save_plot: Whether to save the plot

        Returns:
            AUC score
        """
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
            plot_path = os.path.join(self.image_dir, "roc_curve.png")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logging.info("ROC curve plot saved to: %s", plot_path)

        plt.show()
        return roc_auc

    def save_model(self) -> None:
        """Save model, mappings, and training configuration."""
        if not self.is_trained:
            logging.warning("Model is not trained. Nothing to save.")
            return

        # Save model
        model_path = os.path.join(self.model_dir, "lightfm_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)

        # Save mappings
        mappings_path = os.path.join(self.model_dir, "mappings.pkl")
        with open(mappings_path, 'wb') as f:
            pickle.dump({
                'user_id_map': self.user_id_map,
                'song_id_map': self.song_id_map,
                'num_users': self.num_users,
                'num_items': self.num_items
            }, f)

        # Save training configuration
        config = {
            'model_params': self.model_params,
            'threshold': self.threshold,
            'training_history': self.history
        }

        config_path = os.path.join(self.model_dir, "training_configs.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logging.info("Model saved to: %s", self.model_dir)

    def load_model(self, model_dir: Optional[str] = None) -> None:
        """
        Load a pre-trained model and mappings.

        Args:
            model_dir: Directory containing saved model files
        """
        if model_dir is None:
            model_dir = self.model_dir

        # Load model
        model_path = os.path.join(model_dir, "lightfm_model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

        # Load mappings
        mappings_path = os.path.join(model_dir, "mappings.pkl")
        if not os.path.exists(mappings_path):
            raise FileNotFoundError(f"Mappings file not found: {mappings_path}")

        with open(mappings_path, 'rb') as f:
            mappings = pickle.load(f)
            self.user_id_map = mappings['user_id_map']
            self.song_id_map = mappings['song_id_map']
            self.num_users = mappings['num_users']
            self.num_items = mappings['num_items']

        # Load configuration
        config_path = os.path.join(model_dir, "training_configs.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                self.history = config.get('training_history', {})
                self.threshold = config.get('threshold', 0.5)

        self.is_trained = True
        logging.info("Model loaded successfully from: %s", model_dir)

    def hyperparameter_tuning(self, param_grid: Dict, train_df: pd.DataFrame,
                              test_df: pd.DataFrame, num_epochs: int = 10) -> Dict:
        """
        Simple hyperparameter tuning interface.

        Args:
            param_grid: Dictionary of parameters to try
            train_df: Training data
            test_df: Test data
            num_epochs: Number of epochs for each configuration

        Returns:
            Dictionary with best parameters and results
        """
        logging.info("Starting hyperparameter tuning")

        best_auc = 0
        best_params = None
        results = []

        # Simple grid search implementation
        for loss in param_grid.get('loss', ['logistic']):
            for lr in param_grid.get('learning_rate', [0.05]):
                for components in param_grid.get('no_components', [30]):

                    params = {
                        'loss': loss,
                        'learning_rate': lr,
                        'no_components': components,
                        'random_state': 42
                    }

                    logging.info("Testing parameters: %s", params)

                    # Create temporary model with current parameters
                    temp_model = LightFMRecommender(
                        model_dir=self.model_dir,
                        image_dir=self.image_dir,
                        threshold=self.threshold,
                        **params
                    )

                    # Train and evaluate
                    temp_model.train(train_df, test_df, num_epochs=num_epochs, save_model=False)
                    final_auc = temp_model.history['test_auc'][-1]

                    results.append({
                        'params': params,
                        'final_auc': final_auc
                    })

                    if final_auc > best_auc:
                        best_auc = final_auc
                        best_params = params

        logging.info("Hyperparameter tuning completed. Best AUC: %.4f", best_auc)
        logging.info("Best parameters: %s", best_params)

        return {
            'best_params': best_params,
            'best_auc': best_auc,
            'all_results': results
        }

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = "model_ckpts/lightfm")
    parser.add_argument('--image_path', type = str, default = "images/models/lightfm")
    parser.add_argument('--num_epochs', type = int, default = 10)

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]
    )

    try:
        # Load data
        logging.info("Loading training and test data...")
        train_df = pd.read_csv("data/processed_data/train_processed_data.csv")
        test_df = pd.read_csv("data/processed_data/test_processed_data.csv")
        logging.info("Data loaded successfully. Train shape: %s, Test shape: %s",
                     train_df.shape, test_df.shape)

        # Initialize recommender
        recommender = LightFMRecommender(
            model_dir="model_ckpts/lightfm",
            image_dir="images/models/lightfm",
            threshold=0.5,
            loss='logistic',
            no_components=30,
            learning_rate=0.05,
            random_state=42
        )

        # Train model
        history = recommender.train(train_df, test_df, num_epochs=10, save_model=True)

        # Plot metrics
        recommender.plot_metrics(save_plot=True)

        # Get final test predictions for ROC curve
        test_user_indices = test_df['msno'].map(recommender.user_id_map).astype(int)
        test_song_indices = test_df['song_id'].map(recommender.song_id_map).astype(int)
        test_target = test_df['target'].values

        final_test_preds = recommender.model.predict(test_user_indices.values, test_song_indices.values)
        final_test_prob = 1 / (1 + np.exp(-final_test_preds))

        # Plot ROC curve
        roc_auc = recommender.plot_roc_curve(test_target, final_test_prob, save_plot=True)

        # Print final results
        final_auc = history['test_auc'][-1]
        final_loss = history['test_loss'][-1]
        logging.info("Training completed. Final Test AUC: %.4f, Final Test Loss: %.4f, ROC AUC: %.4f",
                     final_auc, final_loss, roc_auc)

    except Exception as e:
        logging.error("Error in main execution: %s", str(e))
        raise

# How to reproduce results:
"""
To reproduce the results:

1. Ensure the data files are in the correct location:
   - data/processed_data/train_processed_data.csv
   - data/processed_data/test_processed_data.csv

2. Run the script. The model will be trained and saved to model_ckpts/lightfm/

3. To load a pre-trained model and make predictions:

   from lightfm_recommender import LightFMRecommender

   # Initialize recommender
   recommender = LightFMRecommender()

   # Load pre-trained model
   recommender.load_model("model_ckpts/lightfm")

   # Make predictions
   user_ids = ["user1", "user2", ...]
   item_ids = ["song1", "song2", ...]
   predictions = recommender.predict(user_ids, item_ids)

4. For hyperparameter tuning:

   param_grid = {
       'loss': ['logistic', 'bpr'],
       'learning_rate': [0.01, 0.05, 0.1],
       'no_components': [20, 30, 40]
   }

   results = recommender.hyperparameter_tuning(param_grid, train_df, test_df)
"""
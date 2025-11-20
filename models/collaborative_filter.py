# collaborative_filter.py
import os
import logging
from typing import Optional, Tuple, Dict, Any

import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


logger = logging.getLogger(__name__)


def get_device():
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Dataset Loading
class InteractionDataset(Dataset):
    def __init__(self, user_idxs: np.ndarray, item_idxs: np.ndarray, labels: np.ndarray):
        self.user_idxs = user_idxs.astype(np.int64)
        self.item_idxs = item_idxs.astype(np.int64)
        self.labels = labels.astype(np.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.user_idxs[idx], self.item_idxs[idx], self.labels[idx]

# Matrix Factorization Model Class
class MFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, emb_dim: int, user_bias=True, item_bias=True):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.user_bias_flag = user_bias
        self.item_bias_flag = item_bias
        if user_bias:
            self.user_b = nn.Embedding(n_users, 1)
        if item_bias:
            self.item_b = nn.Embedding(n_items, 1)

        # initialization model
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        if user_bias:
            nn.init.zeros_(self.user_b.weight)
        if item_bias:
            nn.init.zeros_(self.item_b.weight)

    def forward(self, user_idx, item_idx):
        u = self.user_emb(user_idx)
        v = self.item_emb(item_idx)
        score = (u * v).sum(dim=1, keepdim=True)  # (batch,1)
        if self.user_bias_flag:
            score = score + self.user_b(user_idx)
        if self.item_bias_flag:
            score = score + self.item_b(item_idx)
        return score.squeeze(1)  # logits


# Collaborative Filtering with Matrix Factorization Class Module
class CollaborativeFilteringMF:
    def __init__(
        self,
        train_path: str,
        test_path: str,
        image_save_dir: str = "images/models/collaborative_filter",
        model_save_dir: str = "model_ckpts/collaborative_filter",
        matrix_save_dir: str = "model_ckpts/collaborative_filter",
        emb_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-6,
        batch_size: int = 16384,
        epochs: int = 5,
        num_workers: int = 4,
        topk_sim: int = 100,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.train_path = train_path
        self.test_path = test_path
        self.image_save_dir = image_save_dir
        self.model_save_dir = model_save_dir
        self.matrix_save_dir = matrix_save_dir
        os.makedirs(self.image_save_dir, exist_ok=True)
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.matrix_save_dir, exist_ok=True)

        self.emb_dim = emb_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_workers = num_workers
        self.topk_sim = topk_sim
        self.seed = seed

        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        # prepare data placeholders set during data preparation
        self.user2idx: Dict[Any, int] = {}
        self.item2idx: Dict[Any, int] = {}
        self.idx2user: Optional[np.ndarray] = None
        self.idx2item: Optional[np.ndarray] = None
        self.model: Optional[MFModel] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion = nn.BCEWithLogitsLoss()

        # history for plotting
        self.history = {"precision": [], "recall": [], "f1": [], "auc": []}

    # Data loading & mapping
    def _prepare_data(self):
        logger.info("Loading train/test CSVs...")
        train_df = pd.read_csv(self.train_path)
        test_df = pd.read_csv(self.test_path)

        # factorize user/item to contiguous indices
        logger.info("Factorizing user and item ids to contiguous indices...")
        users_all, user_idx = pd.factorize(pd.concat([train_df["msno"], test_df["msno"]], axis=0), sort=True)
        items_all, item_idx = pd.factorize(pd.concat([train_df["song_id"], test_df["song_id"]], axis=0), sort=True)

        users = pd.concat([train_df["msno"], test_df["msno"]], axis=0).unique()
        items = pd.concat([train_df["song_id"], test_df["song_id"]], axis=0).unique()
        self.idx2user = users
        self.idx2item = items
        self.user2idx = {u: i for i, u in enumerate(users)}
        self.item2idx = {i_: j for j, i_ in enumerate(items)}

        logger.info(f"Num users: {len(self.idx2user)}, Num items: {len(self.idx2item)}")

        def map_df(df):
            u_idx = df["msno"].map(self.user2idx).to_numpy(dtype=np.int64)
            it_idx = df["song_id"].map(self.item2idx).to_numpy(dtype=np.int64)
            labels = df["target"].to_numpy(dtype=np.float32)
            return u_idx, it_idx, labels

        train_u, train_i, train_y = map_df(train_df)
        test_u, test_i, test_y = map_df(test_df)

        # free memory
        del train_df, test_df

        # create dataset and data loader
        train_ds = InteractionDataset(train_u, train_i, train_y)
        test_ds = InteractionDataset(test_u, test_i, test_y)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)

        return train_loader, test_loader, len(self.idx2user), len(self.idx2item)

    # Train step
    def fit(self):
        train_loader, test_loader, n_users, n_items = self._prepare_data()

        logger.info("Building model...")
        self.model = MFModel(n_users, n_items, self.emb_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        logger.info(f"Start training for {self.epochs} epochs. Batch size: {self.batch_size}")
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            running_loss = 0.0
            n_batches = 0
            for batch_idx, (u, i, y) in enumerate(train_loader):
                u = u.to(self.device)
                i = i.to(self.device)
                y = y.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(u, i)
                loss = self.criterion(logits, y)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                n_batches += 1

            avg_loss = running_loss / max(1, n_batches)
            logger.info(f"Epoch {epoch}/{self.epochs} - train_loss: {avg_loss:.6f}")

            # evaluation on test set
            precision, recall, f1, auc = self.evaluate(test_loader)
            self.history["precision"].append(precision)
            self.history["recall"].append(recall)
            self.history["f1"].append(f1)
            self.history["auc"].append(auc)

            logger.info(f"Epoch {epoch} metrics - precision: {precision:.6f}, recall: {recall:.6f}, f1: {f1:.6f}, auc: {auc:.6f}")

            # save checkpoint for this epoch
            ckpt_path = os.path.join(self.model_save_dir, f"mf_epoch{epoch}.pt")
            self.save_model(ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")

        # asave final model and user-item similarity matrix
        final_ckpt = os.path.join(self.model_save_dir, "mf_final.pt")
        self.save_model(final_ckpt)
        logger.info(f"Saved final model: {final_ckpt}")

        # create and save user-item similarity sparse matrix (top-k per user)
        logger.info("Computing and saving user-item similarity matrix (top-k per user)...")
        self.save_user_item_matrix(topk=self.topk_sim)
        logger.info("All done.")

    # Evaluation step
    def evaluate(self, loader: DataLoader) -> Tuple[float, float, float, float]:
        self.model.eval()
        preds = []
        labels = []
        with torch.inference_mode():
            for u, i, y in loader:
                u = u.to(self.device)
                i = i.to(self.device)
                logits = self.model(u, i)
                prob = torch.sigmoid(logits).detach().cpu().numpy()
                preds.append(prob)
                labels.append(y.numpy())
        preds = np.concatenate(preds, axis=0)
        labels = np.concatenate(labels, axis=0).astype(int)

        # threshold 0.5 for precision/recall/f1
        pred_bin = (preds >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, pred_bin, average="binary", zero_division=0)
        try:
            auc = float(roc_auc_score(labels, preds))
        except Exception:
            auc = 0.0

        return precision, recall, f1, auc

    # Saving / Loading model func
    def save_model(self, path: str):
        state = {
            "model_state": self.model.state_dict(),
            "user2idx": self.user2idx,
            "item2idx": self.item2idx,
            "idx2user": self.idx2user,
            "idx2item": self.idx2item,
            "emb_dim": self.emb_dim,
            "seed": self.seed,
        }
        torch.save(state, path)

    def load_model(self, path: str, map_location: Optional[torch.device] = None):
        map_location = map_location or self.device
        state = torch.load(path, map_location=map_location)
        self.user2idx = state["user2idx"]
        self.item2idx = state["item2idx"]
        self.idx2user = state["idx2user"]
        self.idx2item = state["idx2item"]
        self.emb_dim = state.get("emb_dim", self.emb_dim)
        n_users = len(self.idx2user)
        n_items = len(self.idx2item)
        self.model = MFModel(n_users, n_items, self.emb_dim).to(self.device)
        self.model.load_state_dict(state["model_state"])
        logger.info(f"Loaded model from {path}")

    # Save user-item similarity matrix
    def save_user_item_matrix(self, topk: int = 100):
        """
        Compute user-item scores via dot(user_emb, item_emb) and save a sparse matrix keeping top-k per user.
        To save memory, compute in user-chunks.
        """
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")

        self.model.eval()
        user_emb = self.model.user_emb.weight.detach().cpu().numpy()
        item_emb = self.model.item_emb.weight.detach().cpu().numpy()

        n_users = user_emb.shape[0]
        n_items = item_emb.shape[0]
        logger.info(f"Creating top-{topk} sparse similarity matrix for {n_users} users and {n_items} items")

        rows = []
        cols = []
        data = []

        chunk_size = 2048  
        for start in range(0, n_users, chunk_size):
            end = min(n_users, start + chunk_size)
            u_chunk = user_emb[start:end] 
            # compute scores chunk x items -> matrix multiplication
            scores = u_chunk.dot(item_emb.T)  # shape (chunk, n_items)

            # for each user in chunk, select topk indices
            for idx_in_chunk, sc in enumerate(scores):
                if topk >= n_items:
                    topk_idx = np.argpartition(-sc, n_items - 1)[:n_items]
                else:
                    topk_idx = np.argpartition(-sc, topk)[:topk]
                    # sort those topk descending
                    topk_idx = topk_idx[np.argsort(-sc[topk_idx])]
                user_idx = start + idx_in_chunk
                top_scores = sc[topk_idx]
                rows.extend([user_idx] * len(topk_idx))
                cols.extend(topk_idx.tolist())
                data.extend(top_scores.tolist())

            logger.info(f"Processed users {start}..{end-1}")

        # build CSR sparse matrix (users x items)
        sim_csr = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        save_path = os.path.join(self.matrix_save_dir, "user_item_sim_topk.npz")
        sp.save_npz(save_path, sim_csr)
        logger.info(f"Saved user-item similarity sparse matrix to {save_path}")

        # save embeddings for downstream retrieval
        emb_path = os.path.join(self.matrix_save_dir, "embeddings.npz")
        np.savez_compressed(emb_path, user_emb=user_emb, item_emb=item_emb)
        logger.info(f"Saved embeddings to {emb_path}")

        np.save(os.path.join(self.matrix_save_dir, "user_factors.npy"), user_emb)
        np.save(os.path.join(self.matrix_save_dir, "item_factors.npy"), item_emb)
        logger.info("Saved user_factors.npy and item_factors.npy")


    # Plot metrics
    def plot_metrics(self):
        epochs = np.arange(1, len(self.history["precision"]) + 1)
        plt.figure()
        plt.plot(epochs, self.history["precision"], label="Precision")
        plt.plot(epochs, self.history["recall"], label="Recall")
        plt.plot(epochs, self.history["f1"], label="F1")
        plt.plot(epochs, self.history["auc"], label="AUC")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.title("Metrics per Epoch")
        plt.legend()
        file_path = os.path.join(self.image_save_dir, "metrics_epoch.png")
        plt.savefig(file_path, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved metric plot to {file_path}")

    # Utility: predict for given user-item pairs in batches
    def predict_pairs(self, user_array: np.ndarray, item_array: np.ndarray, batch_size: int = 65536):
        if self.model is None:
            raise RuntimeError("Model not trained or loaded.")
        self.model.eval()
        preds = []
        with torch.inference_mode():
            n = len(user_array)
            for start in range(0, n, batch_size):
                end = min(n, start + batch_size)
                u = torch.tensor(user_array[start:end], dtype=torch.long, device=self.device)
                it = torch.tensor(item_array[start:end], dtype=torch.long, device=self.device)
                logits = self.model(u, it)
                preds.append(torch.sigmoid(logits).detach().cpu().numpy())
        return np.concatenate(preds, axis=0)

# If run as script: example usage and reproduce instructions
if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    # Paths as per your request
    train_csv = "data/processed_data/train_processed_data_mf.csv"
    test_csv = "data/processed_data/test_processed_data_mf.csv"
    image_dir = "images/models/collaborative_filter"
    model_dir = "model_ckpts/collaborative_filter"
    matrix_dir = "model_ckpts/collaborative_filter"

    cf = CollaborativeFilteringMF(
        train_path = train_csv,
        test_path = test_csv,
        image_save_dir = image_dir,
        model_save_dir = model_dir,
        matrix_save_dir = matrix_dir,
        emb_dim = 64,
        lr = 1e-3,
        batch_size = 16384,
        epochs = 5, 
        num_workers = 4,
        topk_sim = 100,
    )

    # Run training once
    cf.fit()

    # plot final metrics
    cf.plot_metrics()

    logger.info("Training run finished. See saved artifacts in the specified directories.")

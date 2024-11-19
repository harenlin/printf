import os
import random
import math
import torch
import numpy as np
import scipy.sparse as sp
from tqdm import tqdm
from typing import Tuple, Union
from functools import partial
from torch.multiprocessing import Pool


def handle_reproducibility(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True


def split_edges(
    edge_list: Union[np.ndarray, torch.Tensor],
    val_ratio: float,
    test_ratio: float,
) -> tuple:
    """Split edges into training, validation, and testing sets.
    """
    perm = torch.randperm(len(edge_list))
    edge_list = edge_list[perm]

    n_val = math.floor(val_ratio * len(edge_list))
    n_test = math.floor(test_ratio * len(edge_list))

    val_edge_list = edge_list[:n_val]
    test_edge_list = edge_list[n_val:n_val + n_test]
    train_edge_list = edge_list[n_val + n_test:]

    return train_edge_list, val_edge_list, test_edge_list


def normalize_matrix(A: sp.csr_matrix) -> torch.Tensor:
    row_sum = np.asarray(A.sum(axis=1))
    D_inv = np.power(row_sum, -0.5)
    D_inv[np.isinf(D_inv)] = 0.0
    D_inv = sp.diags(D_inv.flatten())
    
    A_norm = D_inv.dot(A)
    A_norm = A_norm.dot(D_inv).tocoo()

    i = torch.from_numpy(np.stack([A_norm.row, A_norm.col]))
    v = torch.from_numpy(A_norm.data)
    A_norm = torch.sparse_coo_tensor(i, v, A_norm.shape, dtype=torch.float)
    return A_norm


def get_A_norm(edge_list: np.ndarray, n_user: int, n_item: int) -> Tuple[torch.Tensor, torch.Tensor]:
    user, item = edge_list.T
    item = item + n_user
    row = np.concatenate([user, item])
    col = np.concatenate([item, user])
    A = sp.csr_matrix(
        (np.ones(len(row)), (row, col)),
        shape=(n_user + n_item, n_user + n_item),
    )
    
    A_norm = normalize_matrix(A)

    A_hat = A + sp.eye(A.shape[0])
    A_hat_norm = normalize_matrix(A_hat)

    return A_norm, A_hat_norm


def get_user2pos_item(edge_list: np.ndarray, n_user: int) -> list:
    user2pos_item = [[] for _ in range(n_user)]
    for user, item in edge_list:
        user, item = user.item(), item.item()
        user2pos_item[user].append(item)
    for user, pos_item_list in enumerate(user2pos_item):
        user2pos_item[user] = torch.tensor(pos_item_list, dtype=torch.long)
    return user2pos_item


def _sample_neg_items_for_single_user(n_item: int, n_neg_sample: int, seed: int, pos_item_list: list) -> np.ndarray:
    neg_item_list = np.delete(np.arange(n_item), pos_item_list)
    rng = np.random.default_rng(seed)
    sampled_neg_items = rng.choice(
        neg_item_list,
        min(n_neg_sample, len(neg_item_list)),
        replace=False,
    )
    return sampled_neg_items


def sample_neg_items(eval_user_list: list, pos_item_list: list, n_item: int, n_neg_sample: int, seed: int) -> torch.Tensor:
    """`user2pos_item` should contain all positive items to exclude from negative sampling.
    """
    print("Sampling evaluation data...")
    pos_item_list = [pos_item.numpy() for pos_item in pos_item_list]
    with Pool(32) as p:
        neg_item_list = list(tqdm(
            p.imap_unordered(
                partial(
                    _sample_neg_items_for_single_user,
                    n_item,
                    n_neg_sample,
                    seed,
                ),
                pos_item_list, 
                chunksize=10000,
            ),
            total=len(eval_user_list),
        ))
    neg_item_list = torch.from_numpy(np.stack(neg_item_list))
    return neg_item_list


class Metrics():
    def __init__(self, topk, device):
        self.topk = topk
        self.device = device
        self.d_list = torch.log2(
            torch.tensor(range(2, 10000 + 2), dtype=torch.float, device=device)
        )
        self.ideal_g_list = torch.ones(10000, device=self.device)

    @staticmethod
    def recall(tp_list, n_pos):
        return torch.sum(tp_list) / n_pos

    def ndcg(self, g_list, n_pos):
        dcg = torch.sum(g_list / self.d_list[:self.topk])
        n_ideal_rel = min(n_pos, self.topk)
        idcg = torch.sum(self.ideal_g_list[:n_ideal_rel] / self.d_list[:n_ideal_rel])
        return dcg / idcg


def eval_recsys(
    user_list: list,
    user2pos_item: list,
    user2exclude_item: list,
    user_emb: torch.Tensor,
    item_emb: torch.Tensor,
    metrics: Metrics,
) -> Tuple[float, float]:
    sum_recall = 0
    sum_ndcg = 0
    for _, user in enumerate(tqdm(user_list)):
        pos_items = user2pos_item[user]
        n_pos_item = len(pos_items)

        dot_prod = torch.sum(torch.mul(user_emb[user], item_emb), dim=1)
        dot_prod[user2exclude_item[user]] = -float("inf")
        _, retrieved_items = torch.topk(dot_prod, k=metrics.topk)

        tp_list = torch.isin(retrieved_items, pos_items.to(retrieved_items.device), assume_unique=True)
        
        sum_recall += metrics.recall(tp_list, n_pos_item)
        sum_ndcg += metrics.ndcg(tp_list, n_pos_item)
    return sum_recall, sum_ndcg


if __name__ == "__main__":
    print(sample_neg_items(
        [0, 1],
        [
            [0, 3],
            [0, 1, 1, 2, 4],
        ],
        n_user=2,
        n_item=5,
        n_neg_sample=2,
    ))

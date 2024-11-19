import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

from utils import get_user2pos_item


class AmazonReview(Dataset):
    def __init__(self, data_dir, mode, n_user_eval=None):
        self.data_dir = Path(data_dir)
        self.mode = mode

        if (
            not (self.data_dir / "A.pt").exists()
            # or not (self.data_dir / "item_emb_dimrd.pt").exists()
        ):
            print("Preprocess data first.")
            exit()
        self.n_user = np.loadtxt(self.data_dir / "n_user.txt", delimiter=",", dtype=np.int64).item()
        self.n_item = np.loadtxt(self.data_dir / "n_item.txt", delimiter=",", dtype=np.int64).item()
        data_A = torch.load(self.data_dir / "A.pt")
        self.A = data_A["A"]
        self.A_hat = data_A["A_hat"]

        self.train_list = np.loadtxt(self.data_dir / "train_list.csv", delimiter=",", dtype=np.int64)
        self.val_list = np.loadtxt(self.data_dir / "val_list.csv", delimiter=",", dtype=np.int64)
        self.train_user2pos_item = get_user2pos_item(self.train_list, self.n_user)
        self.val_user2pos_item = get_user2pos_item(self.val_list, self.n_user)

        if mode == "train":
            self.val_user_list = []
            for user in range(self.n_user):
                if len(self.val_user2pos_item[user]):
                    self.val_user_list.append(user)
            if n_user_eval and n_user_eval < len(self.val_user_list):
                self.val_user_list = torch.tensor(self.val_user_list)[
                    torch.randperm(len(self.val_user_list))[:n_user_eval]
                ]
        elif mode == "test":
            self.test_list = np.loadtxt(self.data_dir / "test_list.csv", delimiter=",", dtype=np.int64)
            self.test_user2pos_item = get_user2pos_item(self.test_list, self.n_user)

            self.test_user_list = []
            for user in range(self.n_user):
                if len(self.test_user2pos_item[user]):
                    self.test_user_list.append(user)
            self.train_val_user2pos_item = [None] * self.n_user
            for user in self.test_user_list:
                self.train_val_user2pos_item[user] = torch.cat([
                    self.train_user2pos_item[user],
                    self.val_user2pos_item[user],
                ])
    
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        assert self.mode == "train"
        user, pos_item = self.train_list[idx]
        pos_item_list = self.train_user2pos_item[user]

        # Oversample for effeciency, so only very few have to sample more than once
        while True:
            n_neg_sample = 10
            cand = torch.randint(self.n_item, size=(n_neg_sample,))
            mask = torch.isin(cand, pos_item_list)
            if len(cand[~mask]):
                neg_item = cand[~mask][0]
                break

        return user, pos_item, neg_item

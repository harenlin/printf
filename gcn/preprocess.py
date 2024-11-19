import json
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm
from typing import Tuple
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

from utils import handle_reproducibility, split_edges, get_A_norm
            

class KCoreGraph():
    def __init__(self, edge_list, n_node):
        self.graph = [[] for _ in range(n_node)]
        for u, v in edge_list:
            self.graph[u].append(v)
            self.graph[v].append(u)
 
    def dfs(self, u, visited, removed, degree, k, progress):
        stack = [u]
        while len(stack):
            u = stack.pop()
            if not removed[u] and degree[u] < k:
                removed[u] = True
                if not visited[u]:
                    visited[u] = True
                    progress.update(1)
                for v in self.graph[u]:
                    degree[v] -= 1
                for v in self.graph[u]:
                    if not removed[v]:
                        stack.append(v)
            elif not visited[u]:
                visited[u] = True
                progress.update(1)
                for v in self.graph[u]:
                    if not visited[v] and not removed[v]:
                        stack.append(v)
 
    def get_k_core_edge_list(self, k):
        visited = np.zeros(len(self.graph), dtype=np.bool8)
        removed = np.zeros(len(self.graph), dtype=np.bool8)
        degree = np.zeros(len(self.graph), dtype=np.int64)
        for u, v_list in enumerate(self.graph):
            degree[u] = len(v_list)
 
        progress = tqdm(total=len(self.graph))
        for u in range(len(self.graph)):
            if not visited[u] and not removed[u]:
                self.dfs(u, visited, removed, degree, k, progress)
        progress.close()
 
        edge_list = []
        for u, v_list in enumerate(self.graph):
            if not removed[u]:
                for v in v_list:
                    if u < v and not removed[v]:  # Only user->item egdes (u < v)
                        edge_list.append([u, v])
        return np.array(edge_list)


class Text(Dataset):
    def __init__(self, text_list, tokenizer):
        self.text_list = text_list
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.text_list)
    
    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.text_list[idx],
            padding="max_length",
            truncation=True,
            max_length=384,
            return_tensors="pt",
        )
        return tok.input_ids.view(-1)
        # return (
        #     tok.input_ids.view(-1),
        #     tok.token_type_ids.view(-1),
        #     tok.attention_mask.view(-1),
        # )


class AmazonReview():
    def __init__(self, data_dir, device, k_core):
        """Preprocess data into graph and get item descriptions.
        The graph will be saved as edge list containing only undirected (user, item) pairs.
        User: [0, n_user)
        Item: [0, n_item)
        """
        data_dir = Path(data_dir)

        start_time = time()
        self.df = self.get_df(data_dir / "5-core.json")
        # self.df = self.get_df(data_dir / "review.json")
        print(f"Load DF time: {round(time() - start_time, 2)} sec")

        if k_core > 5:
            start_time = time()
            self.df = self.filter_df_k_core(self.df, k=k_core)
            print(f"Filter DF time: {round(time() - start_time, 2)} sec")

        start_time = time()
        user_set = set(self.df["user_id"])
        item_set = set(self.df["item_id"])
        self.user_list = sorted(user_set)
        self.item_list = sorted(item_set)
        item_text_list, item_id2emb_idx = self.get_item_text(data_dir / "metadata.json", item_set)
        self.emb_list, self.n_item_w_text = self.get_text_emb(item_text_list, self.item_list, item_id2emb_idx, device)
        print(f"Process description emb time: {round(time() - start_time, 2)} sec")
        print(f"Done.")

    @staticmethod
    def get_df(path: Path) -> pd.DataFrame:
        df = pd.read_json(path, orient="records", lines=True)[["reviewerID", "asin"]]
        df = df[df["reviewerID"].astype(bool) & df["asin"].astype(bool)]
        df = df.drop_duplicates()
        df = df.reset_index(drop=True)
        df = df.rename(columns={"reviewerID": "user_id", "asin": "item_id"})
        return df

    @staticmethod
    def filter_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.groupby("user_id").filter(lambda x: len(x) >= 3)
        df = df.reset_index(drop=True)
        return df

    @staticmethod
    def filter_df_k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
        user_list = df["user_id"].unique()
        item_list = df["item_id"].unique()
        user_mapping = {user: id for id, user in enumerate(user_list)}
        item_mapping = {item: id + len(user_list) for id, item in enumerate(item_list)}
        df["user_id"] = df["user_id"].map(user_mapping)
        df["item_id"] = df["item_id"].map(item_mapping)

        edge_list = df[["user_id", "item_id"]].to_numpy()

        print(f"Processing {k}-core...")
        graph = KCoreGraph(edge_list, len(user_list) + len(item_list))
        edge_list = graph.get_k_core_edge_list(k)
        print(len(edge_list))
        del graph

        df = pd.DataFrame({"user_id": edge_list[:, 0], "item_id": edge_list[:, 1]})
        df["user_id"] = df["user_id"].map(lambda i: user_list[i])
        df["item_id"] = df["item_id"].map(lambda i: item_list[i - len(user_list)])

        return df[["user_id", "item_id"]]

    @staticmethod
    def get_item_text(path: Path, item_set: set) -> Tuple[list, dict]:
        item_text_list = []
        item_id2emb_idx = {}
        with open(path) as file:
            for line in file:
                single_item = json.loads(line)
                # print(json.dumps(single_item, sort_keys=True, indent=4))

                item_id = single_item["asin"]
                if item_id in item_set:
                    item_desc = ""
                    if single_item.get("title"):
                        item_desc += single_item["title"] + " "
                    if single_item.get("description") and single_item["description"][0] != "":
                        item_desc += single_item["description"][0]
                    if item_desc != "":
                        if item_id not in item_id2emb_idx:
                            item_id2emb_idx[item_id] = len(item_text_list)
                            item_text_list.append(item_desc)

        return item_text_list, item_id2emb_idx

    @staticmethod
    @torch.no_grad()
    def get_text_emb(text_list: list, item_list: list, item_id2emb_idx: dict, device: str, user=False) -> Tuple[torch.Tensor, int]:
        # tokenizer = AutoTokenizer.from_pretrained("google/bert_uncased_L-12_H-128_A-2")
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        dataset = Text(text_list, tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=256,
            num_workers=16,
        )
        # model = AutoModel.from_pretrained("google/bert_uncased_L-12_H-128_A-2")
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        model.to(device)
        model.eval()

        emb_list = []
        # for input_ids, token_type_ids, attention_mask in tqdm(data_loader):
        for input_ids in tqdm(data_loader):
            input_ids = input_ids.to(device)
            # token_type_ids = token_type_ids.to(device)
            # attention_mask = attention_mask.to(device)
            emb_list.append(
                model(
                    **{
                        "input_ids": input_ids,
                        # "token_type_ids": token_type_ids,
                        # "attention_mask": attention_mask,
                    }
                ).last_hidden_state[:, 0].detach().cpu()
                # ).last_hidden_state.detach().cpu()
            )
        emb_list = torch.cat(emb_list)
        n_item_w_text = len(emb_list)

        emb_list_all = []
        for item_id in item_list:
            if item_id in item_id2emb_idx:
                emb_list_all.append(emb_list[item_id2emb_idx[item_id]])
            else:  # item has no text
                emb_list_all.append(torch.zeros(len(emb_list[0])))
                # emb_list_all.append(torch.randn(len(emb_list[0])))
        emb_list_all = torch.stack(emb_list_all)

        return emb_list_all, n_item_w_text

    def save_data(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "info.txt", "w") as file:
            log_list = [
                f"Users: {len(self.user_list)}",
                f"Items: {len(self.item_list)}",
                f"User-item edges: {len(self.df)}",
                f"Items with textual content: {self.n_item_w_text}",
            ]
            file.writelines("\n".join(log_list))
        
        np.savetxt(save_dir / "n_user.txt", [len(self.user_list)], delimiter=",", fmt="%s")
        np.savetxt(save_dir / "n_item.txt", [len(self.item_list)], delimiter=",", fmt="%s")
        torch.save(
            {
                "df": self.df,
                "user_list": self.user_list,
                "item_list": self.item_list,
            },
            save_dir / "data.pt",
        )
        torch.save(self.emb_list, save_dir / "item_emb.pt")


class MovieLens():
    def __init__(self, data_dir, device, k_core):
        """Preprocess data into graph and get movie overviews.
        The graph will be saved as edge list containing only undirected (user, item) pairs.
        User: [0, n_user)
        Item: [0, n_item)
        """
        data_dir = Path(data_dir)

        start_time = time()
        self.df = self.get_df(data_dir / "ratings.dat")
        # self.df = self.get_df(data_dir / "ratings.csv")
        print(f"Load DF time: {round(time() - start_time, 2)} sec")

        start_time = time()
        self.df = AmazonReview.filter_df_k_core(self.df, k=k_core)
        print(f"Filter DF time: {round(time() - start_time, 2)} sec")

        start_time = time()
        user_set = set(self.df["user_id"])
        item_set = set(self.df["item_id"])
        self.user_list = sorted(user_set)
        self.item_list = sorted(item_set)
        item_text_list, item_id2emb_idx = self.get_item_text(data_dir / "movies.csv", data_dir / "overview.csv", item_set)
        self.emb_list, self.n_item_w_text = AmazonReview.get_text_emb(item_text_list, self.item_list, item_id2emb_idx, device)
        print(f"Process description emb time: {round(time() - start_time, 2)} sec")
        print(f"Done.")

    @staticmethod
    def get_df(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, sep="::", header=None, names=["user_id", "item_id"], usecols=[0, 1])
        # df = pd.read_csv(path, usecols=["userId", "movieId"])
        # df = df.rename(columns={"userId": "user_id", "movieId": "item_id"})
        return df

    @staticmethod
    def get_item_text(title_path: Path, overview_path: Path, item_set: set) -> Tuple[list, dict]:
        df = pd.read_csv(title_path)
        df = df.set_index("movieId").join(pd.read_csv(overview_path).set_index("movieId"))
        item_text_list = []
        item_id2emb_idx = {}
        for item_id, row in df.iterrows():
            if item_id in item_set:
                item_text = row["title"] + " "
                if type(row["overview"]) == str:  # not nan
                    item_text += row["overview"]
                # print(item_text)
                if item_text != "":
                    item_id2emb_idx[item_id] = len(item_text_list)
                    item_text_list.append(item_text)

        return item_text_list, item_id2emb_idx

    def save_data(self, save_dir: str) -> None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "info.txt", "w") as file:
            log_list = [
                f"Users: {len(self.user_list)}",
                f"Items: {len(self.item_list)}",
                f"User-item edges: {len(self.df)}",
                f"Items with textual content: {self.n_item_w_text}",
            ]
            file.writelines("\n".join(log_list))
        
        np.savetxt(save_dir / "n_user.txt", [len(self.user_list)], delimiter=",", fmt="%s")
        np.savetxt(save_dir / "n_item.txt", [len(self.item_list)], delimiter=",", fmt="%s")
        torch.save(
            {
                "df": self.df,
                "user_list": self.user_list,
                "item_list": self.item_list,
            },
            save_dir / "data.pt",
        )
        torch.save(self.emb_list, save_dir / "item_emb.pt")


def process_single_domain(data_dir, save_dir):
    """Reindex, split into train/val/text sets, and compute adjacency matrix.
    """
    data_dir = Path(data_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(data_dir / "data.pt")
    df = data["df"]
    user_list = data["user_list"]
    item_list = data["item_list"]

    user_mapping = {user_id: i for i, user_id in enumerate(user_list)}
    item_mapping = {item_id: i for i, item_id in enumerate(item_list)}
    df["user_id"] = df["user_id"].map(user_mapping)
    df["item_id"] = df["item_id"].map(item_mapping)

    edge_list = df[["user_id", "item_id"]].to_numpy()
    train_list, val_list, test_list = split_edges(edge_list, val_ratio=0.05, test_ratio=0.2)

    train_A, train_A_hat = get_A_norm(train_list, len(user_list), len(item_list))

    np.savetxt(save_dir / "train_list.csv", train_list, delimiter=",", fmt="%s")
    np.savetxt(save_dir / "val_list.csv", val_list, delimiter=",", fmt="%s")
    np.savetxt(save_dir / "test_list.csv", test_list, delimiter=",", fmt="%s")
    torch.save(
        {
            "A": train_A,
            "A_hat": train_A_hat,
        },
        save_dir / "A.pt",
    )


if __name__ == "__main__":
    """
    handle_reproducibility(seed=0)
    data_dir = "data/Grocery/"
    AmazonReview(
        data_dir,
        device="cuda:0",
        k_core=10,
    ).save_data(data_dir)
    handle_reproducibility(seed=0)
    process_single_domain(data_dir, save_dir=data_dir)
    """
    
    handle_reproducibility(seed=0)
    data_dir = "data/CDs/"
    AmazonReview(
        data_dir,
        device="cuda:0",
        k_core=12,
    ).save_data(data_dir)
    handle_reproducibility(seed=0)
    process_single_domain(data_dir, save_dir=data_dir)

    """
    handle_reproducibility(seed=0)
    data_dir = "data/MovieLens/"
    MovieLens(
        data_dir,
        device="cuda:0",
        k_core=10,
    ).save_data(data_dir)
    handle_reproducibility(seed=0)
    process_single_domain(data_dir, save_dir=data_dir)
    """

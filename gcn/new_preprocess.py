import csv
import json
import torch
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from time import time
from tqdm import tqdm
from typing import Tuple
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader, random_split 

from utils import handle_reproducibility, split_edges, get_A_norm
            

class KCoreGraph():
    def __init__(self, edge_list, n_node):
        self.graph = [[] for _ in range(n_node)]
        for u, v in edge_list:
            self.graph[u].append(v)
            self.graph[v].append(u)
 
    def dfs(self, u, visited, removed, degree, k, progress):
        """ Method called by get_k_core_edge_list in KCoreGraph """
        # (1) Maintain a stack for DFS check
        stack = [u]
        # (2) Start DFS traversal
        while len(stack):
            u = stack.pop()
            # (2-a) If current node is not removed, 
            #       but the degree is less than k 
            # -> It should be removed.
            if not removed[u] and degree[u] < k:
                removed[u] = True
                # (2-a) Mark as removed and visited
                if not visited[u]: 
                    visited[u] = True
                    progress.update(1)
                # (2-a) Also, update the degree of its neighbors
                for v in self.graph[u]:
                    degree[v] -= 1
                # (2-a) Last, check its neighbors, 
                # if not removed, it should be re-checked again 
                # -> Push into stack.
                for v in self.graph[u]:
                    if not removed[v]:
                        stack.append(v)
            # (2-b) If current node should not be removed 
            #       and has valid degree
            elif not visited[u]:
                # (2-b) Mark as visited first
                visited[u] = True
                progress.update(1)
                # (2-b) Moreover, check its neighbors, 
                # if it is not visited and not removed 
                # -> Push into stack for re-check.
                for v in self.graph[u]:
                    if not visited[v] and not removed[v]:
                        stack.append(v)
 

    def get_k_core_edge_list(self, k):
        visited = np.zeros(len(self.graph), dtype=np.bool8)
        removed = np.zeros(len(self.graph), dtype=np.bool8)
        degree = np.array([len(v_list) for u, v_list in enumerate(self.graph)])
 
        progress = tqdm(total=len(self.graph))
        for u in range(len(self.graph)):
            if not visited[u] and not removed[u]:
                self.dfs(u, visited, removed, degree, k, progress)
        progress.close()
 
        edge_list = []
        for u, v_list in enumerate(self.graph):
            if not removed[u]:
                for v in v_list:
                    # Only user->item egdes (u < v)
                    if u < v and not removed[v]:  
                        edge_list.append([u, v])
        return np.array(edge_list)



class Text(Dataset):
    def __init__(self, text_list, tokenizer, max_length=384):
        self.text_list = text_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.text_list)
    
    def __getitem__(self, idx):
        tok = self.tokenizer(
            self.text_list[idx], 
            padding="max_length",
            truncation=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        return tok.input_ids.view(-1)



class AmazonReview():
    def __init__(self, data_dir, device, k_core):
        """ 
           1) Preprocess data into graph structure.
           2) Get item descriptions and images.
           3) Get user reviews.
           4) The graph will be saved as edge list,
              containing only undirected (user, item) pairs.

            - User: [0, n_user)
            - Item: [0, n_item)
        """
        data_dir = Path(data_dir)

        start_time = time()
        self.df = self.get_df(data_dir / "5-core.json")
        print(f"Load DF time: {round(time() - start_time, 2)} sec")

        """
        if k_core > 5:
            start_time = time()
            self.df = self.filter_df_k_core(self.df, k=k_core)
            print(f"Filter DF time: {round(time() - start_time, 2)} sec")
        """

        if k_core > 5:
            start_time = time()
            prev_df = self.df.copy(deep=True)
            # Extract K-core edge_lists
            new_df = self.filter_df_k_core(self.df, k=k_core)
            # Merge original dataframe and edge_lists via user_id & item_id
            self.df = pd.merge(prev_df, new_df, how='inner', on=["user_id", "item_id"])
            # Remember to drop duplicate rows and reset index in the dataframe 
            self.df = self.df.drop_duplicates()
            self.df = self.df.reset_index(drop=True)
            # Last but not least, clean the review data first
            self.df["review"] = self.df["review"].apply(
                lambda review : " ".join(
                    str(review).strip().strip("\"").strip("\n").split()
                )
            )
            print(f"Filter DF time: {round(time() - start_time, 2)} sec")
        

        start_time = time()

        user_set = set(self.df["user_id"])
        item_set = set(self.df["item_id"])
        self.user_list = sorted(user_set)
        self.item_list = sorted(item_set)

        """ 
            Now, after confirming the edges and reviews,
            include "metadata.json" for the information 
            of items' descriptions, titles, and images.
        """
        
        # Generate items' text and image 
        # -> for the utilization of ALBEF img-text alignment.
        self.generate_Item_TextAndImage(Path(data_dir / "metadata.json"), item_set)

        # Generate items' title + description[0] as their text feature
        # -> for the utilization of item embedding initialization. 
        item_text_list, item_id2emb_idx = \
            self.get_item_text(data_dir / "metadata.json", item_set)

        
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        with open(data_dir / "item_list.pkl", "wb") as fp: 
            pickle.dump(self.item_list, fp)
        with open(data_dir / "item_id2emb_idx.pkl", "wb") as fp: 
            pickle.dump(item_id2emb_idx, fp)
        
        
        self.emb_list, self.n_item_w_text = self.get_text_emb(
            item_text_list, self.item_list, item_id2emb_idx, device)

        print(f"Process description emb time: {round(time() - start_time, 2)} sec")
        print(f"Done.")



    @staticmethod
    def get_df(path: Path) -> pd.DataFrame:
        df = pd.read_json(path, orient="records", lines=True)
        df = df[["reviewerID", "asin", "reviewText", "overall"]]
        df = df[df["reviewerID"].astype(bool) & \
                df["asin"].astype(bool) &       \
                df["reviewText"].astype(bool) & \
                df["overall"].astype(bool)]
        df = df.drop_duplicates(subset=['reviewerID', 'asin'], keep='last') 
        df = df.reset_index(drop=True)
        df = df.rename(columns={"reviewerID": "user_id", 
                                "asin": "item_id",
                                "reviewText": "review",
                                "overall": "rating"})
        return df



    @staticmethod
    def filter_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.groupby("user_id").filter(lambda x: len(x) >= 3)
        df = df.reset_index(drop=True)
        return df



    @staticmethod
    def filter_df_k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
        user_list, item_list = df["user_id"].unique(), df["item_id"].unique()
        user_mapping = {user: id for id, user in enumerate(user_list)}
        item_mapping = {item: id + len(user_list) for id, item in enumerate(item_list)} 
        df["user_id"], df["item_id"] = df["user_id"].map(user_mapping), df["item_id"].map(item_mapping)
        
        edge_list = df[["user_id", "item_id"]].to_numpy()
        print(f"Current edge_list: ", edge_list)

        print(f"Processing {k}-core...")
        graph = KCoreGraph(edge_list, len(user_list) + len(item_list))
        edge_list = graph.get_k_core_edge_list(k)
        print(f"Length of edge_list: ", len(edge_list))
        del graph

        df = pd.DataFrame({"user_id": edge_list[:, 0], "item_id": edge_list[:, 1]})
        df["user_id"] = df["user_id"].map(lambda i: user_list[i])
        df["item_id"] = df["item_id"].map(lambda i: item_list[i - len(user_list)])

        return df[["user_id", "item_id"]]



    @staticmethod
    def generate_Item_TextAndImage(path: Path, item_set: set):
        """ 
            The given item_set is the only valid items 
            for the corresponding category of k-core graph amazon data. 
        """
        df = pd.read_json(path, orient="records", lines=True)
        csv_rows, idx, item_id2emb_idx = [], 0, {}
        with open(path) as file:
            for line in file:
                single_item = json.loads(line)
                item_id = single_item["asin"]
                if item_id in item_set:
                    item_title, item_descrption, item_imgurl = " ", " ", ""
                    """
                    if single_item.get("title") and \
                       single_item.get("description") and \
                       single_item.get("imageURLHighRes"):
                        item_title = single_item["title"]
                        item_descrption = single_item["description"][0]
                        item_descrption = item_descrption.replace('"', '').replace('\r', '').replace('\n', '')
                        item_descrption = " " if item_descrption == "" else item_descrption
                        item_imgurl = single_item["imageURLHighRes"][0]
                    """
                    
                    if single_item.get("title"): item_title = single_item["title"]
                    if single_item.get("description"): 
                        item_descrption = single_item["description"][0]
                        item_descrption = item_descrption.replace('"', '').replace('\r', '').replace('\n', '')
                        item_descrption = " " if item_descrption == "" else item_descrption
                    if single_item.get("imageURLHighRes"): item_imgurl = single_item["imageURLHighRes"][0]

                    if item_id not in item_id2emb_idx: # avoid duplicate adding
                        csv_rows.append([item_id, item_title, item_descrption, item_imgurl])
                        item_id2emb_idx[item_id] = idx
                        idx += 1

        print("The size of current item_set:", len(item_set))
        print("The size of current item_id2emb_idx:", len(item_id2emb_idx))

        # write into csv
        csv_fields = ["item_id", "title", "description", "imageURLHighRes"]
        with open(Path(data_dir + "item_TextAndImage.csv"), 'w') as f:
            write = csv.writer(f)
            write.writerow(csv_fields)
            write.writerows(csv_rows)
        print("Save to " + data_dir + "item_TextAndImage.csv done!")



    @staticmethod
    def get_item_text(path: Path, item_set: set) -> Tuple[list, dict]:
        item_text_list, item_id2emb_idx = [], {}
        with open(path) as file:
            for line in file:
                single_item = json.loads(line)
                item_id = single_item["asin"]
                if item_id in item_set:
                    item_desc = ""
                    if single_item.get("title"):
                        item_desc += single_item["title"] + " "

                    if single_item.get("description") and \
                        single_item["description"][0] != "":
                        item_desc += single_item["description"][0]

                    if item_desc != "":
                        if item_id not in item_id2emb_idx:
                            item_id2emb_idx[item_id] = len(item_text_list)
                            item_text_list.append(item_desc)
        
        print("The size of current item_id2emb_idx:", len(item_id2emb_idx))
        return item_text_list, item_id2emb_idx



    @staticmethod
    @torch.no_grad()
    def get_text_emb(text_list: list,
                     item_list: list, 
                     item_id2emb_idx: dict, 
                     device: str, 
                     user=False) -> Tuple[torch.Tensor, int]:
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        dataset = Text(text_list, tokenizer)
        data_loader = DataLoader(dataset, batch_size=256, num_workers=16)
        model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
        model.to(device)
        model.eval()

        emb_list = []
        # for input_ids, token_type_ids, attention_mask in tqdm(data_loader):
        for input_ids in tqdm(data_loader):
            input_ids = input_ids.to(device)
            emb_list.append(model(**{"input_ids": input_ids}).last_hidden_state[:, 0].detach().cpu())
        emb_list = torch.cat(emb_list)
        n_item_w_text = len(emb_list)

        emb_list_all = []
        for item_id in item_list: # for all valid items
            if item_id in item_id2emb_idx: # items have description & title
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



def download_images(data_dir):
    import os
    import cv2
    import urllib.request
    img_dir = Path(data_dir, "images")
    if not os.path.exists(img_dir): os.makedirs(img_dir)
    df = pd.read_csv(Path(data_dir, "item_TextAndImage.csv"))
    item_ids, url_lists = list(df["item_id"]), list(df["imageURLHighRes"])
    for item_id, url in zip(item_ids, url_lists):
        item_img_path = str(img_dir) + "/" + item_id + ".jpg"
        try:
            urllib.request.urlretrieve(url, item_img_path)
            print("Image saved:", item_img_path)
        except:
            continue



def generate_review_emb(df, n_user, n_item, device, save_dir, embed_dim):
    user_ids, item_ids, review_list = list(df["user_id"]), list(df["item_id"]), list(df["review"])
    
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    dataset = Text(review_list, tokenizer, 512)
    data_loader = DataLoader(dataset, batch_size=16, num_workers=16)
    model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    model.to(device)
    model.eval()

    emb_list = []
    for input_ids in tqdm(data_loader):
        input_ids = input_ids.to(device)
        emb_list.append(model(**{"input_ids": input_ids}).last_hidden_state[:, 0].detach().cpu())
    emb_list = torch.cat(emb_list)

    torch.save(emb_list, save_dir / "review_emb.pt")
    # print(len(user_ids), len(item_ids), len(review_list), emb_list.shape)
    
    emb_list = torch.load(save_dir / "review_emb.pt")
    print("Original Review Embedding Shape:", emb_list.shape)

    from model import AutoEncoder

    n_train = round(len(emb_list) * 0.9)
    train_set, val_set = random_split(emb_list, [n_train, len(emb_list) - n_train])
    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
    model = AutoEncoder(len(emb_list[0]), embed_dim)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)    

    for i in range(50):
        print(f"epoch {i + 1} --------")
        model.train()
        train_loss = 0
        for input in tqdm(train_loader):
            input = input.to(device)
            optimizer.zero_grad()
            loss = model(input)
            loss.backward()
            optimizer.step()
            train_loss += loss
        train_loss /= len(train_loader)
        print(f"train loss: {train_loss:.5f}")
        with torch.no_grad():
            model.eval()
            val_loss = 0
            for input in tqdm(val_loader):
                input = input.to(device)
                loss = model(input)
                val_loss += loss * len(input)
            val_loss /= len(val_set)
            print(f"valid loss: {val_loss:.5f}")

    data_loader = DataLoader(emb_list, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
    with torch.no_grad():
        model.eval()
        encoded_emb = []
        for input in tqdm(data_loader):
            input = input.to(device)
            encoded_emb.append(model.encoder(input).cpu())
        encoded_emb = torch.cat(encoded_emb)

    print("Dimension Reducted Review Embedding Shape:", encoded_emb.shape)    
    torch.save(encoded_emb, save_dir / "review_emb_dimrd.pt")



def generate_review_cube(df, n_user, n_item, device, save_dir):
    # dimrd_emb = torch.load(save_dir / "review_emb.pt")
    dimrd_emb = torch.load(save_dir / "review_emb_dimrd.pt")
    print(dimrd_emb.shape)
    
    user_ids, item_ids = list(df["user_id"]), list(df["item_id"])
    assert len(user_ids) == len(item_ids) == dimrd_emb.size(0), "Error!!! Dimension mismatch!!!"

    train_list = np.loadtxt(save_dir / "train_list.csv", delimiter=",", dtype=np.int64).tolist()
    val_list = np.loadtxt(save_dir / "val_list.csv", delimiter=",", dtype=np.int64).tolist()
    test_list = np.loadtxt(save_dir / "test_list.csv", delimiter=",", dtype=np.int64).tolist()
    
    print(len(train_list), len(val_list), len(test_list))
    cnt = 0
    review_cube = torch.full((n_user, n_item, dimrd_emb.size(1)), 0.0)
    for user_id, item_id, review_emb in tqdm(zip(user_ids, item_ids, dimrd_emb)):
        if ([user_id, item_id] in val_list) or ([user_id, item_id] in test_list): continue
        else: 
            review_cube[user_id, item_id, :] = review_emb[:]
            cnt += 1
    assert cnt == len(train_list), "Review cube error!!! Size mismatch with the length of train_list."

    torch.save(review_cube.to_sparse(), save_dir / "review_cube.pt")
    review_cube = torch.load(save_dir / "review_cube.pt")
    print(review_cube.shape)
    

"""
def generate_review_A_cube(df=df, n_user, n_item, device, save_dir):
    dimrd_emb = torch.load(save_dir / "review_emb_dimrd.pt")
    print(dimrd_emb.shape)
    
    user_ids, item_ids = list(df["user_id"]), list(df["item_id"])
    assert len(user_ids) == len(item_ids) == dimrd_emb.size(0), "Error!!! Dimension mismatch!!!"

    train_list = np.loadtxt(save_dir / "train_list.csv", delimiter=",", dtype=np.int64).tolist()
    val_list = np.loadtxt(save_dir / "val_list.csv", delimiter=",", dtype=np.int64).tolist()
    test_list = np.loadtxt(save_dir / "test_list.csv", delimiter=",", dtype=np.int64).tolist()
    
    cnt = 0
    review_A = torch.full((n_user + n_item, n_item + n_user, dimrd_emb.size(1)), 0.0)
    for user_id, item_id, review_emb in tqdm(zip(user_ids, item_ids, dimrd_emb)):
        if ([user_id, item_id] in val_list) or ([user_id, item_id] in test_list): continue
        else: 
            review_A[user_id, n_user + item_id, :] = review_emb[:]
            review_A[n_user + item_id, user_id, :] = review_emb[:]
            cnt += 1
    assert cnt == len(train_list), "Review cube error!!! Size mismatch with the length of train_list."

    torch.save(review_A.to_sparse(), save_dir / "review_A.pt")
    review_A = torch.load(save_dir / "review_A.pt")
    print(review_A.shape)
"""    



def create_item_json_for_albef(data_dir):
    # each train_file (json) contains a python list where each item is 
    # {'image': img_path, 'caption': text or list_of_text}
    df = pd.read_csv(Path(data_dir / "item_TextAndImage.csv"))
    img_ids, texts_t, texts_d = list(df["item_id"]), list(df["title"]), list(df["description"])
    print(len(img_ids), len(texts_t), len(texts_d))
    
    img_txt_list = []
    for img_id, text_t, text_d in zip(img_ids, texts_t, texts_d):
        if not (Path(data_dir / "images" / str(img_id + ".jpg"))).exists(): continue
        img_txt_list.append({
            "image": # the path is set from the directory of albef
                "./../gcn/" + str(Path(data_dir / "images" / str(img_id + ".jpg"))), 
            "caption": 
                str(text_t) # str(text_t + " " + text_d)
        })
    with open((data_dir / "item_json_for_albef.json"), 'w', encoding='utf-8') as fp:
        json.dump(img_txt_list, fp, ensure_ascii=False, indent=4)
    print("Wrtie to", str(data_dir / "item_json_for_albef.json"), "done!")



def process_single_domain(data_dir, save_dir):
    # Reindex, split into train/val/text sets, and compute adjacency matrix.
    data_dir, save_dir = Path(data_dir), Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    data = torch.load(data_dir / "data.pt")
    df = data["df"]
    user_list = data["user_list"]
    item_list = data["item_list"]

    user_mapping = {user_id: i for i, user_id in enumerate(user_list)}
    item_mapping = {item_id: i for i, item_id in enumerate(item_list)}

    with open(Path(data_dir) / "user_mapping.pkl", "wb") as fp: 
        pickle.dump(user_mapping, fp)
    with open(Path(data_dir) / "item_mapping.pkl", "wb") as fp: 
        pickle.dump(item_mapping, fp)

    df["user_id"] = df["user_id"].map(user_mapping)
    df["item_id"] = df["item_id"].map(item_mapping)
    
    edge_list = df[["user_id", "item_id"]].to_numpy()
    train_list, val_list, test_list = split_edges(edge_list, val_ratio=0.05, test_ratio=0.2)

    train_A, train_A_hat = get_A_norm(train_list, len(user_list), len(item_list))

    np.savetxt(save_dir / "train_list.csv", train_list, delimiter=",", fmt="%s")
    np.savetxt(save_dir / "val_list.csv", val_list, delimiter=",", fmt="%s")
    np.savetxt(save_dir / "test_list.csv", test_list, delimiter=",", fmt="%s")
    torch.save({"A": train_A, "A_hat": train_A_hat}, save_dir / "A.pt")
   
     
    # a) Generate item images
    download_images(data_dir)
    # b) Generate item's image & text pair for ALBEF
    create_item_json_for_albef(data_dir)
    # c) Generate review embeddings
    df["review"] = df["review"].apply(
        lambda review : " ".join(
            str(review).strip().strip("\"").strip("\n").split()
        )
    )
    generate_review_emb(df=df, n_user=len(user_mapping), n_item=len(item_mapping), 
                        device="cuda:0", save_dir=save_dir, embed_dim=128)
    generate_review_cube(df=df, n_user=len(user_mapping), n_item=len(item_mapping),
                         device="cuda:0", save_dir=save_dir)
    
    # generate_review_A_cube(df=df, n_user=len(user_mapping), n_item=len(item_mapping),
    #                        device="cuda:0", save_dir=save_dir)
    



if __name__ == "__main__":
    # CDs  Clothing_Shoes_and_Jewelry  Grocery  Luxury_Beauty  Magazine_Subscriptions
    
    # handle_reproducibility(seed=0)
    # data_dir = "data/Grocery/"
    # AmazonReview(data_dir, device="cuda:0", k_core=10).save_data(data_dir)
    # handle_reproducibility(seed=0)
    # process_single_domain(data_dir, save_dir=data_dir)

    
    handle_reproducibility(seed=0)
    data_dir = "data/Electronics/"
    AmazonReview(data_dir, device="cuda:0", k_core=16).save_data(data_dir)
    handle_reproducibility(seed=0)
    process_single_domain(data_dir, save_dir=data_dir)
    
    # handle_reproducibility(seed=0)
    # data_dir = "data/Tools/"
    # AmazonReview(data_dir, device="cuda:1", k_core=10).save_data(data_dir)
    # handle_reproducibility(seed=0)
    # process_single_domain(data_dir, save_dir=data_dir)

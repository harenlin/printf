import os
import csv
import cv2
import torch
import argparse
import numpy as np
import pandas as pd
import urllib.request
from pathlib import Path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="data")
    parser.add_argument("--csv_file", type=str, default="item_TextAndImage.csv")
    parser.add_argument("--dataset", type=str, default="Electronics")
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    args = parser.parse_args()

    img_dir = args.data_dir / args.dataset / "images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    df = pd.read_csv(args.data_dir / args.dataset / args.csv_file)
    item_ids, url_lists = list(df["item_id"]), list(df["imageURLHighRes"])
    for item_id, url in zip(item_ids, url_lists):
        full_path = str(img_dir) + "/" + item_id + '.jpg'
        try:
            urllib.request.urlretrieve(url, full_path)
        except:
            continue

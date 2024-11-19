import os
import json
import time
import torch
import random
import pickle
import datetime
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
import ruamel.yaml as yaml
from abc import ABC, abstractmethod

from PIL import Image, ImageFile
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from models.model_pretrain import ALBEF
from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer

import utils
from dataset.utils import pre_caption
# from dataset import create_dataset, create_sampler, create_loader


class pretrain_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):        
        self.ann = []
        print(ann_file)
        for f in ann_file:
            print(f)
            self.ann += json.load(open(f,'r'))
        self.transform = transform
        self.max_words = max_words
        
    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):    
        ann = self.ann[index]
        item_id = str(ann['image'][-14:-4])
        caption = pre_caption(ann['caption'], self.max_words)
        image = Image.open(ann['image']).convert('RGB')   
        image = self.transform(image)
        return item_id, image, caption



class BaseModel(nn.Module, ABC):
    """Base class for models."""
    @abstractmethod
    def forward(self, *inputs):
        return NotImplemented

    def __str__(self) -> str:
        """For printing the model and the number of trainable parameters."""
        n_params = sum([p.numel() for p in self.parameters() if p.requires_grad])

        separate_line_str = "-" * 70

        return (
            f"{separate_line_str}\n{super().__str__()}\n{separate_line_str}\n"
            f"Trainable parameters: {n_params}\n{separate_line_str}"
        )

class AutoEncoder(BaseModel):
    def __init__(self, input_size, encode_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 384, bias=False),
            nn.LeakyReLU(),
            nn.Linear(384, encode_size, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.Linear(encode_size, 384, bias=False),
            nn.LeakyReLU(),
            nn.Linear(384, input_size, bias=False),
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        loss = self.loss_fn(self.decoder(self.encoder(x)), x)
        return loss

def train_ae(init_item_emb: torch.Tensor, embed_dim: int, save_path: Path):
    n_train = round(len(init_item_emb) * 0.9)
    train_set, val_set = random_split(init_item_emb, [n_train, len(init_item_emb) - n_train])
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
    model = AutoEncoder(len(init_item_emb[0]), embed_dim)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)    

    for i in range(args.ae_epochs):
        print(f"epoch {i + 1} --------")
        model.train()
        train_loss = 0
        for input in tqdm(train_loader):
            input = input.to(args.device)
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
                input = input.to(args.device)
                loss = model(input)
                val_loss += loss * len(input)
            val_loss /= len(val_set)
            print(f"valid loss: {val_loss:.5f}")

    data_loader = DataLoader(init_item_emb, batch_size=512, shuffle=False, num_workers=16, pin_memory=True)
    with torch.no_grad():
        model.eval()
        encoded_emb = []
        for input in tqdm(data_loader):
            input = input.to(args.device)
            encoded_emb.append(model.encoder(input).cpu())
        encoded_emb = torch.cat(encoded_emb)

    torch.save(encoded_emb, save_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default="./../gcn/data/")
    parser.add_argument('--dataset', type=str, default="Grocery")
    parser.add_argument('--checkpoint', type=str, default="./output/4m_extend/checkpoint_09.pth") 
    parser.add_argument('--ckpt_ver', type=str, default="")
    parser.add_argument('--config', type=str, default="./configs/Pretrain.yaml")
    parser.add_argument('--config_train_file', type=str, 
        default="./../gcn/data/Grocery/item_json_for_albef.json")
    parser.add_argument('--text_encoder', type=str, default="bert-base-uncased")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--ae_epochs', type=int, default=50)
    args = parser.parse_args()

    data_dir = Path(Path(args.data_dir) / Path(args.dataset))
    
    print("Creating model!")
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer, init_deit=True)

    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']                       
        model.load_state_dict(state_dict)    
        print('Load checkpoint from %s'%args.checkpoint)
    
    model = model.to(args.device)
    text_encoder = model.text_encoder
    image_encoder = model.visual_encoder


    print("Creating dataset and preprocessing for items!")
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])
    config["train_file"] = [args.config_train_file]
    dataset = pretrain_dataset(config["train_file"], transform)
    data_loader = DataLoader(dataset, batch_size=32, num_workers=16)
    

    print("Start generating items' image and text embedding!")
    item_ids, img_embs, txt_embs = [], [], []
    for item_id, image, text in tqdm(data_loader):
        item_id = list(item_id)

        image = image.to(args.device, non_blocking=True) 
        text = tokenizer(text, padding='longest', truncation=True, max_length=25, return_tensors="pt").to(args.device)

        image_embed = image_encoder(image)[:,0].detach().cpu()
        text_embed = text_encoder.bert(text.input_ids, 
            attention_mask=text.attention_mask, return_dict=True, mode="text").last_hidden_state[:,0].detach().cpu()      
        # print(len(item_id), image_embed.shape, text_embed.shape)

        item_ids += item_id
        img_embs.append(image_embed)
        txt_embs.append(text_embed)
    
    img_embs, txt_embs = torch.cat(img_embs), torch.cat(txt_embs)
    print(len(item_ids), img_embs.shape, txt_embs.shape)

    items_list = pickle.load(open(Path(data_dir / "item_list.pkl"), "rb"))
    print("There are", len(items_list), "distinct items in this dataset.")

    img_emb_list_all, txt_emb_list_all = [], []
    for item_id in items_list: # for all valid items
        if item_id in item_ids: # if the item have title and image
            index = item_ids.index(item_id)
            img_emb_list_all.append(img_embs[index])
            txt_emb_list_all.append(txt_embs[index])
        else: # if the item have no title and image
            img_emb_list_all.append(torch.zeros(768))
            txt_emb_list_all.append(torch.zeros(768))
    img_emb_list_all = torch.stack(img_emb_list_all)
    txt_emb_list_all = torch.stack(txt_emb_list_all)
    print(img_emb_list_all.shape, txt_emb_list_all.shape) 

    torch.save(img_emb_list_all, data_dir / "item_image_emb.pt")
    torch.save(txt_emb_list_all, data_dir / "item_text_emb.pt")
    print("Item initial embedding (before train_ae) saved successfully!")
    

    print("Generating dimension reducted embedding for items!")
    init_image_emb = torch.load(data_dir / str(args.ckpt_ver + "item_image_emb.pt"))
    init_text_emb = torch.load(data_dir / str(args.ckpt_ver + "item_text_emb.pt"))

    train_ae(init_text_emb, 64, Path(data_dir / str(args.ckpt_ver + "item_text_emb_dimrd.pt")))
    train_ae(init_image_emb, 64, Path(data_dir / str(args.ckpt_ver + "item_image_emb_dimrd.pt")))
    init_text_emb_dimrd = torch.load(data_dir / str(args.ckpt_ver + "item_text_emb_dimrd.pt"))
    init_image_emb_dimrd = torch.load(data_dir / str(args.ckpt_ver + "item_image_emb_dimrd.pt"))
    init_item_emb_dimrd = torch.cat((init_text_emb_dimrd, init_image_emb_dimrd), dim=1)
    torch.save(init_item_emb_dimrd, data_dir / str(args.ckpt_ver + "init_item_emb_dimrd.pt"))
    print(init_item_emb_dimrd.shape)
    print("Wrtie to", str(args.ckpt_ver + "init_item_emb_dimrd.pt"), "done!")

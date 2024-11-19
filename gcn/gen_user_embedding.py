import argparse
import torch
from pathlib import Path
from utils import handle_reproducibility

def gen_printf_user(args):
    ckpt_dir = args.ckpt_dir / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    if args.model == "printf":
        if args.item_emb_method == "txtimg":
            init_item_emb = torch.load(args.data_dir / args.dataset / "init_item_emb_dimrd.pt")
        elif args.item_emb_method == "txt":
            init_item_emb = torch.load(args.data_dir / args.dataset / "item_text_emb_dimrd.pt")
        elif args.item_emb_method == "img":
            init_item_emb = torch.load(args.data_dir / args.dataset / "item_image_emb_dimrd.pt")
        # print(init_item_emb)
        print("Current item embedding shape:", init_item_emb.shape)
        init_user_emb = None
        review_emb = torch.load(args.data_dir / args.dataset / "review_cube.pt")
        print("Current review embedding cube shape:", review_emb.shape)
        review_bool = torch.sparse_coo_tensor(
            review_emb._indices(), 
            torch.ones(review_emb._indices().shape[1]), 
            review_emb.shape
        )
        # print(review_bool)
        cnt = torch.sparse.sum(review_bool, dim=0).to_dense()
        # (1) sum the dim of user to get overall review emb
        squeezed_review_emb = torch.sparse.sum(review_emb, dim=0).to_dense()
        # (2) average review emb
        avg_review_emb = squeezed_review_emb / cnt
        # (3) generate D(128,128) relation matrix
        attn = torch.matmul(torch.t(avg_review_emb), init_item_emb) # (r_128, i_128)
        # (4) norm it
        attn = torch.nn.functional.softmax(attn, dim=1)
        print(attn)
        # (5) propagate review information via item to user
        ri_interact = torch.matmul(review_emb.to_dense(), attn) # (n_u, n_i, 128)
        ri_interact = torch.nn.functional.softmax(ri_interact, dim=1)
        # print(ri_interact)
        user_emb = torch.sum(ri_interact * init_item_emb, dim=1)
        torch.save(user_emb, args.data_dir / args.dataset / "printf_user_emb.pt")
        print(user_emb)
        print(user_emb.shape)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="data/")
    parser.add_argument("--ckpt_dir", type=Path, default="ckpt/")
    parser.add_argument("--dataset", type=str, default="Electronics")
    parser.add_argument("--model", type=str, default="printf")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--review_embed_dim", type=int, default=128)
    parser.add_argument("--item_emb_method", type=str, choices=["txt", "img", "txtimg"], default="txtimg")
    parser.add_argument("--user_emb_method", type=str, choices=["rand", "agg", "none"], default="none")
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    handle_reproducibility(seed=args.seed)
    gen_printf_user(args)

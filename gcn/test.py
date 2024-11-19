import argparse
import torch
from pathlib import Path

from dataset import AmazonReview
from utils import handle_reproducibility, Metrics, eval_recsys
from model import PRINTF

@torch.no_grad()
def test(args):
    dataset = AmazonReview(
        data_dir=args.data_dir / args.dataset,
        mode="test",
    )

    init_item_emb = torch.zeros(dataset.n_item, args.embed_dim)
    init_user_emb = torch.zeros(dataset.n_user, args.embed_dim)

    model = PRINTF(
        init_item_emb=init_item_emb,
        init_user_emb=init_user_emb,
        n_layer=args.n_gcn_layer,
        dropout=0.0,
    )
    dataset.A = dataset.A.to(args.device)

    model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device), strict=False)
    model.to(args.device)
    model.eval()

    metrics = Metrics(topk=args.topk, device=args.device)

    user_emb, item_emb = model.get_emb(dataset.A)

    sum_recall, sum_ndcg = eval_recsys(
        dataset.test_user_list,
        dataset.test_user2pos_item,
        dataset.train_val_user2pos_item,
        user_emb,
        item_emb,
        metrics,
    )

    test_log = {
        "test_avg_recall": sum_recall / len(dataset.test_user_list),
        "test_avg_ndcg": sum_ndcg / len(dataset.test_user_list),
    }
    for key, value in test_log.items():
        print(f"{key:30s}: {value:.4}")
    return test_log


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="data/")
    parser.add_argument("--ckpt_path", type=Path, default="ckpt/Electronics/best_model_train.pt")
    parser.add_argument("--dataset", type=str, default="Electronics")
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--device", type=torch.device, default="cuda:6")
    parser.add_argument("--n_gcn_layer", type=int, default=7)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    handle_reproducibility(seed=0)
    args = parse_args()
    test(args)

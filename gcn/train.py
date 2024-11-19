import argparse
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from tqdm import tqdm

from dataset import AmazonReview
# from new_preprocess import AmazonReview
from utils import eval_recsys, handle_reproducibility, Metrics
from model import PRINTF

def train(args):
    ckpt_dir = args.ckpt_dir / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset = AmazonReview(
        data_dir=args.data_dir / args.dataset,
        mode="train",
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=16,
        pin_memory=True,
    )

    init_item_emb = torch.load(args.data_dir / args.dataset / "init_item_emb_dimrd.pt")
    init_user_emb = torch.load(args.data_dir / args.dataset / "printf_user_emb.pt")
    dataset.A = dataset.A.to(args.device)
    model = PRINTF(init_item_emb = init_item_emb, init_user_emb = init_user_emb, n_layer = args.n_gcn_layer, dropout=0.0)
    model.to(args.device)

    if args.optimizer == "AdamW":
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "SGD":
        optimizer = SGD(model.parameters(), lr=args.lr, weight_decay=args.wd)

    args.ckpt_path = ckpt_dir / f"best_model_{args.exp_name}.pt"
    print(model)

    metrics = Metrics(topk=20, device=args.device)

    best_metric = 0
    last_best_epoch = 0
    for epoch in range(1, args.n_epoch + 1):
        print(f"----- Epoch {epoch} -----")
        model.train()
        train_loss = 0
        for user, pos_item, neg_item in tqdm(data_loader):
            user = user.to(args.device)
            pos_item = pos_item.to(args.device)
            neg_item = neg_item.to(args.device)
            optimizer.zero_grad()
            loss = model(user, pos_item, neg_item, dataset.A)
            loss.backward()
            optimizer.step()

            train_loss += loss

        train_log = {
            "train_loss": train_loss / len(data_loader),
        }
        for key, value in train_log.items():
            print(f"{key:30s}: {value:.4}")
        
        if epoch % args.n_epoch_per_valid != 0:
            continue
        
        with torch.no_grad():
            model.eval()

            user_emb, item_emb = model.get_emb(dataset.A)

            sum_recall, sum_ndcg = eval_recsys(
                dataset.val_user_list,
                dataset.val_user2pos_item,
                dataset.train_user2pos_item,
                user_emb,
                item_emb,
                metrics,
            )

            valid_log = {
                "valid_avg_recall": sum_recall / len(dataset.val_user_list),
                "valid_avg_ndcg": sum_ndcg / len(dataset.val_user_list),
            }
            for key, value in valid_log.items():
                print(f"{key:30s}: {value:.4}")

        log = {**train_log, **valid_log}
        if log[args.metric_for_best] > best_metric:
            torch.save(model.state_dict(), args.ckpt_path)
            print(f"{'':30s}*** Best model saved ***")
            last_best_epoch = epoch
            best_metric = log[args.metric_for_best]
        elif epoch - last_best_epoch >= 100:
            print(f"*** Early stopped. Hasn't improved for {epoch - last_best_epoch} epochs ***")
            break

    # test_log = test(args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default="data/")
    parser.add_argument("--ckpt_dir", type=Path, default="ckpt/")
    parser.add_argument("--dataset", type=str, default="Electronics")
    parser.add_argument("--exp_name", type=str, default="train")
    parser.add_argument("--device", type=torch.device, default="cuda:0")
    parser.add_argument("--metric_for_best", type=str, default="valid_avg_ndcg")
    parser.add_argument("--optimizer", type=str, default="AdamW")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--n_epoch", type=int, default=1500)
    parser.add_argument("--n_epoch_per_valid", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--model", type=str, default="RITgcn")
    parser.add_argument("--n_gcn_layer", type=int, default=7, help="Only for GCN-based model")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    handle_reproducibility(seed=args.seed)
    train(args)

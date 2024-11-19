import dgl
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from sparselinear import SparseLinear


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


class PRINTF(BaseModel):
    """ init_user_emb and init_item_emb will be passed in from CMIM and RAUM, and will do EPIM. """
    def __init__(self, init_item_emb, init_user_emb, n_layer, dropout):
        super().__init__()
        self.item_emb = nn.Embedding.from_pretrained(init_item_emb, freeze=False)
        self.user_emb = nn.Embedding.from_pretrained(init_user_emb, freeze=False)
        self.n_layer = n_layer
        self.dropout = dropout

    def propagate(self, A, user_emb, item_emb):
        n_user = len(user_emb)

        if self.dropout and self.training:
            indices = A.indices()
            values = A.values()
            sample_indices = torch.rand(len(values)) + (1 - self.dropout)
            sample_indices = sample_indices.int().bool()
            indices = indices[:, sample_indices]
            values = values[sample_indices] / (1 - self.dropout)
            A = torch.sparse_coo_tensor(indices, values,  A.size(), device=A.device)
        
        all_emb = torch.cat([user_emb, item_emb])
        emb_list = [all_emb]
        for _ in range(self.n_layer):
            all_emb = torch.sparse.mm(A, all_emb)
            emb_list.append(all_emb)
        emb_list = torch.stack(emb_list)

        all_emb_mean = torch.mean(emb_list, dim=0)
        user_emb, item_emb = all_emb_mean[:n_user], all_emb_mean[n_user:]
        return user_emb, item_emb

    def get_emb(self, A):
        user_emb = self.user_emb.weight
        item_emb = self.item_emb.weight
        user_emb, item_emb = self.propagate(A, user_emb, item_emb)
        return user_emb, item_emb

    def forward(self, user, pos_item, neg_item, A):
        user_emb, item_emb = self.get_emb(A)

        user_emb = user_emb[user]
        pos_item_emb = item_emb[pos_item]
        neg_item_emb = item_emb[neg_item]
        pos_dot_prod = torch.sum(torch.mul(user_emb, pos_item_emb), dim=1)
        neg_dot_prod = torch.sum(torch.mul(user_emb, neg_item_emb), dim=1)
        loss = nn.functional.softplus(neg_dot_prod - pos_dot_prod).mean()
        # loss = -nn.functional.softplus(pos_dot_prod - neg_dot_prod).mean()

        return loss


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


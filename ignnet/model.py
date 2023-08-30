import torch
from torch.nn import Linear, BatchNorm1d
from torch_geometric.data import Data
from .layers import GCNConv as MPL, GreenBlock, FeedForwardPart


class IGNNetDefaultModel(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.fst = MPL(1, 64, device=device)
        self.snd = GreenBlock(64, device=device)
        self.thrd = GreenBlock(128, device=device)
        self.thrd_ = MPL(256, 256, device=device)
        self.frth = MPL(256, 256, device=device)
        self.ffth = Linear(384, 256, device=device)
        self.sxth = BatchNorm1d(256, device=device)
        self.svth = GreenBlock(256, device=device)
        self.eighth = BatchNorm1d(512, device=device)
        self.nnth = GreenBlock(512, device=device)
        self.tnth = MPL(1024, 1024, device=device)
        self.lvnth = Linear(1408, 256, device=device)
        self.twlth = BatchNorm1d(256, device=device)
        self.thrtnth = FeedForwardPart(768, 1, device=device)

    def forward(self, data: Data):
        edge_index = data.edge_index
        x = data.x
        skip_1 = self.fst(x, edge_index=edge_index)

        skip_2 = self.snd(skip_1, edge_index=edge_index)
        skip_2 = self.thrd(skip_2, edge_index=edge_index)
        skip_2 = self.thrd_(skip_2, edge_index=edge_index)
        skip_2 = self.frth(skip_2, edge_index=edge_index)
        skip_2 = torch.cat([skip_1, skip_2], dim=1)
        skip_2 = self.ffth(skip_2)
        skip_2 = self.sxth(skip_2)

        skip_3 = self.svth(skip_2, edge_index)
        skip_3 = self.eighth(skip_3)
        skip_3 = self.nnth(skip_3, edge_index)
        skip_3 = self.tnth(skip_3, edge_index)
        skip_3 = torch.cat([skip_3, skip_2])
        skip_3 = self.lvnth(skip_3)
        skip_3 = self.twlth(skip_3)

        last = torch.cat([skip_3, skip_2, skip_1], dim=1)
        return self.thrtnth(last)

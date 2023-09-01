import torch
from torch.nn import Linear, BatchNorm1d, Softmax
from .layers import IGNNetMPL as MPL, GreenBlock, FeedForwardPart
from .preprocessor import PreProcessor


class IGNNetDefaultModel(torch.nn.Module):
    def __init__(
        self,
        edge_index: torch.Tensor,
        adj_mat: torch.Tensor,
        num_features: int,
        num_classes: int,
        device="cuda",
    ):
        super().__init__()
        self.edge_index = edge_index
        self.num_features = num_features
        self.num_classes = num_classes
        self.normalize_input = BatchNorm1d(num_features)
        self.fst = Linear(1, 64, bias=False, device=device)
        self.snd = GreenBlock(64, adj_mat, device=device)
        self.thrd = GreenBlock(128, adj_mat, device=device)
        self.thrd_ = MPL(256, 256, adj_mat, device=device)
        self.frth = MPL(256, 256, adj_mat, device=device)
        self.ffth = Linear(320, 256, bias=False, device=device)
        self.sxth = BatchNorm1d(256, device=device)
        self.svth = GreenBlock(256, adj_mat, device=device)
        self.eighth = BatchNorm1d(512, device=device)
        self.nnth = GreenBlock(512, adj_mat, device=device)
        self.tnth = MPL(1024, 1024, adj_mat, device=device)
        self.lvnth = Linear(1280, 256, bias=False, device=device)
        self.twlth = BatchNorm1d(256, device=device)
        self.thrtnth = FeedForwardPart(576, 1, device=device)
        self.final = Linear(1, 1, bias=False, device=device)
        self.final_ = Softmax()

    def forward(self, x: torch.Tensor, batch_size=8):
        edge_index = self.edge_index
        skip_1 = self.fst(x)

        skip_2 = self.snd(skip_1, edge_index=edge_index)
        skip_2 = self.thrd(skip_2, edge_index=edge_index)
        skip_2 = self.thrd_(skip_2, edge_index=edge_index)
        skip_2 = self.frth(skip_2, edge_index=edge_index)
        skip_2 = torch.cat([skip_1, skip_2], dim=2)
        skip_2 = self.ffth(skip_2)
        skip_2 = torch.cat([self.sxth(x) for x in skip_2], dim=0).reshape(
            batch_size, self.num_features, 256
        )

        skip_3 = self.svth(skip_2, edge_index)
        skip_3 = torch.cat([self.eighth(x) for x in skip_3], dim=0).reshape(
            batch_size, self.num_features, 512
        )

        skip_3 = self.nnth(skip_3, edge_index)
        skip_3 = self.tnth(skip_3, edge_index)
        skip_3 = torch.cat([skip_3, skip_2], dim=2)
        skip_3 = self.lvnth(skip_3)
        skip_3 = torch.cat([self.twlth(x) for x in skip_3], dim=0).reshape(
            batch_size, self.num_features, 256
        )

        last = torch.cat([skip_3, skip_2, skip_1], dim=2)
        last = self.thrtnth(last)
        last = self.final(last).reshape(batch_size, self.num_features)
        return self.final_(last)

    @classmethod
    def from_preprocessor(cls, preprocessor: PreProcessor):
        return cls(
            preprocessor.edge_list,
            preprocessor.adj_matrix,
            preprocessor.num_nodes,
            preprocessor.num_classes,
            device=preprocessor.device,
        )

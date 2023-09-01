import torch
from torch.nn import Linear, Parameter, Sigmoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree


class IGNNetMPL(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        adj_mat: torch.Tensor,
        aggr="add",
        device="cuda",
    ):
        super().__init__(aggr=aggr, device=device)
        self._adj_mat = adj_mat
        self.lin = Linear(in_channels, out_channels, bias=False, device=device)
        self.activation = Sigmoid()
        self.bias = Parameter(torch.empty(out_channels, device=device))

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(1))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        result = self.activation(out)
        result[result == float("nan")] = 0
        return result


class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, device="cuda"):
        super().__init__(aggr="add", device=device)  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False, device=device)
        self.bias = Parameter(torch.empty(out_channels)).to(device=device)
        self.activation = torch.nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0))
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        deg_inv_sqrt[deg_inv_sqrt == float("nan")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        print(out)

        # Step 6: Apply a final bias vector.
        # out += self.bias

        return self.activation(out)

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j


class GreenBlock(torch.nn.Module):
    def __init__(self, in_channels, adj_mat, device="cuda"):
        super().__init__()
        self.mpl = IGNNetMPL(in_channels, in_channels, adj_mat, device=device)
        self.upper_1 = Linear(in_channels, in_channels, bias=False, device=device)
        self.upper_2 = Linear(in_channels, in_channels, bias=False, device=device)
        self.lower_1 = Linear(in_channels, in_channels, bias=False, device=device)
        self.lower_2 = Linear(in_channels, in_channels, bias=False, device=device)
        doubled_in_channels = 2 * in_channels
        self.last = Linear(
            doubled_in_channels, doubled_in_channels, bias=False, device=device
        )
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        fst = self.mpl(x, edge_index)
        upper = self.upper_2(self.upper_1(fst))
        lower = self.lower_2(self.lower_1(fst))
        combined = torch.cat([upper, lower], dim=2)
        # return combined
        last = self.last(combined)
        return self.relu(last)


class FeedForwardPart(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, device="cuda"):
        super().__init__()

        self.mods = [
            Linear(in_channels, 64, device=device, bias=False),
            Linear(64, 32, device=device, bias=False),
            Linear(32, 16, device=device, bias=False),
            Linear(16, 8, device=device, bias=False),
            Linear(8, 4, device=device, bias=False),
            Linear(4, 2, device=device, bias=False),
            Linear(2, 1, device=device, bias=False),
        ]
        self._mods = torch.nn.ModuleList(self.mods)

    def forward(self, x):
        for mod in self.mods:
            x = mod(x)
        return x

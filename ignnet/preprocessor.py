import torch
import pandas as pd


class PreProcessor(object):
    def __init__(
        self,
        num_nodes: int,
        adj_matrix: torch.Tensor,
        edge_list: torch.Tensor,
        num_classes: int,
        device="cuda",
    ):
        self.num_nodes = num_nodes
        self.adj_matrix = adj_matrix
        self.edge_list = edge_list
        self.num_classes = num_classes
        self._device = device

    @classmethod
    def from_dataframe(
        cls,
        X: pd.DataFrame,
        y: pd.DataFrame,
        corr_threshold: float = 0.2,
        auto_corr_val: float = 0.7,
        device="cuda",
    ):
        correlation_matrix = torch.from_numpy(X.corr().values).to(device=device)
        num_nodes = len(X.columns)
        num_classes = len(y.unique())  # type:ignore
        edge_list, adj_mat = cls._corr_mat_data(
            correlation_matrix,
            corr_threshold=corr_threshold,
            auto_corr_val=auto_corr_val,
            device=device,
        )  # Is this the right correlation?  Takes values in [0, 1]
        return cls(num_nodes, adj_mat, edge_list, num_classes, device=device)

    @staticmethod
    def _corr_mat_data(
        corr_matrix: torch.Tensor,
        corr_threshold: float = 0.2,
        auto_corr_val: float = 0.7,
        device="cuda",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a tuple (EdgeList, AdjMat)
        """
        mask = corr_matrix > corr_threshold
        filtered_corr_matrix = torch.where(
            mask, corr_matrix, torch.zeros_like(corr_matrix)
        )
        eye_mask = torch.eye(
            corr_matrix.shape[0], corr_matrix.shape[1], device=device
        ).bool()
        filtered_corr_matrix = torch.where(
            eye_mask,
            torch.full_like(corr_matrix, auto_corr_val, device=device),
            filtered_corr_matrix,
        )
        if (filtered_corr_matrix > corr_threshold).sum() == filtered_corr_matrix.shape[
            0
        ]:
            msg = (
                f"Too many correlations filtered out by corr_threshold={corr_threshold}"
            )
            raise ValueError(msg)

        edge_list = filtered_corr_matrix.nonzero(as_tuple=False)
        return (edge_list, filtered_corr_matrix)

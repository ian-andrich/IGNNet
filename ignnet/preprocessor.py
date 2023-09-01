import torch
import pandas as pd
from torch.utils.data import TensorDataset


class PreProcessor(object):
    """
    The model requires preprocessing in order to instantiate -- it requires the
    correlation matrix in order to be defined.  Further helper methods are provided
    in order to facilitate the ease of preparing the data
    """

    def __init__(
        self,
        dataset: TensorDataset,
        num_nodes: int,
        adj_matrix: torch.Tensor,
        edge_list: torch.Tensor,
        num_classes: int,
        device: str = "cuda",
    ):
        self.dataset = dataset
        self.num_nodes = num_nodes
        self.adj_matrix = adj_matrix
        self.edge_list = edge_list
        self.num_classes = num_classes
        self._device = device

    @classmethod
    def from_dataframe(
        cls,
        X: pd.DataFrame,
        y: pd.Series,
        auto_corr_val: float = 0.7,
        device="cuda",
        auto_coerce_y=True,
    ):
        """
        Assumes the X is all numerical.
        """
        X_ = torch.Tensor(X.values)
        if auto_coerce_y:
            y_ = torch.tensor(y.map(int).values)
        else:
            y_ = torch.Tensor(y.values)
        dataset = TensorDataset(X_, y_)
        correlation_matrix = torch.from_numpy(X.corr().values).to(device=device)
        num_nodes = len(X.columns)
        num_classes = len(y.map(int).unique())  # type:ignore
        edge_list, adj_mat = cls._auto_corr_mat_data(
            correlation_matrix,
            auto_corr_val,
            device=device,
        )  # Is this the right correlation?  Takes values in [0, 1]
        return cls(dataset, num_nodes, adj_mat, edge_list, num_classes, device=device)

    @classmethod
    def _auto_corr_mat_data(
        cls, corr_matrix: torch.Tensor, auto_corr_val: float = 0.7, device="cuda"
    ):
        for corr_threshold in [0.2, 0.1, 0.05, -0.01]:
            try:
                return cls._corr_mat_data(
                    corr_matrix,
                    corr_threshold,
                    auto_corr_val=auto_corr_val,
                    device=device,
                )

            except ValueError:
                pass

        raise ValueError("Should be unreachable")

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

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph_utils import corr_graph_topk


@dataclass(frozen=True)
class GraphArrays:
    price: np.ndarray  # [T, S, L, F]
    text: np.ndarray  # [T, S, D] (zeros if not used)
    text_mask: np.ndarray  # [T, S] bool
    future_ret: np.ndarray  # [T, S] or [T, S, H]
    dates: list[str]
    tickers: list[str]
    eligibility_mask: np.ndarray | None = None  # [T, S] bool
    unavailable_mask: np.ndarray | None = None  # [T, S] bool
    eligibility_liquidity: float | None = None
    eligibility_price_floor: float | None = None
    eligibility_market_cap_floor: float | None = None


class GraphSnapshotDataset(Dataset):
    """Turns prebuilt arrays into PyG `Data` snapshots.

    Each item is one *date snapshot* graph containing all stocks as nodes.
    Graph edges are computed from the **past log-return window** in `price[..., 0]`
    if your feature 0 is Log_Ret (as in this scaffold).
    """

    def __init__(
        self,
        arrays: GraphArrays,
        topk: int,
        corr_window: int,
        use_abs_corr: bool = True,
    ):
        self.arr = arrays
        self.topk = topk
        self.corr_window = corr_window
        self.use_abs_corr = use_abs_corr

        assert self.arr.price.ndim == 4, "price must be [T,S,L,F]"
        assert self.arr.text.ndim == 3, "text must be [T,S,D]"
        assert self.arr.future_ret.ndim in (2, 3), "future_ret must be [T,S] or [T,S,H]"
        assert len(self.arr.dates) == self.arr.price.shape[0]

    def __len__(self) -> int:
        return int(self.arr.price.shape[0])

    def __getitem__(self, idx: int) -> Data:
        price = torch.tensor(self.arr.price[idx], dtype=torch.float32)  # [S,L,F]
        text = torch.tensor(self.arr.text[idx], dtype=torch.float32)  # [S,D]
        mask = torch.tensor(self.arr.text_mask[idx].astype(np.float32), dtype=torch.float32)  # [S]
        y_reg = torch.tensor(self.arr.future_ret[idx], dtype=torch.float32)  # [S] or [S,H]
        if y_reg.ndim == 1:
            y = y_reg
            y_reg = y_reg.unsqueeze(-1)  # [S,1] for consistent regression code
        else:
            y = y_reg[:, -1].clone()  # default scalar label: last horizon

        # Build correlation graph from log returns (assumed feature 0) over the tail window.
        # returns_window: [S, W]
        L = price.shape[1]
        w = min(self.corr_window, L)
        returns_window = price[:, -w:, 0].cpu().numpy()
        edge_index, edge_attr = corr_graph_topk(
            returns_window, topk=self.topk, use_abs=self.use_abs_corr
        )

        data = Data(
            price=price,
            text=text,
            text_mask=mask,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_reg=y_reg,
            y=y,
            date=self.arr.dates[idx],
        )
        data.num_nodes = price.shape[0]
        return data


def load_graph_arrays(path: str) -> GraphArrays:
    npz = np.load(path, allow_pickle=True)
    eligibility_mask = npz["eligibility_mask"].astype(bool) if "eligibility_mask" in npz else None
    unavailable_mask = npz["unavailable_mask"].astype(bool) if "unavailable_mask" in npz else None
    return GraphArrays(
        price=npz["price"].astype(np.float32),
        text=npz["text"].astype(np.float32),
        text_mask=npz["mask"].astype(bool),
        future_ret=npz["future_ret"].astype(np.float32),
        dates=list(npz["dates"].tolist()),
        tickers=list(npz["tickers"].tolist()),
        eligibility_mask=eligibility_mask,
        unavailable_mask=unavailable_mask,
        eligibility_liquidity=(
            float(npz["eligibility_liquidity"]) if "eligibility_liquidity" in npz else None
        ),
        eligibility_price_floor=(
            float(npz["eligibility_price_floor"]) if "eligibility_price_floor" in npz else None
        ),
        eligibility_market_cap_floor=(
            float(npz["eligibility_market_cap_floor"])
            if "eligibility_market_cap_floor" in npz
            else None
        ),
    )


def save_graph_arrays(path: str, arrays: GraphArrays) -> None:
    payload = dict(
        price=arrays.price.astype(np.float32),
        text=arrays.text.astype(np.float32),
        mask=arrays.text_mask.astype(bool),
        future_ret=arrays.future_ret.astype(np.float32),
        dates=np.array(arrays.dates, dtype=object),
        tickers=np.array(arrays.tickers, dtype=object),
    )
    if arrays.eligibility_mask is not None:
        payload["eligibility_mask"] = arrays.eligibility_mask.astype(bool)
    if arrays.unavailable_mask is not None:
        payload["unavailable_mask"] = arrays.unavailable_mask.astype(bool)
    if arrays.eligibility_liquidity is not None:
        payload["eligibility_liquidity"] = float(arrays.eligibility_liquidity)
    if arrays.eligibility_price_floor is not None:
        payload["eligibility_price_floor"] = float(arrays.eligibility_price_floor)
    if arrays.eligibility_market_cap_floor is not None:
        payload["eligibility_market_cap_floor"] = float(arrays.eligibility_market_cap_floor)
    np.savez_compressed(path, **payload)

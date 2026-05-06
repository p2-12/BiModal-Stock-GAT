from pydantic import BaseModel


class PredictRequest(BaseModel):
    ticker: str
    date: str


class GraphSnapshotRequest(BaseModel):
    date: str
    tickers: list[str] | None = None

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class GoldenDatasetConfig:
    start_date: str
    end_date: str
    tickers: list[str]
    version: str


class GoldenDatasetGenerator:
    def __init__(self, output_root: str = "artifacts/golden"):
        self.output_root = Path(output_root)

    def generate(self, price_frames: dict[str, pd.DataFrame], cfg: GoldenDatasetConfig) -> Path:
        version_dir = self.output_root / cfg.version
        version_dir.mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "version": cfg.version,
            "start_date": cfg.start_date,
            "end_date": cfg.end_date,
            "tickers": sorted(cfg.tickers),
            "files": {},
        }
        start_ts = pd.Timestamp(cfg.start_date)
        end_ts = pd.Timestamp(cfg.end_date)

        for ticker in sorted(cfg.tickers):
            frame = price_frames[ticker].sort_index().loc[start_ts:end_ts].copy()
            file_path = version_dir / f"{ticker}.parquet"
            frame.to_parquet(file_path)
            manifest["files"][ticker] = {
                "path": file_path.name,
                "rows": int(len(frame)),
                "checksum": self._sha256(file_path),
            }

        manifest_path = version_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        (version_dir / "manifest.sha256").write_text(self._sha256(manifest_path))
        return version_dir

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

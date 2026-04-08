## Possible sample code??

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# Settings
# ============================================================

SEQ_LEN = 60
ROLLING_WINDOW = 20
HORIZON = 5
ENTRY_Z = 1.0
EXIT_Z = 0.5
HIDDEN_DIM = 32
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
SEED = 42
MIN_EVENT_SAMPLES = 120
PROB_THRESHOLD = 0.52
MIN_TRADES = 20


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Paths
# ============================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    if start is None:
        start = Path(__file__).resolve()

    for parent in [start] + list(start.parents):
        if (parent / "src" / "data" / "csv").exists() or (parent / "data" / "csv").exists():
            return parent

    raise FileNotFoundError("Could not find project root with src/data/csv or data/csv.")


def find_csv_file(root: Path, preferred_name: Optional[str] = None) -> Path:
    folders = [root / "src" / "data" / "csv", root / "data" / "csv"]
    candidates = []
    for folder in folders:
        if folder.exists():
            candidates.extend(sorted(folder.glob("*.csv")))

    if not candidates:
        raise FileNotFoundError("No CSV files found under src/data/csv or data/csv.")

    if preferred_name:
        for p in candidates:
            if p.name == preferred_name:
                return p

    return candidates[0]


# ============================================================
# Data loading
# ============================================================

def load_sector_csv(csv_path: Path) -> pd.DataFrame:
    """
    Handles the wide CSV format shown in your screenshot:
      row 1 = ticker names
      row 2 = fields (Open/High/Low/Close/Volume)
      row 3 = Date line
      row 4+ = actual data
    """
    try:
        df = pd.read_csv(
            csv_path,
            header=[0, 1],
            index_col=0,
            parse_dates=True,
            skiprows=[2],
        )
    except Exception:
        df = pd.read_csv(
            csv_path,
            header=[0, 1],
            index_col=0,
            parse_dates=True,
        )

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()].sort_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_tuples(
            [(str(a).strip(), str(b).strip()) for a, b in df.columns]
        )

    return df


def list_tickers(raw: pd.DataFrame) -> list[str]:
    if not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("Expected MultiIndex columns with tickers in level 0.")
    return [str(x).strip() for x in raw.columns.get_level_values(0).unique().tolist()]


def get_pair_frame(raw: pd.DataFrame, ticker_a: str, ticker_b: str) -> pd.DataFrame:
    a = raw[ticker_a].copy()
    b = raw[ticker_b].copy()

    out = pd.DataFrame(index=raw.index)
    out["price_a"] = a["Close"]
    out["price_b"] = b["Close"]
    out["volume_a"] = a["Volume"]
    out["volume_b"] = b["Volume"]
    out = out.dropna().copy()
    return out


# ============================================================
# Pair setup
# ============================================================

def estimate_beta_from_train(pair_df: pd.DataFrame, train_fraction: float = 0.6) -> float:
    """
    Estimate hedge ratio beta using only the early part of the series
    to avoid look-ahead leakage.
    """
    n = len(pair_df)
    split = max(20, int(n * train_fraction))
    train = pair_df.iloc[:split].copy()

    x = np.log(train["price_b"].values)
    y = np.log(train["price_a"].values)

    if len(x) < 2 or np.std(x) < 1e-12:
        return 1.0

    beta = np.polyfit(x, y, 1)[0]
    if not np.isfinite(beta):
        return 1.0
    return float(beta)


def add_features(pair_df: pd.DataFrame, beta: float, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    df = pair_df.copy()

    df["log_a"] = np.log(df["price_a"])
    df["log_b"] = np.log(df["price_b"])

    df["spread"] = df["log_a"] - beta * df["log_b"]
    df["spread_mean"] = df["spread"].rolling(window).mean()
    df["spread_std"] = df["spread"].rolling(window).std()
    df["zscore"] = (df["spread"] - df["spread_mean"]) / (df["spread_std"] + 1e-8)

    df["ret_a"] = df["price_a"].pct_change()
    df["ret_b"] = df["price_b"].pct_change()

    df["vol_a"] = df["ret_a"].rolling(window).std()
    df["vol_b"] = df["ret_b"].rolling(window).std()

    df["vol_ratio"] = df["volume_a"] / (df["volume_b"] + 1e-8)
    df["spread_change"] = df["spread"].diff()
    df["zscore_change"] = df["zscore"].diff()

    df = df.dropna().copy()
    return df


# ============================================================
# Event-based dataset
# ============================================================

def build_event_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    seq_len: int = SEQ_LEN,
    horizon: int = HORIZON,
    entry_z: float = ENTRY_Z,
    exit_z: float = EXIT_Z,
):
    """
    Use only "event" rows where abs(zscore) is large enough.
    Label = 1 if spread mean-reverts enough within the next horizon bars.
    """
    X, y = [], []
    entry_spread, future_spread, entry_zs = [], [], []
    event_idx = []

    max_i = len(df) - horizon - 1
    start_i = seq_len - 1

    for i in range(start_i, max_i + 1):
        z = float(abs(df["zscore"].iloc[i]))
        if z < entry_z:
            continue

        future_abs_z = df["zscore"].iloc[i + 1 : i + 1 + horizon].abs().values
        if len(future_abs_z) < horizon:
            continue

        label = float(np.min(future_abs_z) <= exit_z)

        seq = df[feature_cols].iloc[i - seq_len + 1 : i + 1].values
        if seq.shape != (seq_len, len(feature_cols)):
            continue

        X.append(seq)
        y.append(label)
        entry_spread.append(float(df["spread"].iloc[i]))
        future_spread.append(float(df["spread"].iloc[i + horizon]))
        entry_zs.append(float(df["zscore"].iloc[i]))
        event_idx.append(i)

    if len(X) == 0:
        return (
            np.empty((0, seq_len, len(feature_cols)), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            pd.DataFrame(),
        )

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    meta = pd.DataFrame(
        {
            "event_index": event_idx,
            "entry_spread": entry_spread,
            "future_spread": future_spread,
            "entry_zscore": entry_zs,
            "label": y,
        }
    )
    return X, y, meta


# ============================================================
# Dataset / model
# ============================================================

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        h0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim, device=x.device)

        out, _ = self.lstm(x, (h0, c0))
        last_hidden = out[:, -1, :]
        logits = self.head(last_hidden).squeeze(-1)
        return logits


# ============================================================
# Train / evaluation
# ============================================================

def safe_auc(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, probs))
    except Exception:
        return float("nan")


def train_model(model: nn.Module, train_loader: DataLoader, device: torch.device, pos_weight: float = 1.0) -> None:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for _ in range(EPOCHS):
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device):
    model.eval()
    probs_all = []
    y_all = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits)

            probs_all.append(probs.cpu().numpy())
            y_all.append(yb.numpy())

    probs = np.concatenate(probs_all) if probs_all else np.array([])
    y_true = np.concatenate(y_all) if y_all else np.array([])

    if len(probs) == 0:
        return {}

    preds = (probs >= 0.5).astype(np.float32)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "auc": safe_auc(y_true, probs),
    }
    return metrics, probs, y_true


def strategy_metrics(meta_test: pd.DataFrame, probs: np.ndarray, prob_threshold: float = PROB_THRESHOLD):
    if len(meta_test) == 0 or len(probs) == 0:
        return {
            "trades": 0,
            "trade_win_rate": np.nan,
            "avg_pnl_proxy": np.nan,
            "total_pnl_proxy": np.nan,
            "max_drawdown_proxy": np.nan,
        }

    z_ok = meta_test["entry_zscore"].abs().values >= ENTRY_Z
    signals = (probs >= prob_threshold) & z_ok

    if signals.sum() == 0:
        return {
            "trades": 0,
            "trade_win_rate": np.nan,
            "avg_pnl_proxy": np.nan,
            "total_pnl_proxy": 0.0,
            "max_drawdown_proxy": np.nan,
        }

    chosen = meta_test.loc[signals].copy()
    entry = chosen["entry_spread"].values
    future = chosen["future_spread"].values

    # trade direction: positive spread -> short, negative spread -> long
    trade_dir = -np.sign(entry)
    trade_dir[trade_dir == 0] = 1.0

    # tiny transaction cost proxy
    cost_per_trade = 0.0005
    pnl_proxy = trade_dir * (future - entry) - cost_per_trade

    cum_pnl = np.cumsum(pnl_proxy)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = running_max - cum_pnl
    max_dd = float(np.max(drawdown)) if len(drawdown) else np.nan

    win_rate = float((pnl_proxy > 0).mean()) if len(pnl_proxy) else np.nan

    return {
        "trades": int(len(pnl_proxy)),
        "trade_win_rate": win_rate,
        "avg_pnl_proxy": float(np.mean(pnl_proxy)) if len(pnl_proxy) else np.nan,
        "total_pnl_proxy": float(np.sum(pnl_proxy)) if len(pnl_proxy) else np.nan,
        "max_drawdown_proxy": max_dd,
    }

def choose_threshold(meta_val: pd.DataFrame, val_probs: np.ndarray) -> float:
    best_t = PROB_THRESHOLD
    best_score = -np.inf

    for t in np.arange(0.45, 0.71, 0.01):
        m = strategy_metrics(meta_val, val_probs, prob_threshold=float(t))
        if m["trades"] < MIN_TRADES:
            continue
        if not np.isfinite(m["avg_pnl_proxy"]):
            continue

        # prefer thresholds that make money and still trade enough
        score = m["avg_pnl_proxy"] * np.sqrt(m["trades"])
        if score > best_score:
            best_score = score
            best_t = float(t)

    return best_t

# ============================================================
# Per-pair scan
# ============================================================

def scan_pair(raw: pd.DataFrame, ticker_a: str, ticker_b: str, device: torch.device):
    pair = get_pair_frame(raw, ticker_a, ticker_b)

    if len(pair) < (SEQ_LEN + ROLLING_WINDOW + HORIZON + 20):
        return None

    beta = estimate_beta_from_train(pair, train_fraction=0.6)
    feat = add_features(pair, beta=beta, window=ROLLING_WINDOW)

    feature_cols = [
        "spread",
        "zscore",
        "ret_a",
        "ret_b",
        "vol_a",
        "vol_b",
        "vol_ratio",
        "spread_change",
        "zscore_change",
    ]

    X, y, meta = build_event_dataset(
        feat,
        feature_cols=feature_cols,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        entry_z=ENTRY_Z,
        exit_z=EXIT_Z,
    )

    n = len(X)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    X_train, X_val, X_test = X[:train_end], X[train_end:val_end], X[val_end:]
    y_train, y_val, y_test = y[:train_end], y[train_end:val_end], y[val_end:]

    meta_train = meta.iloc[:train_end].reset_index(drop=True)
    meta_val = meta.iloc[train_end:val_end].reset_index(drop=True)
    meta_test = meta.iloc[val_end:].reset_index(drop=True)

    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        return None

    # Scale using train only
    n_train, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, n_feat)
    X_val_2d = X_val.reshape(-1, n_feat)
    X_test_2d = X_test.reshape(-1, n_feat)

    X_train_scaled = scaler.fit_transform(X_train_2d).reshape(n_train, seq_len, n_feat)
    X_val_scaled = scaler.transform(X_val_2d).reshape(len(X_val), seq_len, n_feat)
    X_test_scaled = scaler.transform(X_test_2d).reshape(len(X_test), seq_len, n_feat)

    train_ds = SeqDataset(X_train_scaled, y_train)
    val_ds = SeqDataset(X_val_scaled, y_val)
    test_ds = SeqDataset(X_test_scaled, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = neg / max(pos, 1.0)

    model = LSTMClassifier(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM).to(device)
    train_model(model, train_loader, device, pos_weight=pos_weight)

    val_metrics, val_probs, _ = evaluate_model(model, val_loader, device)
    best_threshold = choose_threshold(meta_val, val_probs)

    test_metrics, test_probs, _ = evaluate_model(model, test_loader, device)
    strat_metrics = strategy_metrics(meta_test, test_probs, prob_threshold=best_threshold)

    result = {
        "ticker_a": ticker_a,
        "ticker_b": ticker_b,
        "beta": beta,
        "n_rows": len(feat),
        "n_events": len(X),
        "label_mean": float(y.mean()),
        "best_threshold": best_threshold,
        "val_auc": val_metrics["auc"],
        "test_accuracy": test_metrics["accuracy"],
        "test_balanced_accuracy": test_metrics["balanced_accuracy"],
        "test_f1": test_metrics["f1"],
        "test_auc": test_metrics["auc"],
        "trade_count": strat_metrics["trades"],
        "trade_win_rate": strat_metrics["trade_win_rate"],
        "avg_pnl_proxy": strat_metrics["avg_pnl_proxy"],
        "total_pnl_proxy": strat_metrics["total_pnl_proxy"],
        "max_drawdown_proxy": strat_metrics["max_drawdown_proxy"],
    }

    return result


def scan_all_pairs(raw: pd.DataFrame, tickers: list[str], max_pairs: Optional[int] = None) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = list(combinations(tickers, 2))
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    results = []

    for i, (ticker_a, ticker_b) in enumerate(pairs, start=1):
        try:
            print(f"[{i}/{len(pairs)}] Scanning {ticker_a}/{ticker_b} ...")
            out = scan_pair(raw, ticker_a, ticker_b, device)
            if out is not None:
                results.append(out)
                print(
                    f"  kept | auc={out['test_auc']:.3f} | bal_acc={out['test_balanced_accuracy']:.3f} "
                    f"| pnl_proxy={out['avg_pnl_proxy']:.5f} | events={out['n_events']}"
                )
            else:
                print("  skipped")
        except Exception as e:
            print(f"  error: {e}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="CSV filename inside src/data/csv or data/csv")
    parser.add_argument("--max_pairs", type=int, default=None, help="Limit number of pairs for quick testing")
    parser.add_argument("--output", type=str, default="pair_scan_results.csv", help="Output CSV name")
    args = parser.parse_args()

    set_seed(SEED)

    root = find_project_root()
    csv_path = find_csv_file(root, preferred_name=args.csv)

    print(f"Project root: {root}")
    print(f"CSV path: {csv_path}")

    raw = load_sector_csv(csv_path)
    tickers = list_tickers(raw)

    print(f"Tickers: {tickers}")
    print(f"Total pairs: {len(tickers) * (len(tickers) - 1) // 2}")

    results = scan_all_pairs(raw, tickers, max_pairs=args.max_pairs)

    if results.empty:
        print("No pairs passed the filters.")
        return

    results = results[results["trade_count"] >= MIN_TRADES].copy()

    results = results.sort_values(
        by=["avg_pnl_proxy", "trade_win_rate", "test_auc"],
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)

    print("\nTop results:")
    cols = [
        "ticker_a",
        "ticker_b",
        "n_events",
        "label_mean",
        "best_threshold",
        "test_auc",
        "test_balanced_accuracy",
        "trade_count",
        "trade_win_rate",
        "avg_pnl_proxy",
        "total_pnl_proxy",
        "max_drawdown_proxy",
    ]
    print(results[cols].head(15).to_string(index=False))

    out_path = root / args.output
    results.to_csv(out_path, index=False)
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
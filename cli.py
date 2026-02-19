"""
cli.py — interface de linha de comando para os modelos de previsão de preços.

Espelha as funcionalidades da API REST para os modelos univariado e multivariado.

Uso:
  python cli.py health --model uni
  python cli.py health --model multi

  python cli.py predict --model uni --ticker AAPL --n-days 5
  python cli.py predict --model uni --prices 182.5 183.0 184.1 ... --n-days 3

  python cli.py predict --model multi --ticker AAPL --n-days 5
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf

# ---------------------------------------------------------------------------
# Caminhos de artefatos
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

PATHS = {
    "uni": {
        "model":   BASE_DIR / "models" / "lstm_model.pt",
        "scaler":  BASE_DIR / "models" / "scaler.pkl",
        "config":  BASE_DIR / "models" / "model_config.json",
    },
    "multi": {
        "model":          BASE_DIR / "models_multi" / "lstm_model.pt",
        "feature_scalers": BASE_DIR / "models_multi" / "feature_scalers.pkl",
        "close_scaler":   BASE_DIR / "models_multi" / "close_scaler.pkl",
        "config":         BASE_DIR / "models_multi" / "model_config.json",
    },
}


# ---------------------------------------------------------------------------
# Definições dos modelos (deve espelhar train.py / train_multi.py)
# ---------------------------------------------------------------------------

class LSTMRegressor(torch.nn.Module):
    """Modelo univariado (Close apenas)."""

    def __init__(self, window_size: int, hidden_1: int = 64, hidden_2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1   = torch.nn.LSTM(input_size=1, hidden_size=hidden_1, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm2   = torch.nn.LSTM(input_size=hidden_1, hidden_size=hidden_2, batch_first=True)
        self.fc1     = torch.nn.Linear(hidden_2, 16)
        self.relu    = torch.nn.ReLU()
        self.fc2     = torch.nn.Linear(16, 1)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out     = self.dropout(out)
        out, _ = self.lstm2(out)
        out     = out[:, -1, :]
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


class MultiLSTMRegressor(torch.nn.Module):
    """Modelo multivariado (OHLC ou OHLCV)."""

    def __init__(self, n_features: int, hidden_1: int = 64, hidden_2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1   = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_1, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm2   = torch.nn.LSTM(input_size=hidden_1, hidden_size=hidden_2, batch_first=True)
        self.fc1     = torch.nn.Linear(hidden_2, 16)
        self.relu    = torch.nn.ReLU()
        self.fc2     = torch.nn.Linear(16, 1)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out     = self.dropout(out)
        out, _ = self.lstm2(out)
        out     = out[:, -1, :]
        out     = self.relu(self.fc1(out))
        return self.fc2(out)


# ---------------------------------------------------------------------------
# Carregamento de artefatos
# ---------------------------------------------------------------------------

def load_uni():
    p = PATHS["uni"]
    for key, path in p.items():
        if not path.exists():
            sys.exit(
                f"Artefato ausente: {path}\n"
                "Execute o treinamento primeiro: python train.py"
            )

    with p["config"].open(encoding="utf-8") as f:
        config = json.load(f)

    model = LSTMRegressor(
        window_size=int(config.get("window_size", 60)),
        hidden_1=int(config.get("hidden_1", 64)),
        hidden_2=int(config.get("hidden_2", 32)),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(torch.load(p["model"], map_location="cpu"))
    model.eval()

    scaler = joblib.load(p["scaler"])
    return model, scaler, config


def load_multi():
    p = PATHS["multi"]
    required = ["model", "feature_scalers", "close_scaler", "config"]
    for key in required:
        if not p[key].exists():
            sys.exit(
                f"Artefato ausente: {p[key]}\n"
                "Execute o treinamento primeiro: python train_multi.py"
            )

    with p["config"].open(encoding="utf-8") as f:
        config = json.load(f)

    n_features = int(config.get("n_features", 4))
    model = MultiLSTMRegressor(
        n_features=n_features,
        hidden_1=int(config.get("hidden_1", 64)),
        hidden_2=int(config.get("hidden_2", 32)),
        dropout=float(config.get("dropout", 0.2)),
    )
    model.load_state_dict(torch.load(p["model"], map_location="cpu"))
    model.eval()

    feature_scalers = joblib.load(p["feature_scalers"])
    close_scaler    = joblib.load(p["close_scaler"])
    return model, feature_scalers, close_scaler, config


# ---------------------------------------------------------------------------
# Busca de dados recentes
# ---------------------------------------------------------------------------

def _flatten_yfinance(data: pd.DataFrame) -> pd.DataFrame:
    """Remove MultiIndex de colunas que yfinance pode retornar."""
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def fetch_latest_close(ticker: str, window_size: int) -> List[float]:
    """Retorna os últimos `window_size` fechamentos do ticker."""
    raw = yf.download(ticker, period="90d", interval="1d", progress=False)
    if raw.empty:
        sys.exit(f"Nenhum dado encontrado para {ticker}.")
    raw = _flatten_yfinance(raw)
    if "Close" not in raw.columns:
        sys.exit("Coluna 'Close' não encontrada nos dados baixados.")
    series = pd.Series(raw["Close"]).dropna().tail(window_size).tolist()
    if len(series) < window_size:
        sys.exit(
            f"Dados insuficientes: preciso de pelo menos {window_size} fechamentos "
            f"recentes para {ticker}."
        )
    return series


def fetch_latest_ohlcv(ticker: str, window_size: int, features: List[str]) -> np.ndarray:
    """Retorna array (window_size, n_features) com os últimos dados OHLCV."""
    raw = yf.download(ticker, period="90d", interval="1d", progress=False)
    if raw.empty:
        sys.exit(f"Nenhum dado encontrado para {ticker}.")
    raw = _flatten_yfinance(raw)

    missing = [f for f in features if f not in raw.columns]
    if missing:
        sys.exit(f"Colunas ausentes nos dados de {ticker}: {missing}")

    df = raw[features].dropna().tail(window_size)
    if len(df) < window_size:
        sys.exit(
            f"Dados insuficientes: preciso de pelo menos {window_size} linhas "
            f"de dados recentes para {ticker}."
        )
    return df.values.astype(np.float64)


# ---------------------------------------------------------------------------
# Lógica de previsão
# ---------------------------------------------------------------------------

def forecast_uni(
    model: LSTMRegressor,
    scaler,
    prices: List[float],
    window_size: int,
    n_days: int,
    device: torch.device,
) -> List[float]:
    arr = np.array(prices, dtype=float)
    if len(arr) < window_size:
        sys.exit(f"São necessários pelo menos {window_size} preços históricos.")

    scaled = scaler.transform(arr.reshape(-1, 1))
    window = scaled[-window_size:]
    preds: List[float] = []

    for _ in range(n_days):
        x = torch.tensor(window.reshape(1, window_size, 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_scaled = model(x).cpu().numpy()[0, 0]
        window = np.vstack([window[1:], [[pred_scaled]]])
        preds.append(float(scaler.inverse_transform([[pred_scaled]])[0, 0]))

    return preds


def forecast_multi(
    model: MultiLSTMRegressor,
    feature_scalers: List,
    close_scaler,
    ohlcv: np.ndarray,          # shape (window_size, n_features)
    window_size: int,
    n_days: int,
    close_idx: int,
    device: torch.device,
) -> List[float]:
    n_features = ohlcv.shape[1]

    # Normaliza cada coluna com seu scaler
    scaled_cols = [
        feature_scalers[i].transform(ohlcv[:, i].reshape(-1, 1)).flatten()
        for i in range(n_features)
    ]
    scaled = np.column_stack(scaled_cols).astype(np.float32)  # (window_size, n_features)
    window = scaled.copy()
    preds: List[float] = []

    for _ in range(n_days):
        x = torch.tensor(window.reshape(1, window_size, n_features), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_scaled = float(model(x).cpu().numpy()[0, 0])

        pred_price = float(close_scaler.inverse_transform([[pred_scaled]])[0, 0])
        preds.append(pred_price)

        # Para continuar prevendo: cria nova linha onde todos os features tomam
        # o valor previsto para o Close (simplificação para projeção multi-step).
        new_row = window[-1].copy()
        new_row[close_idx] = pred_scaled
        window = np.vstack([window[1:], [new_row]])

    return preds


# ---------------------------------------------------------------------------
# Subcomandos
# ---------------------------------------------------------------------------

def cmd_health(args: argparse.Namespace) -> None:
    model_type = args.model
    config_path = PATHS[model_type]["config"]

    if not config_path.exists():
        print(f"[{model_type.upper()}] Nenhum modelo treinado encontrado em {config_path.parent}/")
        return

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    print(f"\n=== Status do modelo [{model_type.upper()}] ===")
    print(f"  Ticker         : {config.get('ticker', '?')}")
    print(f"  Período        : {config.get('period', '?')}")
    print(f"  Janela         : {config.get('window_size', '?')} dias")
    if model_type == "multi":
        print(f"  Features       : {', '.join(config.get('features', []))}")
    print(f"  Hidden layers  : {config.get('hidden_1', '?')} / {config.get('hidden_2', '?')}")
    print(f"  Dropout        : {config.get('dropout', '?')}")
    print(f"  Val loss       : {config.get('val_loss', '?')}")
    print(f"  Otimizado      : {'sim' if config.get('tuned') else 'não'}")
    print(f"  Artefatos em   : {config_path.parent}/")
    print()


def cmd_predict(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = args.model

    if model_type == "uni":
        model, scaler, config = load_uni()
        model.to(device)
        window_size = int(config.get("window_size", 60))
        ticker      = str(config.get("ticker", ""))

        if args.prices:
            prices = args.prices
            source = "fornecidos manualmente"
        else:
            t = args.ticker or ticker
            if not t:
                sys.exit("Forneça --ticker ou --prices para o modelo univariado.")
            print(f"Buscando os últimos {window_size} fechamentos de {t}...")
            prices = fetch_latest_close(t, window_size)
            ticker = t
            source = f"últimos {window_size} fechamentos de {t}"

        print(f"Prevendo {args.n_days} dia(s) com base em: {source}.")
        predictions = forecast_uni(model, scaler, prices, window_size, args.n_days, device)

    else:  # multi
        model, feature_scalers, close_scaler, config = load_multi()
        model.to(device)
        window_size = int(config.get("window_size", 60))
        features    = config.get("features", ["Open", "High", "Low", "Close"])
        close_idx   = int(config.get("close_idx", features.index("Close")))
        ticker      = args.ticker or str(config.get("ticker", ""))

        if not ticker:
            sys.exit("Forneça --ticker para o modelo multivariado.")

        print(f"Buscando os últimos {window_size} dias de dados OHLCV de {ticker}...")
        ohlcv = fetch_latest_ohlcv(ticker, window_size, features)
        print(f"Prevendo {args.n_days} dia(s) com base em dados de {ticker}.")
        predictions = forecast_multi(
            model, feature_scalers, close_scaler,
            ohlcv, window_size, args.n_days, close_idx, device
        )

    _print_predictions(predictions, ticker, model_type)


def _print_predictions(predictions: List[float], ticker: str, model_type: str) -> None:
    label = "univariado" if model_type == "uni" else "multivariado"
    print(f"\n=== Previsão de fechamento [{label}] — {ticker} ===")
    for i, p in enumerate(predictions, start=1):
        print(f"  Dia +{i:2d} : {p:.4f}")
    print()


# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Previsão de preços de ações via LSTM (univariado e multivariado).",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMANDO")

    # ---- health ----
    p_health = sub.add_parser("health", help="Exibe informações sobre o modelo treinado.")
    p_health.add_argument(
        "--model", choices=["uni", "multi"], default="uni",
        help="Qual modelo inspecionar: uni (univariado) ou multi (multivariado). Padrão: uni"
    )

    # ---- predict ----
    p_pred = sub.add_parser("predict", help="Prevê preços de fechamento futuros.")
    p_pred.add_argument(
        "--model", choices=["uni", "multi"], default="uni",
        help="Qual modelo usar: uni (univariado) ou multi (multivariado). Padrão: uni"
    )
    p_pred.add_argument(
        "--ticker", type=str, default=None,
        help="Ticker para buscar dados recentes quando --prices não for fornecido."
    )
    p_pred.add_argument(
        "--prices", type=float, nargs="+", default=None,
        help=(
            "[Somente para --model uni] Lista de preços de fechamento históricos. "
            "Quantidade mínima = window_size do modelo. "
            "Se omitido, os dados são baixados automaticamente via --ticker."
        )
    )
    p_pred.add_argument(
        "--n-days", type=int, default=1, metavar="N",
        help="Quantos dias futuros prever (1–30). Padrão: 1"
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "health":
        cmd_health(args)
    elif args.command == "predict":
        if args.n_days < 1 or args.n_days > 30:
            sys.exit("--n-days deve estar entre 1 e 30.")
        if args.model == "multi" and args.prices:
            print(
                "Aviso: --prices é ignorado no modo multivariado. "
                "Os dados OHLCV são sempre buscados automaticamente via --ticker."
            )
        cmd_predict(args)


if __name__ == "__main__":
    main()

"""
api/main_multi.py — FastAPI para o modelo LSTM multivariado (OHLC/OHLCV).

Execução:
  uvicorn api.main_multi:app --reload --port 8001

Endpoints:
  GET  /health   — status e configuração do modelo
  POST /predict  — previsão de fechamento para N dias
  GET  /metrics  — métricas Prometheus
"""

import json
import time
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models_multi"

MODEL_PATH          = MODELS_DIR / "lstm_model.pt"
FEATURE_SCALERS_PATH = MODELS_DIR / "feature_scalers.pkl"
CLOSE_SCALER_PATH   = MODELS_DIR / "close_scaler.pkl"
CONFIG_PATH         = MODELS_DIR / "model_config.json"

# ---------------------------------------------------------------------------
# Métricas Prometheus
# ---------------------------------------------------------------------------

REQUEST_COUNT   = Counter(
    "api_multi_requests_total",
    "Total de requisicoes (multi)",
    ["endpoint", "status"],
)
REQUEST_LATENCY = Histogram(
    "api_multi_request_latency_seconds",
    "Latencia das requisicoes (multi)",
    ["endpoint"],
)

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class MultiLSTMRegressor(torch.nn.Module):
    """Espelho de MultiLSTMRegressor de train_multi.py."""

    def __init__(
        self,
        n_features: int,
        hidden_1: int = 64,
        hidden_2: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm1   = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_1, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm2   = torch.nn.LSTM(input_size=hidden_1,   hidden_size=hidden_2, batch_first=True)
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

def load_artifacts():
    for path in (MODEL_PATH, FEATURE_SCALERS_PATH, CLOSE_SCALER_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Artefato ausente: {path}. Execute 'python train_multi.py' primeiro."
            )

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open(encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {
            "window_size": 60,
            "ticker": "DESCONHECIDO",
            "features": ["Open", "High", "Low", "Close"],
            "close_idx": 3,
            "n_features": 4,
            "hidden_1": 64,
            "hidden_2": 32,
            "dropout": 0.2,
        }

    model = MultiLSTMRegressor(
        n_features=int(config.get("n_features", 4)),
        hidden_1=int(config.get("hidden_1", 64)),
        hidden_2=int(config.get("hidden_2", 32)),
        dropout=float(config.get("dropout", 0.2)),
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    feature_scalers = joblib.load(FEATURE_SCALERS_PATH)
    close_scaler    = joblib.load(CLOSE_SCALER_PATH)

    return model, feature_scalers, close_scaler, config


# ---------------------------------------------------------------------------
# Schemas Pydantic
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    ticker: Optional[str] = Field(
        None,
        description=(
            "Ticker do ativo (ex.: 'AAPL'). Se omitido, usa o ticker com que o modelo foi treinado. "
            "Os dados OHLCV recentes são buscados automaticamente via yfinance."
        ),
    )
    n_days: int = Field(1, ge=1, le=30, description="Quantidade de dias futuros a prever.")


class PredictResponse(BaseModel):
    ticker: str
    features: List[str]
    predictions: List[float]
    n_days: int


# ---------------------------------------------------------------------------
# Helpers de inferência
# ---------------------------------------------------------------------------

def fetch_latest_ohlcv(ticker: str, window_size: int, features: List[str]) -> np.ndarray:
    """Retorna array (window_size, n_features) com os últimos fechamentos OHLCV."""
    raw = yf.download(ticker, period="90d", interval="1d", progress=False)
    if raw.empty:
        raise HTTPException(status_code=502, detail=f"Sem dados para o ticker '{ticker}'.")

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    missing = [f for f in features if f not in raw.columns]
    if missing:
        raise HTTPException(
            status_code=502,
            detail=f"Colunas ausentes nos dados de '{ticker}': {missing}. Disponíveis: {list(raw.columns)}",
        )

    df = raw[features].dropna().tail(window_size)
    if len(df) < window_size:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Dados insuficientes para '{ticker}': preciso de pelo menos "
                f"{window_size} linhas recentes, obtive {len(df)}."
            ),
        )

    return df.values.astype(np.float64)


def forecast(
    ohlcv: np.ndarray,           # shape (window_size, n_features)
    n_days: int,
) -> List[float]:
    """Roda a janela deslizante e retorna preços de fechamento previstos."""
    window_size = ohlcv.shape[0]
    n_features  = ohlcv.shape[1]

    # Normaliza cada coluna com seu scaler próprio
    scaled_cols = [
        FEATURE_SCALERS[i].transform(ohlcv[:, i].reshape(-1, 1)).flatten()
        for i in range(n_features)
    ]
    window = np.column_stack(scaled_cols).astype(np.float32)  # (window_size, n_features)

    preds: List[float] = []
    for _ in range(n_days):
        x = torch.tensor(
            window.reshape(1, window_size, n_features), dtype=torch.float32, device=DEVICE
        )
        with torch.no_grad():
            pred_scaled = float(MODEL(x).cpu().numpy()[0, 0])

        pred_price = float(CLOSE_SCALER.inverse_transform([[pred_scaled]])[0, 0])
        preds.append(pred_price)

        # Avança a janela: nova linha = cópia da última, com Close atualizado
        new_row              = window[-1].copy()
        new_row[CLOSE_IDX]   = pred_scaled
        window               = np.vstack([window[1:], [new_row]])

    return preds


# ---------------------------------------------------------------------------
# Inicialização (carregada uma vez na importação do módulo)
# ---------------------------------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL, FEATURE_SCALERS, CLOSE_SCALER, MODEL_CONFIG = load_artifacts()
MODEL.to(DEVICE)
MODEL.eval()

WINDOW_SIZE: int      = int(MODEL_CONFIG.get("window_size", 60))
TICKER: str           = str(MODEL_CONFIG.get("ticker", "DESCONHECIDO"))
FEATURES: List[str]   = MODEL_CONFIG.get("features", ["Open", "High", "Low", "Close"])
CLOSE_IDX: int        = int(MODEL_CONFIG.get("close_idx", FEATURES.index("Close")))

# ---------------------------------------------------------------------------
# Aplicação FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stock LSTM Forecaster — Multivariado",
    description="Previsão de fechamento de ações usando LSTM com features OHLC/OHLCV.",
    version="0.1.0",
)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start    = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", summary="Status do modelo")
def health():
    return {
        "status":      "ok",
        "model":       MODEL_PATH.name,
        "ticker":      TICKER,
        "features":    FEATURES,
        "n_features":  len(FEATURES),
        "window_size": WINDOW_SIZE,
        "close_idx":   CLOSE_IDX,
        "hidden_1":    MODEL_CONFIG.get("hidden_1"),
        "hidden_2":    MODEL_CONFIG.get("hidden_2"),
        "dropout":     MODEL_CONFIG.get("dropout"),
        "val_loss":    MODEL_CONFIG.get("val_loss"),
        "tuned":       MODEL_CONFIG.get("tuned", False),
        "model_type":  MODEL_CONFIG.get("model_type", "multivariate"),
    }


@app.post("/predict", response_model=PredictResponse, summary="Prevê preços de fechamento")
def predict(payload: PredictRequest):
    ticker = payload.ticker or TICKER
    if not ticker or ticker == "DESCONHECIDO":
        raise HTTPException(
            status_code=400,
            detail="Forneça 'ticker' na requisição ou retreine o modelo com um ticker válido.",
        )

    try:
        ohlcv       = fetch_latest_ohlcv(ticker, WINDOW_SIZE, FEATURES)
        predictions = forecast(ohlcv, payload.n_days)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao prever: {exc}") from exc

    return PredictResponse(
        ticker=ticker,
        features=FEATURES,
        predictions=predictions,
        n_days=payload.n_days,
    )


@app.get("/metrics", summary="Métricas Prometheus", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

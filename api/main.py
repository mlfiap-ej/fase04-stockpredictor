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
from pydantic import BaseModel, Field, conlist
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "lstm_model.pt"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
CONFIG_PATH = MODELS_DIR / "model_config.json"

REQUEST_COUNT = Counter("api_requests_total", "Total de requisicoes", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Latencia das requisicoes", ["endpoint"])

class LSTMRegressor(torch.nn.Module):
    def __init__(self, window_size: int, hidden_1: int = 64, hidden_2: int = 32, dropout: float = 0.2):
        super().__init__()
        self.lstm1 = torch.nn.LSTM(input_size=1, hidden_size=hidden_1, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_1, hidden_size=hidden_2, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_2, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        return self.fc2(out)


def load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Artefatos do modelo nao encontrados. Rode o treinamento primeiro.")

    if CONFIG_PATH.exists():
        with CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = json.load(f)
    else:
        config = {"window_size": 60, "ticker": "DESCONHECIDO", "hidden_1": 64, "hidden_2": 32, "dropout": 0.2}

    model = LSTMRegressor(
        window_size=int(config.get("window_size", 60)),
        hidden_1=int(config.get("hidden_1", 64)),
        hidden_2=int(config.get("hidden_2", 32)),
        dropout=float(config.get("dropout", 0.2)),
    )
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler, config


class PredictRequest(BaseModel):
    prices: Optional[conlist(float, min_length=1)] = Field(
        None, description="Historico de precos de fechamento. Se omitido, usa os ultimos dados disponiveis do ticker."
    )
    n_days: int = Field(1, ge=1, le=30, description="Quantidade de dias futuros a prever.")


class PredictResponse(BaseModel):
    predictions: List[float]
    ticker: str


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, scaler, model_config = load_artifacts()
model.to(device)
model.eval()
WINDOW_SIZE = int(model_config.get("window_size", 60))
TICKER = str(model_config.get("ticker", "DESCONHECIDO"))

app = FastAPI(title="Stock LSTM Forecaster", version="0.1.0")


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    REQUEST_COUNT.labels(endpoint=request.url.path, status=str(response.status_code)).inc()
    return response


def forecast(prices: List[float], n_days: int) -> List[float]:
    arr = np.array(prices, dtype=float)
    if arr.shape[0] < WINDOW_SIZE:
        raise HTTPException(status_code=400, detail=f"Envie pelo menos {WINDOW_SIZE} precos de fechamento.")

    scaled = scaler.transform(arr.reshape(-1, 1))
    window = scaled[-WINDOW_SIZE:]
    preds: List[float] = []

    for _ in range(n_days):
        x_input = torch.tensor(window.reshape(1, WINDOW_SIZE, 1), dtype=torch.float32, device=device)
        with torch.no_grad():
            pred_scaled = model(x_input).cpu().numpy()[0, 0]
        window = np.vstack([window[1:], [pred_scaled]])
        pred_value = scaler.inverse_transform([[pred_scaled]])[0, 0]
        preds.append(float(pred_value))

    return preds


def fetch_latest_prices(ticker: str, window_size: int) -> List[float]:
    data = yf.download(ticker, period="90d", interval="1d", progress=False)
    if data.empty or "Close" not in data:
        raise HTTPException(status_code=502, detail="Nao foi possivel obter precos recentes para o ticker informado.")
    closes_obj = data["Close"]
    # Se vier DataFrame (ex.: multi-ticker), reduz para 1 coluna; se Serie, apenas espreme
    if isinstance(closes_obj, pd.DataFrame):
        closes_obj = closes_obj.iloc[:, 0]
    else:
        closes_obj = closes_obj.squeeze()
    closes = pd.Series(closes_obj).dropna().tail(window_size).tolist()
    if len(closes) < window_size:
        raise HTTPException(
            status_code=400,
            detail=f"Dados insuficientes: preciso de pelo menos {window_size} fechamentos recentes para este ticker.",
        )
    return closes


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH.name, "ticker": TICKER, "window_size": WINDOW_SIZE}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        price_series = payload.prices
        if price_series is None:
            price_series = fetch_latest_prices(TICKER, WINDOW_SIZE)
        predictions = forecast(price_series, payload.n_days)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro ao prever: {exc}") from exc
    return PredictResponse(predictions=predictions, ticker=TICKER)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

"""
train_multi.py — versão multivariada do pipeline de treino.

Diferença em relação a train.py:
  - Usa 4 features de entrada por timestep: Open, High, Low, Close.
  - Cada feature é normalizada separadamente (um MinMaxScaler por coluna).
  - O alvo (y) continua sendo o fechamento (Close) do próximo dia.
  - O scaler do Close é salvo separadamente para a API poder reverter a escala.
  - O modelo salva os artefatos em models_multi/ para não sobrescrever o modelo univariado.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

FEATURES: List[str] = ["Open", "High", "Low", "Close"]
CLOSE_IDX: int = FEATURES.index("Close")  # índice da coluna alvo dentro de FEATURES


# ---------------------------------------------------------------------------
# Reproducibilidade
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Coleta de dados
# ---------------------------------------------------------------------------

def fetch_ohlcv(ticker: str, period: str) -> pd.DataFrame:
    print(f"Baixando dados OHLCV de {ticker} para o período {period}...")
    raw = yf.download(ticker, period=period, interval="1d", progress=False)
    if raw.empty:
        raise ValueError(f"Nenhum dado encontrado para {ticker} no período {period}.")

    # Achata MultiIndex de colunas que o yfinance pode retornar
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    missing = [f for f in FEATURES if f not in raw.columns]
    if missing:
        raise ValueError(f"Colunas ausentes no download: {missing}. Disponíveis: {list(raw.columns)}")

    df = raw[FEATURES].dropna()
    print(
        f"Recebemos {len(df)} registros válidos com as colunas: {', '.join(FEATURES)}."
    )
    if len(df) < 80:
        raise ValueError("Poucos pontos para treinar: forneça um período maior.")
    return df


# ---------------------------------------------------------------------------
# Preparação de dados
# ---------------------------------------------------------------------------

def prepare_datasets(
    df: pd.DataFrame,
    window_size: int,
    train_split: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[MinMaxScaler], MinMaxScaler]:
    """
    Retorna:
      X_train, y_train, X_val, y_val — arrays prontos para PyTorch
      feature_scalers                — lista de scalers, um por feature (mesmo índice de FEATURES)
      close_scaler                   — scaler exclusivo do Close, para inverter as previsões
    """
    print(
        f"Preparando dados multivariados: janela={window_size} dias, "
        f"divisão treino/validação = {int(train_split*100)}/{int((1-train_split)*100)}%."
    )

    values = df.values  # shape (N, n_features)
    n_features = values.shape[1]
    split_idx = int(len(values) * train_split)
    if split_idx <= window_size:
        raise ValueError("Janela muito grande para o volume de dados.")

    # Um scaler independente por coluna, ajustado apenas no trecho de treino
    feature_scalers: List[MinMaxScaler] = []
    scaled_cols = []
    for i in range(n_features):
        sc = MinMaxScaler()
        sc.fit(values[:split_idx, i].reshape(-1, 1))
        scaled_cols.append(sc.transform(values[:, i].reshape(-1, 1)).flatten())
        feature_scalers.append(sc)

    scaled = np.column_stack(scaled_cols)  # shape (N, n_features)
    close_scaler = feature_scalers[CLOSE_IDX]

    print(f"Normalização aplicada a cada uma das {n_features} colunas separadamente.")

    # Cria sequências
    # X: (samples, window_size, n_features)  y: (samples, 1) — próximo Close normalizado
    X, y = [], []
    for i in range(window_size, len(scaled)):
        X.append(scaled[i - window_size : i])          # janela de features
        y.append(scaled[i, CLOSE_IDX])                 # somente o Close do próximo dia

    X = np.array(X, dtype=np.float32)    # (N, window, n_features)
    y = np.array(y, dtype=np.float32)    # (N,)

    split_seq = split_idx - window_size
    X_train, y_train = X[:split_seq], y[:split_seq]
    X_val, y_val = X[split_seq:], y[split_seq:]

    print(f"Sequências geradas: {len(X_train)} para treino e {len(X_val)} para validação.")
    return X_train, y_train, X_val, y_val, feature_scalers, close_scaler


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class MultiLSTMRegressor(torch.nn.Module):
    """
    LSTM duplo que aceita múltiplas features de entrada por timestep.
    input_size deve ser igual ao número de features (ex.: 4 para OHLC).
    """

    def __init__(
        self,
        n_features: int,
        hidden_1: int = 64,
        hidden_2: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.lstm1 = torch.nn.LSTM(input_size=n_features, hidden_size=hidden_1, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.lstm2 = torch.nn.LSTM(input_size=hidden_1, hidden_size=hidden_2, batch_first=True)
        self.fc1 = torch.nn.Linear(hidden_2, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        self.n_features = n_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, window_size, n_features)
        out, _ = self.lstm1(x)
        out = self.dropout(out)
        out, _ = self.lstm2(out)
        out = out[:, -1, :]          # último timestep
        out = self.relu(self.fc1(out))
        return self.fc2(out)         # (batch, 1)


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    print(f"Montando lotes de tamanho {batch_size} para treinar e validar.")
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int = 5,
) -> Tuple[Dict[str, list], Dict[str, float]]:
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        print(f"Iniciando época {epoch}/{epochs}...")
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                val_losses.append(criterion(model(X_batch), y_batch).item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"Época {epoch:03d}: erro no treino = {train_loss:.6f}, erro na validação = {val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping ativado.")
                break

    model.load_state_dict(best_state)
    return history, {"val_loss": best_val}


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def evaluate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    close_scaler: MinMaxScaler,
    device: torch.device,
) -> Dict[str, float]:
    """
    Métricas calculadas sobre os preços reais (escala original):

    - MAE  (Erro Médio Absoluto)     — erro médio em unidade de preço.
    - RMSE (Raiz do Erro Quadrático) — penaliza erros grandes.
    - MAPE (Erro Percentual Médio)   — erro em %; abaixo de 5% é bom.
    - R²   (Coeficiente de Det.)     — o quanto o modelo explica a variação (0 a 1).
    - Acurácia Direcional            — % de acertos de subida/descida.
    """
    model.eval()
    preds_scaled, y_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch.to(device)).cpu().numpy().flatten()
            preds_scaled.extend(preds)
            y_scaled.extend(y_batch.numpy().flatten())

    y_true = close_scaler.inverse_transform(np.array(y_scaled).reshape(-1, 1)).flatten()
    y_pred = close_scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100)
    r2 = float(r2_score(y_true, y_pred))
    if len(y_true) > 1:
        dir_acc = float(np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred))) * 100)
    else:
        dir_acc = float("nan")

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2, "directional_accuracy": dir_acc}


def print_metrics(metrics: Dict[str, float]) -> None:
    explanations = {
        "mae":                  ("MAE  (Erro Médio Absoluto)",      "Em média, o modelo errou {:.4f} unidades de preço por dia. Quanto menor, melhor."),
        "rmse":                 ("RMSE (Raiz do Erro Quadrático)",   "Erros grandes pesam mais aqui: {:.4f}. Valores próximos ao MAE indicam erros consistentes."),
        "mape":                 ("MAPE (Erro Percentual Médio)",     "O modelo errou em média {:.2f}% do preço real. Abaixo de 5% é considerado bom."),
        "r2":                   ("R²   (Coeficiente de Det.)",       "O modelo explica {:.4f} da variação dos preços (máximo = 1.0)."),
        "directional_accuracy": ("DIR  (Acurácia Direcional)",       "O modelo acertou a direção (subida/queda) em {:.2f}% dos dias. Acima de 50% é melhor que aleatório."),
    }
    print("\n" + "=" * 60)
    print("RESULTADOS DA AVALIAÇÃO DO MODELO (MULTIVARIADO)")
    print("=" * 60)
    for key, value in metrics.items():
        label, explanation = explanations[key]
        print(f"\n{label}")
        print(f"  Valor : {value:.4f}")
        print(f"  Interpretação: {explanation.format(value)}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Persistência
# ---------------------------------------------------------------------------

def save_artifacts(
    model: torch.nn.Module,
    feature_scalers: List[MinMaxScaler],
    close_scaler: MinMaxScaler,
    metrics: Dict[str, float],
    config: dict,
    model_dir: Path,
    artifacts_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), model_dir / "lstm_model.pt")
    joblib.dump(feature_scalers, model_dir / "feature_scalers.pkl")
    joblib.dump(close_scaler, model_dir / "close_scaler.pkl")

    with (model_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    with (artifacts_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Modelo salvo em {model_dir / 'lstm_model.pt'}")
    print(f"Scalers salvos em {model_dir}/")
    print(f"Métricas salvas em {artifacts_dir / 'metrics.json'}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Treina LSTM multivariada (OHLCV) para prever fechamento de ações."
    )
    parser.add_argument("--ticker",        type=str,   default="AAPL")
    parser.add_argument("--period",        type=str,   default="5y")
    parser.add_argument("--window-size",   type=int,   default=60)
    parser.add_argument("--epochs",        type=int,   default=30)
    parser.add_argument("--batch-size",    type=int,   default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--patience",      type=int,   default=5)
    parser.add_argument("--tune",          action="store_true", help="Otimização de hiperparâmetros via Optuna.")
    parser.add_argument("--trials",        type=int,   default=10)
    parser.add_argument("--seed",          type=int,   default=42)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}.")
    print(f"Features de entrada: {', '.join(FEATURES)}. Alvo: Close do próximo dia.")

    try:
        df = fetch_ohlcv(args.ticker, args.period)
    except Exception as exc:
        print(f"Erro ao coletar dados: {exc}")
        sys.exit(1)

    try:
        X_train, y_train, X_val, y_val, feature_scalers, close_scaler = prepare_datasets(
            df, window_size=args.window_size
        )
    except Exception as exc:
        print(f"Erro ao preparar dados: {exc}")
        sys.exit(1)

    n_features = len(FEATURES)

    best_params = {
        "hidden_1": 64,
        "hidden_2": 32,
        "dropout": 0.2,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "patience": args.patience,
    }

    if args.tune:
        print(f"Buscando melhores hiperparâmetros via Optuna em {args.trials} tentativas...")
        print(
            "Parâmetros sendo testados:",
            "hidden_1 (32–128), hidden_2 (16–64),",
            "dropout (0.0–0.5), learning_rate (log: 1e-4 a 5e-3),",
            "batch_size (16, 32, 64), patience (3–8).",
        )

        def objective(trial: optuna.Trial) -> float:
            hidden_1 = trial.suggest_int("hidden_1", 32, 128, step=32)
            hidden_2 = trial.suggest_int("hidden_2", 16, 64, step=16)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
            patience = trial.suggest_int("patience", 3, 8)

            tl, vl = make_dataloaders(X_train, y_train, X_val, y_val, batch_size)
            m = MultiLSTMRegressor(n_features=n_features, hidden_1=hidden_1, hidden_2=hidden_2, dropout=dropout).to(device)
            _, summary = train_model(m, tl, vl, device, args.epochs, lr, patience)
            return summary["val_loss"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.trials)
        best_params.update(study.best_params)
        print(
            f"Melhor trial {study.best_trial.number}: val_loss={study.best_value:.6f}",
            f"| params={study.best_params}",
        )

    train_loader, val_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, int(best_params["batch_size"])
    )
    model = MultiLSTMRegressor(
        n_features=n_features,
        hidden_1=int(best_params["hidden_1"]),
        hidden_2=int(best_params["hidden_2"]),
        dropout=float(best_params["dropout"]),
    ).to(device)

    print(
        f"Treinando modelo final: hidden1={best_params['hidden_1']}, "
        f"hidden2={best_params['hidden_2']}, dropout={best_params['dropout']:.2f}, "
        f"lr={best_params['learning_rate']}, batch={best_params['batch_size']}, "
        f"patience={best_params['patience']}."
    )

    history, training_summary = train_model(
        model, train_loader, val_loader, device,
        args.epochs, float(best_params["learning_rate"]), int(best_params["patience"])
    )

    metrics = evaluate(model, val_loader, close_scaler, device)
    print_metrics(metrics)

    config = {
        "ticker": args.ticker,
        "period": args.period,
        "window_size": args.window_size,
        "features": FEATURES,
        "close_idx": CLOSE_IDX,
        "n_features": n_features,
        "epochs": args.epochs,
        "batch_size": int(best_params["batch_size"]),
        "learning_rate": float(best_params["learning_rate"]),
        "patience": int(best_params["patience"]),
        "seed": args.seed,
        "hidden_1": int(best_params["hidden_1"]),
        "hidden_2": int(best_params["hidden_2"]),
        "dropout": float(best_params["dropout"]),
        "val_loss": float(training_summary.get("val_loss", 0.0)),
        "train_loss_last": float(history["train_loss"][-1]) if history["train_loss"] else 0.0,
        "tuned": args.tune,
        "trials": args.trials if args.tune else 0,
        "model_type": "multivariate",
    }

    save_artifacts(
        model=model,
        feature_scalers=feature_scalers,
        close_scaler=close_scaler,
        metrics=metrics,
        config=config,
        model_dir=Path("models_multi"),
        artifacts_dir=Path("artifacts_multi"),
    )
    print("Pronto! Artefatos salvos em models_multi/ e artifacts_multi/.")


if __name__ == "__main__":
    main()

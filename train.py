import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import yfinance as yf
import optuna
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def fetch_prices(ticker: str, period: str) -> pd.DataFrame:
    print(f"Baixando preços de {ticker} para o período {period} (pode levar alguns segundos)...")
    data = yf.download(ticker, period=period, interval="1d", progress=False)
    if data.empty or "Close" not in data:
        raise ValueError(f"Nenhum dado encontrado para {ticker} no periodo {period}.")
    closes = data[["Close"]].dropna().rename(columns={"Close": "close"})
    print(f"Recebemos {len(closes)} fechamentos válidos depois da limpeza.")
    if len(closes) < 80:
        raise ValueError("Poucos pontos para treinar: forneca um periodo maior.")
    return closes


def create_sequences(values: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(window_size, len(values)):
        X.append(values[idx - window_size : idx])
        y.append(values[idx])
    return np.array(X), np.array(y)


def prepare_datasets(
    series: pd.Series, window_size: int, train_split: float = 0.8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    print(
        f"Preparando dados: cada exemplo usará {window_size} dias anteriores. "
        f"Divisão treino/validação = {int(train_split*100)}/{int((1-train_split)*100)}%."
    )
    values = series.values.reshape(-1, 1)
    split_idx = int(len(values) * train_split)
    if split_idx <= window_size:
        raise ValueError("Janela muito grande para o volume de dados.")

    scaler = MinMaxScaler()
    scaler.fit(values[:split_idx])
    scaled = scaler.transform(values)
    print("Normalizando valores com base no conjunto de treino.")

    X, y = create_sequences(scaled, window_size)
    split_sequences = split_idx - window_size
    X_train, y_train = X[:split_sequences], y[:split_sequences]
    X_val, y_val = X[split_sequences:], y[split_sequences:]

    print(
        f"Sequências prontas: {len(X_train)} para treino e {len(X_val)} para validação."
    )

    return X_train, y_train, X_val, y_val, scaler


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


def train_model(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    patience: int = 5,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    wait = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        print(f"Iniciando época {epoch}/{epochs}...")
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(
            f"Época {epoch:03d}: erro no treino = {train_loss:.6f}, erro na validação = {val_loss:.6f}"
        )

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


def evaluate(
    model: torch.nn.Module,
    val_loader: torch.utils.data.DataLoader,
    scaler: MinMaxScaler,
    device: torch.device,
) -> Dict[str, float]:
    """
    Avalia o modelo no conjunto de validação e retorna um dicionário com as métricas:

    - MAE  (Mean Absolute Error): erro médio absoluto em reais/dólares.
              Quanto menor, mais perto o modelo errou na média.
    - RMSE (Root Mean Square Error): penaliza mais os erros grandes.
              Útil para detectar previsões muito fora do esperado.
    - MAPE (Mean Absolute Percentage Error): erro percentual médio.
              Fácil de interpretar: ex. 2.5 significa ~2.5% de erro médio.
    - R²   (Coeficiente de determinação): quanto da variação nos preços
              reais o modelo consegue explicar. Varia de 0 a 1; quanto
              mais próximo de 1, melhor o ajuste.
    - Acurácia direcional: % de vezes que o modelo acertou se o preço
              subiu ou desceu em relação ao dia anterior.
    """
    model.eval()
    preds_scaled, y_scaled = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            preds = model(X_batch).cpu().numpy()
            preds_scaled.extend(preds.flatten())
            y_scaled.extend(y_batch.numpy().flatten())

    y_true = scaler.inverse_transform(np.array(y_scaled).reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(mean_squared_error(y_true, y_pred) ** 0.5)
    denom = np.clip(np.abs(y_true), 1e-8, None)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)
    r2 = float(r2_score(y_true, y_pred))

    # Acurácia direcional: compara se a direção (sobe/desce) foi prevista corretamente
    if len(y_true) > 1:
        true_dir = np.sign(np.diff(y_true))
        pred_dir = np.sign(np.diff(y_pred))
        directional_accuracy = float(np.mean(true_dir == pred_dir) * 100)
    else:
        directional_accuracy = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "r2": r2,
        "directional_accuracy": directional_accuracy,
    }


def make_dataloaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    print(f"Montando lotes de tamanho {batch_size} para treinar e validar.")
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
    )
    return train_loader, val_loader


def save_artifacts(
    model: torch.nn.Module,
    scaler: MinMaxScaler,
    metrics: Dict[str, float],
    config: Dict[str, str],
    model_dir: Path,
    artifacts_dir: Path,
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "lstm_model.pt"
    scaler_path = model_dir / "scaler.pkl"
    config_path = model_dir / "model_config.json"
    metrics_path = artifacts_dir / "metrics.json"

    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Modelo salvo em {model_path}")
    print(f"Scaler salvo em {scaler_path}")
    print(f"Metricas salvas em {metrics_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina um modelo LSTM (PyTorch) para prever fechamento de acoes.")
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker da acao no Yahoo Finance.")
    parser.add_argument("--period", type=str, default="5y", help="Periodo do historico (ex: 1y, 2y, 5y, max).")
    parser.add_argument("--window-size", type=int, default=60, help="Tamanho da janela (dias) usada para sequencias.")
    parser.add_argument("--epochs", type=int, default=30, help="Numero maximo de epocas de treino.")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamanho do batch.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Taxa de aprendizado do otimizador.")
    parser.add_argument("--patience", type=int, default=5, help="Paciencia para early stopping.")
    parser.add_argument("--tune", action="store_true", help="Ativa otimizacao de hiperparametros via Optuna.")
    parser.add_argument("--trials", type=int, default=10, help="Numero de trials Optuna (usado com --tune).")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reproducibilidade.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    print(f"Ticker em uso: {args.ticker}, periodo: {args.period}, janela: {args.window_size} dias.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}. Se houver GPU disponível, ela será usada automaticamente.")

    try:
        closes = fetch_prices(args.ticker, args.period)
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao coletar dados: {exc}")
        sys.exit(1)

    try:
        X_train, y_train, X_val, y_val, scaler = prepare_datasets(
            closes["close"], window_size=args.window_size
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Erro ao preparar dados: {exc}")
        sys.exit(1)

    best_params = {
        "hidden_1": 64,
        "hidden_2": 32,
        "dropout": 0.2,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "patience": args.patience,
    }

    if args.tune:
        print(f"Buscando automaticamente os melhores hiperparâmetros (Optuna) em {args.trials} tentativas...")
        print(
            "O Optuna vai testar combinações de:",
            "camada 1 LSTM (32 a 128 neurônios, passo 32),",
            "camada 2 LSTM (16 a 64 neurônios, passo 16),",
            "dropout (0.00 a 0.50),",
            "taxa de aprendizado (1e-4 a 5e-3 em escala log),",
            "tamanho do lote (8, 16, 32 ou 64),",
            "paciência para early stopping (3 a 8 épocas).",
        )

        def objective(trial: optuna.Trial) -> float:
            hidden_1 = trial.suggest_int("hidden_1", 32, 256, step=32)
            hidden_2 = trial.suggest_int("hidden_2", 16, 128, step=16)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
            batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
            patience = trial.suggest_int("patience", 3, 8)

            train_loader, val_loader = make_dataloaders(
                X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, batch_size=batch_size
            )

            model = LSTMRegressor(
                window_size=args.window_size,
                hidden_1=hidden_1,
                hidden_2=hidden_2,
                dropout=dropout,
            ).to(device)

            _, summary = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                epochs=args.epochs,
                lr=learning_rate,
                patience=patience,
            )

            return summary["val_loss"]

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=args.trials)
        best_params.update(study.best_params)
        print(
            "Melhor combinação encontrada:",
            f"tentativa {study.best_trial.number}, erro de validação {study.best_value:.6f},",
            f"parametros {study.best_params}"
        )
        print(
            "Isso significa que o modelo final usará:",
            f"{best_params['hidden_1']} neurônios na primeira LSTM,",
            f"{best_params['hidden_2']} neurônios na segunda LSTM,",
            f"dropout {best_params['dropout']:.2f},",
            f"taxa de aprendizado {best_params['learning_rate']},",
            f"lotes de {best_params['batch_size']} amostras,",
            f"e early stopping com paciência de {best_params['patience']} épocas."
        )

    train_loader, val_loader = make_dataloaders(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        batch_size=int(best_params["batch_size"]),
    )

    model = LSTMRegressor(
        window_size=args.window_size,
        hidden_1=int(best_params["hidden_1"]),
        hidden_2=int(best_params["hidden_2"]),
        dropout=float(best_params["dropout"]),
    ).to(device)

    print(
        "Treinando o modelo final com:",
        f"camadas LSTM {best_params['hidden_1']} e {best_params['hidden_2']} neurônios,",
        f"dropout {best_params['dropout']:.2f},",
        f"taxa de aprendizado {best_params['learning_rate']},",
        f"lotes de {best_params['batch_size']} e paciência de {best_params['patience']} épocas."
    )

    history, training_summary = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=float(best_params["learning_rate"]),
        patience=int(best_params["patience"]),
    )

    metrics = evaluate(model, val_loader, scaler, device)

    metric_explanations = {
        "mae": (
            "MAE  (Erro Médio Absoluto)",
            "Em média, o modelo errou {:.4f} unidades de preço por dia."
        ),
        "rmse": (
            "RMSE (Raiz do Erro Quadrático Médio)",
            "Erros grandes pesam mais aqui: {:.4f}."
        ),
        "mape": (
            "MAPE (Erro Percentual Médio)",
            "O modelo errou em média {:.2f}% do preço real. Abaixo de 5% é considerado bom para séries financeiras."
        ),
        "r2": (
            "R²   (Coeficiente de Determinação)",
            "O modelo explica {:.4f} da variação dos preços (máximo = 1.0). Quanto mais próximo de 1, melhor."
        ),
        "directional_accuracy": (
            "DIR  (Acurácia Direcional)",
            "O modelo acertou a direção (subida/queda) em {:.2f}% dos dias. Acima de 50% é melhor que aleatório."
        ),
    }

    print("\n" + "=" * 60)
    print("RESULTADOS DA AVALIAÇÃO DO MODELO")
    print("=" * 60)
    for key, value in metrics.items():
        label, explanation = metric_explanations[key]
        print(f"\n{label}")
        print(f"  Valor : {value:.4f}")
        print(f"  Interpretação: {explanation.format(value)}")
    print("=" * 60 + "\n")

    config = {
        "ticker": args.ticker,
        "period": args.period,
        "window_size": args.window_size,
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
    }

    save_artifacts(
        model=model,
        scaler=scaler,
        metrics=metrics,
        config=config,
        model_dir=Path("models"),
        artifacts_dir=Path("artifacts"),
    )
    print("Arquivos salvos em models/ (modelo e scaler) e artifacts/ (métricas). Pronto para servir a API.")


if __name__ == "__main__":
    main()

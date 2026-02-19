# Pipeline LSTM para Previsão de Fechamento de Ações

Este projeto entrega uma pipeline completa: coleta de dados (Yahoo Finance), treinamento de um modelo LSTM, armazenamento de artefatos, e serviçi via API FastAPI com instrumentação para monitoramento Prometheus.

## Requisitos

- Python 3.12+

- Instale dependências, de acordo com o SO usado. No Linux e macOS: 
```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

- PyTorch: o pip instalará a build CPU por padrao; se quiser GPU, ajuste conforme a [documentacao oficial](https://pytorch.org/).

## Treinamento

Existem dois scripts de treino: 

- `train.py`, para treinamento uniescalar
- `train_multi.py` para treinamento multiescalar

As configurações de treinamento são as mesmas, mudando apenas o nome do script. 

```bash
python train.py --ticker WEGE3.SA \
    --period 5y --window-size 10 --epochs 5 \
    --batch-size 32
```

### Treinamento com otimização

Para otimizar hiperparâmetros com Optuna (batch size, hidden sizes, dropout, learning rate, patience), use os parâmetros `tune` e `trials`:

```bash
python train.py --ticker WEGE3.SA \
    --period 5y --window-size 10 --epochs 5 \
    --tune --trials 5
```

Saidas geradas:

- Versão *uni*:

    - `models/lstm_model.pt`: pesos treinados (state_dict do PyTorch).
    - `models/scaler.pkl`: scaler usado no treino.
    - `models/model_config.json`: configuracoes do modelo (ticker, janela, etc.).
    - `artifacts/metrics.json`: metricas finais (MAE, RMSE, MAPE).

- Versão *multi*

    - `models_multi/lstm_model.pt`: pesos treinados (state_dict do PyTorch).
    - `models_multi/feature_scaler.pkl`: scaler para as features usadas no treino.
    - `models_multi/close_scaler.pkl`: scaler para o target usado no treino.
    - `models_multi/model_config.json`: configuracoes do modelo (ticker, janela, etc.).
    - `artifacts_multi/metrics.json`: metricas finais (MAE, RMSE, MAPE).

## Consumo 

### Servindo a API

1) Certifique-se de que os artefatos do modelo existem (rode o treinamento primeiro).

2) Selecione a API desejada, para a versão uniescalar (`main`) ou multiescalar (`main_multi`)

3) Inicie a API:

```bash 
# para multi, mude para main_multi abaixo
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload    
```

#### Endpoints
- `GET /health`: status basico e nome do modelo carregado.
- `GET /metrics`: metrica Prometheus pronta para scrape.
- `POST /predict`: previsao de próximos fechamentos.
    - Body application/json com parametros 
        - n_days: quantidade de dias
        - prices: lista de últimas cotações, para referência. *(obs: quando não preenchido, obtem automaticamente os últimos dias de acordo com o yfinance)*


#### Exemplo de requisicao

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"prices": [150.2, 151.0, 149.8, 150.5, 152.1, 153.2, 152.7, 154.0, 153.5, 154.8, 155.1, 154.9, 155.5, 156.0, 156.4, 155.8, 156.2, 157.0, 158.3, 159.1, 158.7, 159.5, 160.2, 160.8, 161.5, 162.0, 162.7, 163.4, 164.0, 164.8, 165.3, 165.9, 166.2, 166.7, 167.0, 167.4, 167.9, 168.3, 168.7, 169.0, 169.4, 169.8, 170.2, 170.6, 171.0, 171.4, 171.8, 172.2, 172.6, 173.0, 173.4, 173.8, 174.2, 174.6, 175.0, 175.4, 175.8, 176.2], "n_days": 3}'
```

> Para testes, há o arquivo `testes.http` que pode ser usado. 

Retorno esperado:
```json
{
    "predictions": [176.5, 176.9, 177.3], 
    "ticker": "WEGE3.SA"
}
```

### CLI

Existem uma CLI que permite o consumo imediato do modelo pós treinamento. 

Exemplos de uso: 

```bash
# Executa o modelo uniescalar, com predição para 5 dias.
python cli.py predict --n-days 5
```

Exemplo de saída: 

```text
Buscando os últimos 10 fechamentos de WEGE3.SA...
Prevendo 5 dia(s) com base em: últimos 10 fechamentos de WEGE3.SA.

=== Previsão de fechamento [univariado] — WEGE3.SA ===
  Dia + 1 : 52.5016
  Dia + 2 : 52.4672
  Dia + 3 : 52.5819
  Dia + 4 : 52.6455
  Dia + 5 : 52.7145
```

Outros exemplos: 

```bash
# Executa o modelo multiescalar, com predição para 5 dias.
python cli.py predict --model multi --n-days 5
```

```bash
# Informações simples sobre o modelo
python cli.py health
```


Para mais informações, verifique o help:

```bash
python cli.py -h 
```

```bash
python cli.py predict -h  
```

## Build 

É possivel fazer o build de iamgens docker para disponibilizar o modelo treinado, baseado em um ticker. O treinamento é efetuado durante o build da imagem, permitindo que, caso necessário, seja feito um build diario autocontido, disponibilizando um modelo treinado os dados atualizados.

No build, o parâmetro TICKER é opcional, com o ticker WEGE3.SA sendo o padrão.

Para fazer o build:

```bash 
# uniescalar, com TICKER definido
docker build -f Dockerfile_uni --build-arg TICKER=PETR4.SA -t trained-model-uni:1 .
```

```bash 
# multiescalar, com TICKER padrão
docker build -f Dockerfile_multi -t local-train-multi:1 .
```

E para criar o container baseado na imagem, use o comando abaixo, mudando o parametro da primeira porta para a porta desejada, e o nome da imagem desejada.

```bash
docker run -p 8009:8000 final-train-uni:1
```

## Monitoramento

- `GET /metrics` expõe contadores e histogramas de latência para Prometheus.

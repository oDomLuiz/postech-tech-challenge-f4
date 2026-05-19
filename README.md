# Tech Challenge Fase 4 - Predicao de Precos de Acoes com LSTM

Projeto desenvolvido para o Tech Challenge Fase 4, com foco em series temporais, Deep Learning com LSTM e disponibilizacao de previsoes por uma API RESTful em FastAPI.

O projeto coleta dados historicos de acoes, prepara janelas temporais para treinamento, treina um modelo LSTM, avalia o desempenho com metricas de regressao, salva os artefatos de inferencia e expoe endpoints para predicao.

> Este projeto tem finalidade academica. As previsoes geradas nao representam recomendacao de investimento.

## Ativo utilizado

- Empresa: Apple Inc.
- Ticker: `AAPL`
- Variavel-alvo: `Close`
- Janela temporal padrao: 60 dias
- Fonte principal: Yahoo Finance Chart API
- Fontes alternativas: `yfinance` e Stooq

A coleta foi implementada com fallback porque o `yfinance` pode falhar em alguns ambientes com erros como `YFTzMissingError` ou falha ao resolver metadados/timezone do ticker.

## Tecnologias

- Python 3.11
- FastAPI
- Uvicorn
- TensorFlow/Keras
- pandas
- numpy
- scikit-learn
- yfinance
- joblib
- matplotlib
- pytest
- Docker e Docker Compose

## Estrutura do projeto

```text
.
|-- app/
|   |-- api/
|   |   |-- main.py          # Inicializacao da API FastAPI
|   |   |-- routes.py        # Endpoints da API
|   |   `-- schemas.py       # Schemas Pydantic
|   |-- core/
|   |   |-- config.py        # Configuracoes e caminhos do projeto
|   |   `-- monitoring.py    # Logs e metricas basicas
|   `-- model/
|       |-- inference.py     # Inferencia do modelo
|       `-- loader.py        # Carregamento do modelo e scaler
|-- data/
|   |-- raw/                 # Dados brutos coletados
|   `-- processed/           # Graficos e metricas geradas
|-- models/                  # Modelo, scaler e metadata
|-- notebooks/               # Espaco para analises exploratorias
|-- src/
|   |-- collect_data.py      # Coleta dos dados historicos
|   |-- preprocess.py        # Limpeza, normalizacao e janelas temporais
|   |-- train.py             # Treinamento do modelo LSTM
|   |-- evaluate.py          # Avaliacao do modelo
|   `-- utils.py             # Funcoes auxiliares
|-- tests/
|   |-- test_api.py
|   `-- test_preprocess.py
|-- Dockerfile
|-- docker-compose.yml
|-- requirements.txt
`-- README.md
```

## Como executar localmente

### 1. Criar e ativar o ambiente virtual

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Se o PowerShell bloquear a ativacao do ambiente virtual, execute:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Coletar dados historicos

```bash
python -m src.collect_data --ticker AAPL --start-date 2018-01-01
```

Tambem e possivel informar uma data final:

```bash
python -m src.collect_data --ticker AAPL --start-date 2018-01-01 --end-date 2024-07-20
```

O arquivo gerado sera salvo em:

```text
data/raw/AAPL.csv
```

### 4. Treinar o modelo

```bash
python -m src.train --ticker AAPL --window-size 60 --epochs 50 --batch-size 32
```

Para um teste mais rapido durante desenvolvimento:

```bash
python -m src.train --ticker AAPL --epochs 10
```

Ao final do treinamento, estes arquivos sao criados em `models/`:

```text
models/lstm_model.keras
models/scaler.pkl
models/metadata.json
```

### 5. Avaliar o modelo

```bash
python -m src.evaluate --ticker AAPL
```

A avaliacao calcula:

- MAE
- RMSE
- MAPE

E gera arquivos em `data/processed/`, incluindo:

```text
data/processed/AAPL_metrics.json
data/processed/AAPL_actual_vs_predicted.png
```

### 6. Executar a API

```bash
uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000
```

A documentacao interativa ficara disponivel em:

```text
http://localhost:8000/docs
```

## Endpoints da API

### `GET /`

Retorna informacoes basicas da aplicacao.

Exemplo:

```json
{
  "project": "Tech Challenge Fase 4 - LSTM Stock Prediction",
  "status": "running",
  "model": "LSTM"
}
```

### `GET /health`

Verifica se a API esta ativa e se os artefatos do modelo existem.

Exemplo:

```json
{
  "status": "ok",
  "model_ready": true
}
```

Quando `model_ready` for `false`, execute o treinamento antes de usar os endpoints de predicao.

### `GET /metrics`

Retorna metricas simples coletadas pelo middleware da API.

Exemplo:

```json
{
  "total_requests": 10,
  "total_errors": 0,
  "average_response_time_ms": 12.35
}
```

### `POST /predict`

Recebe uma lista de precos historicos e retorna a previsao do proximo preco de fechamento.

A lista precisa conter pelo menos `window_size` valores. Por padrao, sao 60 valores.

Exemplo de request:

```json
{
  "prices": [
    188.85,
    189.87,
    191.04,
    192.35,
    193.12
  ]
}
```

Na pratica, envie pelo menos 60 precos no array.

Exemplo de response:

```json
{
  "predicted_close_price": 194.251,
  "window_size": 60,
  "model_version": "1.0.0"
}
```

### `POST /predict-by-ticker`

Busca dados recentes automaticamente via `yfinance` e gera uma predicao.

Exemplo de request:

```json
{
  "ticker": "AAPL",
  "period": "6mo"
}
```

Exemplo de response:

```json
{
  "ticker": "AAPL",
  "predicted_close_price": 194.251,
  "window_size": 60,
  "model_version": "1.0.0"
}
```

## Execucao com Docker

### Build e execucao com Docker Compose

```bash
docker compose up --build
```

A API ficara disponivel em:

```text
http://localhost:8000/docs
```

O `docker-compose.yml` monta os diretorios `models/` e `data/` como volumes, permitindo manter os dados e artefatos fora do ciclo de vida do container.

### Build manual

```bash
docker build -t lstm-stock-api .
docker run -p 8000:8000 lstm-stock-api
```

## Testes

Execute:

```bash
pytest
```

Ou, se quiser garantir o uso do Python do ambiente virtual no Windows:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Os testes cobrem:

- endpoints principais da API;
- validacao de resposta quando o modelo ainda nao foi treinado;
- criacao de sequencias temporais;
- limpeza e ordenacao dos dados;
- preparacao dos splits de treino, validacao e teste.

## Pipeline do projeto

1. `src.collect_data` coleta dados historicos e salva CSV em `data/raw/`.
2. `src.preprocess` limpa os dados, normaliza os precos e cria janelas temporais.
3. `src.train` treina uma rede LSTM e salva modelo, scaler e metadata.
4. `src.evaluate` carrega o modelo treinado e calcula metricas no conjunto de teste.
5. `app.api` disponibiliza os endpoints REST para inferencia e monitoramento.

## Modelo

A arquitetura implementada em `src/train.py` usa:

- LSTM com 64 unidades e `return_sequences=True`;
- Dropout de 20%;
- LSTM com 32 unidades;
- Dropout de 20%;
- Dense com 16 unidades e ativacao ReLU;
- Dense final com 1 unidade para regressao.

O modelo usa:

- otimizador `Adam`;
- loss `mean_squared_error`;
- `EarlyStopping`;
- `ModelCheckpoint`.

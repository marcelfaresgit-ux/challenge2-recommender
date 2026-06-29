# Tech Challenge - Sistema de Recomendacao para E-commerce

Projeto end-to-end para recomendacao de produtos com base no comportamento de navegacao dos
usuarios. A solucao cobre clean code, dependencias modernas, DVC, MLflow, Docker, baselines
em Scikit-Learn, rede neural PyTorch e API FastAPI.

## Stack

- Python 3.11+
- PyTorch
- Scikit-Learn
- MLflow
- DVC
- FastAPI
- Docker e Docker Compose
- Pytest e Ruff

## Estrutura

```text
.
|-- data/
|   |-- raw/
|   |-- processed/
|   `-- artifacts/
|-- docs/
|-- scripts/
|-- src/ecommerce_recommender/
|-- tests/
|-- dvc.yaml
|-- Dockerfile
|-- docker-compose.yml
|-- params.yaml
`-- pyproject.toml
```

## Como instalar

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .[dev]
python scripts/validate_env.py
```

No Windows, se `python` abrir a Microsoft Store ou nao for encontrado, use:

```powershell
.\scripts\setup_windows.ps1
```

Depois rode as verificacoes com:

```powershell
.\scripts\run_checks_windows.ps1
```

## Como reproduzir

```bash
python -m ecommerce_recommender.data.generate
python -m ecommerce_recommender.features.prepare
python -m ecommerce_recommender.models.train
python -m ecommerce_recommender.models.evaluate
```

Ou com DVC:

```bash
dvc repro
```

## API

Depois de gerar dados e treinar o modelo:

```bash
python -m uvicorn ecommerce_recommender.api.main:app --reload
```

Exemplo:

```bash
curl -X POST http://127.0.0.1:8000/recommendations ^
  -H "Content-Type: application/json" ^
  -d "{\"user_id\":\"u_00042\",\"top_k\":10}"
```

## Docker

```bash
docker compose up --build
```

- API: `http://localhost:8000/docs`
- MLflow: `http://localhost:5000`

## Deploy no Render

O projeto inclui `render.yaml` e um `Dockerfile` pronto para deploy. O container gera dados,
prepara features, treina o modelo e cria metricas durante o build.

1. Suba o repositorio no GitHub.
2. Acesse `https://dashboard.render.com/`.
3. Clique em `New +` e escolha `Blueprint`.
4. Conecte o repositorio `challenge2-recommender`.
5. Confirme o blueprint e aguarde o build.

Depois do deploy, teste:

```text
https://SEU-SERVICO.onrender.com/health
https://SEU-SERVICO.onrender.com/docs
```

Exemplo de chamada:

```bash
curl -X POST https://SEU-SERVICO.onrender.com/recommendations \
  -H "Content-Type: application/json" \
  -d '{"user_id":"u_00042","top_k":10}'
```

## Entregaveis do enunciado

- Estrutura limpa com `src/`, `tests/`, `data/`, `docs/` e `scripts/`.
- `pyproject.toml` com dependencias prod/dev.
- `.env.example`, `.gitignore` e `.dockerignore`.
- Funcoes curtas, type hints e nomes descritivos.
- Design patterns: Factory, Strategy e Template Method.
- Dockerfile multi-stage e `docker-compose.yml`.
- Pipeline DVC com 4 etapas: generate, prepare, train e evaluate.
- MLflow tracking com parametros, metricas e comparacao.
- Rede neural PyTorch para recomendacao.
- Baseline Scikit-Learn.
- Model Card, resultados e roteiro STAR.

## Dataset

O dataset sintetico gerado possui 24.000 interacoes entre 1.200 usuarios e 600 produtos.
Ele simula eventos de `view`, `cart` e `purchase`, alem de rating implicito.

## Testes e qualidade

```bash
python -m pytest
python -m ruff check .
python -m ruff format --check .
```

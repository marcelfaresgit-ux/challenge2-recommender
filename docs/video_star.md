# Roteiro STAR para Video de 5 Minutos

## Situation

Uma empresa de e-commerce precisa recomendar produtos usando comportamento de navegacao:
visualizacoes, carrinho, compras e notas implicitas.

## Task

Construir um pipeline reproduzivel com codigo limpo, DVC, MLflow, Docker, baseline
Scikit-Learn e rede neural PyTorch.

## Action

Foi criada uma estrutura modular em `src/`, com geracao de dataset, preparacao de features,
treino, avaliacao e API. O pipeline DVC tem quatro etapas e o MLflow registra parametros,
metricas e comparacoes. A API FastAPI entrega recomendacoes por usuario.

## Result

A entrega possui projeto instalavel, testes, lint, Docker, documentacao, Model Card e
artefatos versionaveis. O modelo neural pode ser comparado com baseline usando metricas
classicas e metricas de ranking.

# Arquitetura

O projeto implementa uma solucao de recomendacao para e-commerce baseada em interacoes
de navegacao dos usuarios.

## Fluxo

1. `generate`: cria usuarios, itens e interacoes sinteticas com mais de 10.000 eventos.
2. `prepare`: codifica `user_id` e `item_id`, cria alvo binario e separa treino, validacao e teste.
3. `train`: treina baseline Scikit-Learn e MLP PyTorch com embeddings de usuario e item.
4. `evaluate`: compara os modelos com ROC-AUC, PR-AUC, F1, Precision@10 e Recall@10.
5. `api`: entrega recomendacoes por usuario usando FastAPI.

## Padroes aplicados

- Factory: `build_model` centraliza a construcao da rede neural.
- Strategy: baseline e rede neural compartilham a mesma interface de avaliacao por scores.
- Template Method: os estagios DVC padronizam reproducao do pipeline.

## Componentes

- `src/ecommerce_recommender/data`: geracao e entrada de dados.
- `src/ecommerce_recommender/features`: preparacao de features e splits.
- `src/ecommerce_recommender/models`: baselines, MLP, metricas, treino e avaliacao.
- `src/ecommerce_recommender/services`: servico de recomendacao.
- `src/ecommerce_recommender/api`: camada HTTP.

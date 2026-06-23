# Model Card

## Modelo

Rede neural MLP em PyTorch com embeddings para usuarios e produtos.

## Uso pretendido

Rankear produtos provaveis de interesse para usuarios de e-commerce com base em historico
de visualizacoes, carrinho e compras.

## Dados

O projeto usa dataset sintetico versionavel, com usuarios, produtos e interacoes. O dataset
foi desenhado para ter volume suficiente para testes academicos e pipeline reproduzivel.

## Metricas

- Accuracy
- F1-score
- ROC-AUC
- PR-AUC
- Precision@10
- Recall@10

## Limitacoes

- Dados sinteticos nao substituem validacao com dados reais.
- Cold start e tratado apenas como erro controlado na API.
- O modelo nao incorpora texto, imagem, estoque ou margem comercial.

## Riscos e vieses

Usuarios de menor frequencia podem receber recomendacoes menos personalizadas. Em producao,
seria necessario medir cobertura, diversidade, novidade e impacto por segmento.

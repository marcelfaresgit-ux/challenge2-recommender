# Resultados Esperados

Execute o pipeline completo com:

```bash
dvc repro
```

As metricas finais sao salvas em:

```text
data/artifacts/metrics.json
```

O projeto registra no MLflow:

- parametros da MLP
- perda de treino por epoca
- ROC-AUC de validacao
- metricas finais do baseline
- metricas finais da rede neural

## Interpretacao

O baseline Random Forest serve como comparacao forte e simples. A MLP com embeddings e o
modelo central porque aprende representacoes latentes de usuarios e produtos, uma abordagem
adequada para recomendacao baseada em comportamento.

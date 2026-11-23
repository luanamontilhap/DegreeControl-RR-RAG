# Revisão do experimento oficial MultiHop-RAG

## Arquivos analisados

- `experimento_oficial_multihop_rag (2).py` (902 linhas; auditoria das seções de preparação, mapeamento, recuperação, métricas, telemetria, estatística e relatórios).
- Texto com a solicitação metodológica e texto com amostra das colunas/registros preliminares.

Os resultados da amostra foram usados apenas para auditoria. Nenhum resultado antigo foi incorporado à execução futura.

## Alterações feitas

- Degree-Control mantém o score original Bi-Encoder do melhor chunk como numerador: `bi_score / sqrt(1 + grau_ponderado)`. O grafo usa os centróides dos chunks dos documentos candidatos, cosseno positivo e diagonal zero, apenas para obter o grau.
- A codificação da consulta, a seleção dos 20 artigos candidatos e os dois rerankers agora têm medições separadas. Os três rankings partem exatamente da mesma lista de candidatos. Tempos totais do Bi-Encoder, Bi-Encoder + Degree-Control e Bi-Encoder + Cross-Encoder são registrados como soma explícita de suas etapas cronometradas.
- O Cross-Encoder recebe a consulta e o título mais o melhor chunk selecionado pelo Bi-Encoder.
- Aquecimento de consulta e Cross-Encoder ocorre antes das medições.
- A telemetria guarda tempo de parede, delta de CPU do processo, amostras de utilização de CPU do processo, threads do processo e memória CUDA alocada/reservada antes, depois, pico absoluto e incremento sobre o baseline.
- Mapeamento de evidências tem relatório por evidência, contagens de status, URLs duplicadas, evidências ambíguas/não mapeadas e casos em que várias evidências apontam ao mesmo documento.
- Regra de elegibilidade agora exige pergunta não nula e não vazia, ao menos uma evidência e mapeamento inequívoco de cada item. Evidências múltiplas que apontam ao mesmo documento viram um único documento gold e, por si só, não excluem a pergunta. Essa interpretação foi escolhida para cumprir o requisito explícito de não tratar duplicatas como evidências ausentes; a regra está gravada no relatório de auditoria.
- Salvamento por questão em JSONL com fingerprint da execução e retomada sem duplicar resultados. Uma linha final incompleta por interrupção é removida; corrupção no meio do checkpoint ou IDs repetidos fazem a execução parar, em vez de aceitar dados incertos.
- `Candidate Recall@20` é exportado como métrica compartilhada da recuperação e não como medição independente de cada reranker.
- Bootstrap pareado por pergunta com 5.000 reamostragens e seed 42 para F1, Recall, NDCG, Precision, MRR e Full Evidence. A interpretação declara quando o intervalo inclui zero e não chama isso de equivalência.
- Metadados registram dataset/configurações, parâmetros, revisão dos modelos quando disponível, versões, ambiente, hardware, amostra selecionada, contagem de evidências, definições de latência e limitações da telemetria.
- O código é um `.py` importável; a instalação do Colab aparece como instrução de célula separada, sem o comando mágico `!pip` inválido em Python comum.

## Problemas encontrados no original

1. Degree-Control calculava seu numerador com similaridade consulta-centroide, trocando a função de relevância que deveria ser mantida.
2. O sumário selecionava `rerank_seconds`, mas a coluna produzida era `latency_seconds`; isso causaria falha antes do bootstrap.
3. A latência Bi-Encoder agregava codificação e recuperação, sem discriminar suas partes; não havia tempos totais comparáveis dos pipelines.
4. `psutil.cpu_percent(interval=None)` era uma leitura pontual sem janela temporal. `torch.cuda.memory_allocated()` era apenas um estado instantâneo e não registrava pico, reservado, baseline ou incremento.
5. A elegibilidade comparava quantidade de evidências brutas com documentos gold únicos, de modo que evidências duplicadas para um mesmo documento podiam ser interpretadas como ausência.
6. A auditoria de mapeamento não salvava todas as evidências individualmente nem quantificava de modo suficiente URLs duplicadas e colisões de evidência para documento.
7. Os resultados só eram gravados ao fim; uma interrupção perdia o progresso. O script também continha blocos de relatório/inspeção fora de ordem que dependiam de arquivos anteriores e podiam falhar.
8. O comando `!pip install` tornava o arquivo incompatível com compilação/importação Python fora do notebook.

## Métricas e medições

Precision@5 usa denominador fixo 5; Recall@5 usa o número de documentos gold únicos; F1@5 é a média harmônica; Full Evidence@5 vale 1 quando todos os gold únicos estão no top 5; MRR@5 usa o primeiro documento relevante; NDCG@5 usa relevância binária e ganho DCG padrão. Candidate Recall@20 é calculado uma vez por questão e compartilhado pelos três métodos.

Latência é tempo de parede com `perf_counter` e sincronização CUDA antes/depois. A média, mediana, desvio-padrão, p95, mínimo e máximo são exportados por etapa. O tempo total do pipeline soma as etapas medidas que o compõem. CPU do processo é delta de tempo user+system; percentual de CPU é uma média amostrada a cada 10 ms e pode não existir para uma etapa curta. Memória GPU total observada inclui modelos/caches residentes; incremento de pico é relativo ao baseline antes da etapa.

## Verificações executadas

- Compilação Python e análise AST: aprovadas para o script de entrega.
- Verificações sintéticas das fórmulas de Precision/Recall/F1, Full Evidence, MRR e NDCG: aprovadas.
- Ranking vazio: NDCG zero, conforme esperado.
- Amostragem estratificada: tamanho correto e resultado determinístico para seed fixo.
- Bootstrap pareado sintético com diferenças constantes: média e intervalo esperados.
- Verificação estática do numerador original no Degree-Control, ausência da referência quebrada `rerank_seconds` e ausência de magia `!pip` no `.py`: aprovada.

Uma primeira invocação do preparador sintético falhou porque não recebeu `TOP_K`; o teste foi corrigido e repetido com sucesso. Isso não executou o experimento.

## Limitações antes da execução oficial

- O dataset, os pesos dos modelos, os nomes exatos de colunas atuais e a compatibilidade das versões do Hugging Face não foram validados por download; isso deve ser confirmado na primeira execução de preparação no Colab.
- Revisões dos modelos podem não ser expostas pelas versões das bibliotecas; nesse caso o metadado será `null`, embora o nome do modelo e `pip freeze` permaneçam registrados.
- Amostragem CPU de 10 ms pode não capturar etapas muito rápidas, e inclui pequena sobrecarga. A ocupação de outros processos/GPU no Colab não é controlável nem atribuível ao método.
- Picos CUDA são alocador PyTorch, não memória total do dispositivo; a reserva pode persistir após uma etapa. Não se descarregam modelos entre métodos.
- Cache de embeddings depende do nome do modelo e do conteúdo/chunking do corpus. Para reprodutibilidade estrita ao longo do tempo, convém fixar revisão dos modelos/dataset quando disponíveis.
- IC bootstrap contendo zero indica ausência de evidência clara de diferença sob essa análise, não prova equivalência. Tipos de pergunta são preservados e sumarizados, mas não há alegação inferencial estratificada separada.
- O diagnóstico de `null_query` executa codificação e recuperação após a avaliação principal, fora das métricas comparativas; ele acrescenta algum custo à sessão total do Colab, mas não aos tempos por método registrados.

**O benchmark oficial não foi executado.**

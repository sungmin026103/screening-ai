SR Studio v22 — AI screening methodology update

- Human labeling remains fixed at 200 records.
- No iterative active-learning/relabeling step is added after model fitting.
- Screening threshold is selected from out-of-fold cross-validation probabilities to satisfy Recall >= 95% while maximizing WSS.
- Reports measured Recall, FN, Precision, ROC-AUC, Average Precision, WSS@95, and WSS@100.
- WSS@100 is a reference calculation from the same CV predictions; it does not require additional human screening.
- False-negative records can be inspected and exported for error analysis.
- The UI explicitly states that CV performance is internal validation and does not guarantee performance on unlabeled records.

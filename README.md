# QML-EVOVAQ
## Procedure
The QML model has three main parts:

- classical input layer
- quantum circuit
- classical output layer

The Adam baseline trains the full model end-to-end.

The EVOVAQ branch first trains the classical layers, then freezes them and optimizes only the quantum circuit parameters with EVOVAQ. After that, the classical layers can be briefly adapted again.

## Evaluation

Each fold splits into:

- train_inner: Adam training
- evo_val: EVOVAQ fitness
- final_val: final evaluation

The reported models are:

- QML Adam
- QML EVOVAQ
- QML EVOVAQ+Adaptation

The models are evaluated with standard metrics and SAFE-AI metrics.
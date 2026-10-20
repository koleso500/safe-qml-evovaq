# SAFE hybrid QML with stage-wise optimization

Code and reference results for **Stage-wise Optimization of SAFE Hybrid Models for Credit Risk Classification**.

The project compares four models on 2017 New York HMDA mortgage applications:

| Paper name | Name in result CSVs | Training |
| --- | --- | --- |
| RF | Random Forest | Classical baseline |
| E2E-QML | QML Adam | End-to-end Adam |
| SWO-QML-1 | QML EVOVAQ | Classical Adam training, then quantum optimization with a genetic algorithm |
| SWO-QML-2 | QML EVOVAQ+Adaptation | SWO-QML-1 followed by classical fine-tuning |

Evaluation includes accuracy, macro-F1, MSE, and SAFE-AI metrics.

## Setup

Use **Python 3.12** in an activated virtual environment. Run commands from the repository root:

```bash
python -m pip install -r requirements.txt
```

For the original CUDA 11.8 environment, use `requirements-cuda.txt` instead. Training also supports `--device cpu`.

## Data

Download **2017 New York**, **All records**, **Plain language labels and HMDA codes** from the [CFPB historic HMDA page](https://www.consumerfinance.gov/data-research/hmda/historic-data/) and save the CSV as:

```text
data/hmda_2017_ny_all-records_labels.csv
```

The paper configs apply preprocessing and a fixed stratified 10% subsample (seed 42), yielding 17,148 samples with 20 features. Each run uses an 80/20 train/test split. See [data details](data/README.md).

## Reproduce the results

Rebuild Tables II-III and Figure 2 from the archived results, without training or downloading HMDA:

```bash
python -m scripts.reproduce_paper
```

Outputs are saved to `results/paper_rebuilt/`. Reference CSVs are in `paper_results/`. Table III contains theoretical quantum-parameter optimization budgets.

To rerun training from the raw data:

```bash
# Depth sweep: 5 qubits, 1-5 layers
python -m scripts.run_hmda_qml_evovaq --config configs/paper_depth.yaml

# Qubit sweep: 3-7 qubits, 2 layers
python -m scripts.run_hmda_qml_evovaq --config configs/paper_qubits.yaml
```

Both configs use five seeds (42-46) and save metrics, SAFE curves, and plots to separate directories under `results/`. Figure 2's configuration (5 qubits, 5 layers) is included in the depth sweep. Hyperparameters are specified in the YAML files.

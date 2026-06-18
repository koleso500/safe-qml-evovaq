import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_hmda import load_clean_hmda_from_config
from src.evovaq_optimizer import train_quantum_with_evovaq
from src.qml_model import QuantumClassifier, predict_proba_qml
from src.safe_evaluation import evaluate_qml_safe_metrics
from src.train_qml import compute_metrics, evaluate_qml_log_loss, train_model
from src.utils import ensure_dir, get_device, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_NAMES = [
    'Random Forest',
    'QML Adam',
    'QML EVOVAQ',
    'QML EVOVAQ+Adaptation'
]

METRIC_COLUMNS = [
    'acc',
    'f1',
    'mse',
    'train_log_loss',
    'test_log_loss',
    'rga',
    'aurga',
    'aurgr',
    'aurge'
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--run-index', type=int, default=None)
    parser.add_argument('--layers', type=str, default=None)
    parser.add_argument('--qubits', type=str, default=None)

    return parser.parse_args()


def as_list_of_ints(value):
    if value is None:
        raise ValueError('Expected a list of integers, got None.')

    if isinstance(value, str):
        values = [v.strip() for v in value.split(',') if v.strip()]
        return [int(v) for v in values]

    return [int(v) for v in value]


def get_run_seeds(config):
    run_seeds = as_list_of_ints(config['experiment']['run_seeds'])

    if not run_seeds:
        raise ValueError('Experiment must contain at least one seed.')

    return run_seeds


def get_layer_values(args, quantum_cfg):
    if args.layers is not None:
        layer_values = as_list_of_ints(args.layers)
    else:
        layer_values = as_list_of_ints(quantum_cfg['layers'])

    if not layer_values:
        raise ValueError('Quantum layers must contain at least one value.')

    return layer_values


def get_qubit_values(args, quantum_cfg):
    if args.qubits is not None:
        qubit_values = as_list_of_ints(args.qubits)
    else:
        qubit_values = as_list_of_ints(quantum_cfg['n_qubits'])

    if not qubit_values:
        raise ValueError('Quantum qubits must contain at least one value.')

    return qubit_values


def safe_log_loss_value(y_true, probs):
    return float(
        log_loss(
            y_true,
            probs,
            labels=np.arange(probs.shape[1])
        )
    )


def add_standard_row(
    rows,
    *,
    run_id,
    run_seed,
    n_layers,
    n_qubits,
    model_name,
    y_train,
    prob_train,
    y_test,
    prob_test,
    extra=None
):
    acc, f1m, mse = compute_metrics(y_test, prob_test)

    row = {
        'run_id': int(run_id),
        'run_seed': int(run_seed),
        'n_layers': int(n_layers),
        'n_qubits': int(n_qubits),
        'n_quantum_params': int(n_layers * n_qubits * 3),
        'model': model_name,

        'acc': float(acc),
        'f1': float(f1m),
        'mse': float(mse),

        'train_log_loss': safe_log_loss_value(y_train, prob_train),
        'test_log_loss': safe_log_loss_value(y_test, prob_test)
    }

    if extra is not None:
        row.update(extra)

    rows.append(row)

    print(
        f'{model_name} | '
        f'ACC={acc:.4f} | F1={f1m:.4f} | MSE={mse:.6f} | '
        f'train_log_loss={row["train_log_loss"]:.4f} | '
        f'test_log_loss={row["test_log_loss"]:.4f}'
    )

    return row


def merge_safe_metrics(run_rows, safe_store):
    for row in run_rows:
        model_name = row['model']

        if model_name not in safe_store:
            raise KeyError(f'Missing SAFE metrics for model: {model_name}')

        row['rga'] = float(safe_store[model_name]['rga_full'][-1])
        row['aurga'] = float(safe_store[model_name]['aurga'][-1])
        row['aurgr'] = float(safe_store[model_name]['aurgr'][-1])
        row['aurge'] = float(safe_store[model_name]['aurge'][-1])


def summarize_results(results_df):
    available_metrics = [
        col for col in METRIC_COLUMNS
        if col in results_df.columns
    ]

    agg_dict = {}

    for metric in available_metrics:
        agg_dict[f'{metric}_mean'] = (metric, 'mean')
        agg_dict[f'{metric}_std'] = (metric, 'std')
        agg_dict[f'{metric}_median'] = (metric, 'median')
        agg_dict[f'{metric}_q1'] = (metric, lambda s: s.quantile(0.25))
        agg_dict[f'{metric}_q3'] = (metric, lambda s: s.quantile(0.75))

    summary = (
        results_df
        .groupby(
            ['n_qubits', 'n_layers', 'n_quantum_params', 'model'],
            dropna=False
        )
        .agg(
            n_runs=('run_id', 'nunique'),
            **agg_dict
        )
        .reset_index()
    )

    return summary


def build_safe_model_entry(model, prob, model_type, class_order):
    return {
        'model': model,
        'prob': prob,
        'model_type': model_type,
        'class_order': class_order
    }


def make_random_forest(config, run_seed):
    rf_cfg = config['random_forest']

    return RandomForestClassifier(
        n_estimators=int(rf_cfg['n_estimators']),
        max_depth=rf_cfg['max_depth'],
        min_samples_leaf=int(rf_cfg['min_samples_leaf']),
        class_weight=rf_cfg['class_weight'],
        random_state=run_seed,
        n_jobs=int(rf_cfg['n_jobs'])
    )


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seed = int(config['seed'])

    device = get_device()
    print('Using device:', device)

    set_global_seed(seed)

    paths_cfg = config['paths']
    fig_dir = PROJECT_ROOT / paths_cfg['fig_hmda_dir']
    csv_dir = PROJECT_ROOT / paths_cfg['csv_dir']

    ensure_dir(fig_dir)
    ensure_dir(csv_dir)

    x, y, feature_names = load_clean_hmda_from_config(
        PROJECT_ROOT,
        config,
        seed
    )

    experiment_cfg = config['experiment']
    training_cfg = config['training']
    quantum_cfg = config['quantum']
    evovaq_cfg = config['evovaq']
    safe_cfg = config['safe']

    run_seeds = get_run_seeds(config)

    if args.run_index is not None:
        if args.run_index < 1 or args.run_index > len(run_seeds):
            raise ValueError(
                f'--run-index must be between 1 and {len(run_seeds)}'
            )

        run_seeds = [run_seeds[args.run_index - 1]]

    layer_values = get_layer_values(args, quantum_cfg)
    qubit_values = get_qubit_values(args, quantum_cfg)

    test_size = float(experiment_cfg['test_size'])
    stratify_split = bool(experiment_cfg['stratify_split'])

    epochs = int(training_cfg['epochs'])
    batch_size_train = int(training_cfg['batch_size'])
    lr_torch = float(training_cfg['learning_rate'])
    class_weight_balanced = bool(training_cfg['class_weight_balanced'])

    noise_enabled = bool(quantum_cfg['noise_enabled'])
    noise_type = str(quantum_cfg['noise_type'])
    noise_prob = float(quantum_cfg['noise_prob'])

    evovaq_adapt_epochs = int(evovaq_cfg['adapt_epochs'])
    evovaq_adapt_lr = float(evovaq_cfg['adapt_lr'])

    safe_enabled = bool(safe_cfg['enabled'])
    n_segments = int(safe_cfg['n_segments'])
    rga_curve_method = str(safe_cfg['rga_curve_method'])

    noise_levels = np.linspace(
        float(safe_cfg['rgr_noise_min']),
        float(safe_cfg['rgr_noise_max']),
        int(safe_cfg['rgr_noise_points'])
    )

    rge_n_steps_config = safe_cfg['rge_tabular_n_steps']
    rge_masking_method = safe_cfg['rge_tabular_masking_method']
    rge_baseline = safe_cfg['rge_tabular_baseline']

    class_order = np.array([0, 1], dtype=int)
    n_classes = 2

    print(
        'Run | '
        f'n_runs={len(run_seeds)} | '
        f'test_size={test_size} | '
        f'layers={layer_values} | '
        f'n_qubits={qubit_values}'
    )

    print(
        'EVOVAQ fixed params | '
        f"population_size={evovaq_cfg['population_size']} | "
        f"max_gen={evovaq_cfg['max_gen']} | "
        f"tournament_size={evovaq_cfg['tournament_size']}"
    )

    print(
        'Quantum noise | '
        f'enabled={noise_enabled} | '
        f'type={noise_type} | '
        f'prob={noise_prob}'
    )

    all_rows = []

    for run_id, run_seed in enumerate(run_seeds, start=1):
        print(
            f'\n================ RUN {run_id}/{len(run_seeds)} '
            f'| seed={run_seed} ================'
        )

        set_global_seed(run_seed)

        x_train_raw, x_test_raw, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=run_seed,
            stratify=y if stratify_split else None
        )

        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
        x_test = scaler.transform(x_test_raw).astype(np.float32)

        print(
            'Split sizes | '
            f'train={len(x_train)} | '
            f'test={len(x_test)} | '
            f'features={x_train.shape[1]}'
        )

        if rge_n_steps_config is None:
            rge_n_steps = x_test.shape[1]
        else:
            rge_n_steps = int(rge_n_steps_config)

        input_dim = x_train.shape[1]

        print('\n--- Random Forest ---')
        rf_model = make_random_forest(config, run_seed)
        rf_model.fit(x_train, y_train)

        prob_rf_train = rf_model.predict_proba(x_train)
        prob_rf_test = rf_model.predict_proba(x_test)

        for n_qubits in qubit_values:
            n_qubits = int(n_qubits)

            for n_layers in layer_values:
                n_layers = int(n_layers)

                print(
                    f'\n================ QML CONFIG | '
                    f'n_qubits={n_qubits} | '
                    f'n_layers={n_layers} | ================'
                )

                run_rows = []

                add_standard_row(
                    run_rows,
                    run_id=run_id,
                    run_seed=run_seed,
                    n_layers=n_layers,
                    n_qubits=n_qubits,
                    model_name='Random Forest',
                    y_train=y_train,
                    prob_train=prob_rf_train,
                    y_test=y_test,
                    prob_test=prob_rf_test
                )

                set_global_seed(run_seed + 1000 * n_layers + 10000 * n_qubits)

                base_model = QuantumClassifier(
                    input_dim=input_dim,
                    num_classes=n_classes,
                    n_qubits=n_qubits,
                    n_layers=n_layers,
                    noise_enabled=noise_enabled,
                    noise_type=noise_type,
                    noise_prob=noise_prob
                ).to(device)

                print('\n--- QML Adam ---')

                qml_adam = copy.deepcopy(base_model)

                with qml.Tracker(qml_adam.q_device) as adam_tracker:
                    qml_adam = train_model(
                        qml_adam,
                        x_train,
                        y_train,
                        epochs=epochs,
                        batch_size=batch_size_train,
                        lr=lr_torch,
                        device=device,
                        label_prefix=f'QML Adam Q{n_qubits} L{n_layers}',
                        use_validation=False,
                        class_weight_balanced=class_weight_balanced
                    )

                print('Adam tracker total:', adam_tracker.totals)

                prob_adam_train = predict_proba_qml(
                    qml_adam,
                    x_train,
                    device
                )

                prob_adam_test = predict_proba_qml(
                    qml_adam,
                    x_test,
                    device
                )

                add_standard_row(
                    run_rows,
                    run_id=run_id,
                    run_seed=run_seed,
                    n_layers=n_layers,
                    n_qubits=n_qubits,
                    model_name='QML Adam',
                    y_train=y_train,
                    prob_train=prob_adam_train,
                    y_test=y_test,
                    prob_test=prob_adam_test,
                    extra={
                        'adam_tracker_executions': adam_tracker.totals.get(
                            'executions',
                            np.nan
                        ),
                        'adam_tracker_batches': adam_tracker.totals.get(
                            'batches',
                            np.nan
                        )
                    }
                )

                print('\n--- QML EVOVAQ: classical part ---')

                qml_evovaq_start = copy.deepcopy(base_model)

                qml_evovaq_start = train_model(
                    qml_evovaq_start,
                    x_train,
                    y_train,
                    epochs=epochs,
                    batch_size=batch_size_train,
                    lr=lr_torch,
                    device=device,
                    trainable_params=(
                            list(qml_evovaq_start.pre_pqc.parameters())
                            + list(qml_evovaq_start.fc.parameters())
                    ),
                    freeze_params=qml_evovaq_start.q_layer.parameters(),
                    label_prefix=f'QML EVOVAQ start Q{n_qubits} L{n_layers}',
                    use_validation=False,
                    class_weight_balanced=class_weight_balanced
                )

                initial_evovaq_state = copy.deepcopy(
                    qml_evovaq_start.state_dict()
                )

                initial_train_loss = evaluate_qml_log_loss(
                    qml_evovaq_start,
                    x_train,
                    y_train,
                    device
                )

                initial_test_loss = evaluate_qml_log_loss(
                    qml_evovaq_start,
                    x_test,
                    y_test,
                    device
                )

                print(
                    f'EVOVAQ start | '
                    f'train_log_loss={initial_train_loss:.4f} | '
                    f'test_log_loss={initial_test_loss:.4f}'
                )

                print('\n--- QML EVOVAQ: optimization part ---')

                qml_evovaq = copy.deepcopy(base_model)
                qml_evovaq.load_state_dict(initial_evovaq_state)

                qml_evovaq, evovaq_info = train_quantum_with_evovaq(
                    qml_evovaq,
                    x_train,
                    y_train,
                    seed=run_seed,
                    n_layers=n_layers,
                    n_qubits=n_qubits,
                    evovaq_cfg=evovaq_cfg,
                    device=device
                )

                qml_after_evovaq = copy.deepcopy(qml_evovaq)

                prob_evovaq_train = predict_proba_qml(
                    qml_after_evovaq,
                    x_train,
                    device
                )

                prob_evovaq_test = predict_proba_qml(
                    qml_after_evovaq,
                    x_test,
                    device
                )

                add_standard_row(
                    run_rows,
                    run_id=run_id,
                    run_seed=run_seed,
                    n_layers=n_layers,
                    n_qubits=n_qubits,
                    model_name='QML EVOVAQ',
                    y_train=y_train,
                    prob_train=prob_evovaq_train,
                    y_test=y_test,
                    prob_test=prob_evovaq_test,
                    extra={
                        'initial_train_log_loss': initial_train_loss,
                        'initial_test_log_loss': initial_test_loss,
                        **{
                            f'evovaq_{key}': value
                            for key, value in evovaq_info.items()
                        }
                    }
                )

                print('\n--- QML EVOVAQ+Adaptation ---')

                qml_evovaq = train_model(
                    qml_evovaq,
                    x_train,
                    y_train,
                    epochs=evovaq_adapt_epochs,
                    batch_size=batch_size_train,
                    lr=evovaq_adapt_lr,
                    device=device,
                    trainable_params=(
                            list(qml_evovaq.pre_pqc.parameters())
                            + list(qml_evovaq.fc.parameters())
                    ),
                    freeze_params=qml_evovaq.q_layer.parameters(),
                    label_prefix=f'QML EVOVAQ+Adaptation Q{n_qubits} L{n_layers}',
                    use_validation=False,
                    class_weight_balanced=class_weight_balanced
                )

                qml_evovaq_adapted = copy.deepcopy(qml_evovaq)

                prob_adapted_train = predict_proba_qml(
                    qml_evovaq_adapted,
                    x_train,
                    device
                )

                prob_adapted_test = predict_proba_qml(
                    qml_evovaq_adapted,
                    x_test,
                    device
                )

                add_standard_row(
                    run_rows,
                    run_id=run_id,
                    run_seed=run_seed,
                    n_layers=n_layers,
                    n_qubits=n_qubits,
                    model_name='QML EVOVAQ+Adaptation',
                    y_train=y_train,
                    prob_train=prob_adapted_train,
                    y_test=y_test,
                    prob_test=prob_adapted_test,
                    extra={
                        'initial_train_log_loss': initial_train_loss,
                        'initial_test_log_loss': initial_test_loss,
                        **{
                            f'evovaq_{key}': value
                            for key, value in evovaq_info.items()
                        }
                    }
                )

                if safe_enabled:
                    safe_models = {
                        'Random Forest': build_safe_model_entry(
                            rf_model,
                            prob_rf_test,
                            'sklearn',
                            class_order
                        ),
                        'QML Adam': build_safe_model_entry(
                            qml_adam,
                            prob_adam_test,
                            'pytorch',
                            class_order
                        ),
                        'QML EVOVAQ': build_safe_model_entry(
                            qml_after_evovaq,
                            prob_evovaq_test,
                            'pytorch',
                            class_order
                        ),
                        'QML EVOVAQ+Adaptation': build_safe_model_entry(
                            qml_evovaq_adapted,
                            prob_adapted_test,
                            'pytorch',
                            class_order
                        )
                    }

                    run_label = f'run{run_id}_seed{run_seed}_Q{n_qubits}_L{n_layers}'

                    safe_store = evaluate_qml_safe_metrics(
                        safe_models=safe_models,
                        x_final_test=x_test,
                        y_final_test=y_test,
                        feature_names=feature_names,
                        class_order=class_order,
                        n_segments=n_segments,
                        noise_levels=noise_levels,
                        rge_n_steps=rge_n_steps,
                        rge_masking_method=rge_masking_method,
                        rge_baseline=rge_baseline,
                        fig_dir=fig_dir,
                        run_label=run_label,
                        seed=run_seed,
                        device=device,
                        rga_curve_method=rga_curve_method
                    )

                    merge_safe_metrics(run_rows, safe_store)

                all_rows.extend(run_rows)

                partial_df = pd.DataFrame(all_rows)
                partial_csv = csv_dir / 'hmda_qml_evovaq_per_run_partial.csv'
                partial_df.to_csv(partial_csv, index=False)

                print(f'Partial results saved to: {partial_csv}')

    results_df = pd.DataFrame(all_rows)
    summary_df = summarize_results(results_df)

    suffix = f'{len(run_seeds)}runs_{len(qubit_values)}qubits_{len(layer_values)}layers'

    per_run_csv = csv_dir / f'hmda_qml_evovaq_per_run_{suffix}.csv'
    summary_csv = csv_dir / f'hmda_qml_evovaq_summary_{suffix}.csv'

    results_df.to_csv(per_run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print('\n================ SUMMARY ================')
    print(summary_df.to_string(index=False))

    print(f'\nSaved per-run results to: {per_run_csv}')
    print(f'Saved summary to: {summary_csv}')


if __name__ == '__main__':
    main()
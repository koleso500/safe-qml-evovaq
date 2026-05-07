import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from src.data_hmda import load_clean_hmda_from_config
from src.evovaq_optimizer import train_quantum_with_evovaq
from src.qml_model import QuantumClassifier, predict_proba_qml
from src.safe_evaluation import evaluate_qml_safe_metrics
from src.train_qml import compute_metrics, evaluate_qml_log_loss, train_model
from src.utils import ensure_dir, get_device, set_global_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--config', type=str, default='config.yaml')

    return parser.parse_args()


def add_standard_result(results, model_name, y_true, probs):
    acc, f1m, mse = compute_metrics(y_true, probs)
    results[model_name]['acc'].append(acc)
    results[model_name]['f1'].append(f1m)
    results[model_name]['mse'].append(mse)

    return acc, f1m, mse


def main():
    args = parse_args()

    config_path = PROJECT_ROOT / args.config

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    seed = int(config['seed'])
    n_splits = int(config['cross_validation']['n_splits'])

    device = get_device()
    print('Using device:', device)

    set_global_seed(seed)

    paths_cfg = config['paths']
    fig_dir_name = paths_cfg['fig_hmda_dir']
    csv_dir_name = paths_cfg['csv_dir']

    fig_dir = PROJECT_ROOT / fig_dir_name
    csv_dir = PROJECT_ROOT / csv_dir_name

    ensure_dir(fig_dir)
    ensure_dir(csv_dir)

    x, y, feature_names = load_clean_hmda_from_config(
        PROJECT_ROOT,
        config,
        seed
    )

    training_cfg = config['training']
    quantum_cfg = config['quantum']
    evovaq_cfg = config['evovaq']
    safe_cfg = config['safe']

    epochs = int(training_cfg['epochs'])
    batch_size_train = int(training_cfg['batch_size'])
    lr_torch = float(training_cfg['learning_rate'])

    n_qubits = int(quantum_cfg['n_qubits'])
    n_layers = int(quantum_cfg['n_layers'])

    evovaq_adapt_epochs = int(evovaq_cfg['adapt_epochs'])
    evovaq_adapt_lr = float(evovaq_cfg['adapt_lr'])
    inner_evo_val_size = float(evovaq_cfg['inner_evo_val_size'])

    n_segments = int(safe_cfg['n_segments'])
    noise_levels = np.linspace(0.0, 1.0, 11)

    rge_n_steps_config = safe_cfg['rge_tabular_n_steps']
    rge_masking_method = safe_cfg['rge_tabular_masking_method']
    rge_baseline = safe_cfg['rge_tabular_baseline']

    class_order = np.array([0, 1], dtype=int)
    n_classes = 2

    k_fold = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed
    )

    model_names = [
        'QML Adam',
        'QML EVOVAQ',
        'QML EVOVAQ+Adaptation'
    ]

    results = {
        model_name: {
            'acc': [],
            'f1': [],
            'mse': []
        }
        for model_name in model_names
    }

    safe_store_all = {
        model_name: {
            'rga_curve': [],
            'rga_full': [],
            'aurga': [],
            'rgr_curve': [],
            'aurgr': [],
            'rge_curve': [],
            'aurge': []
        }
        for model_name in model_names
    }

    detail_rows = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(x, y), 1):
        if args.fold is not None and fold != args.fold:
            continue

        print(f'\n====== Fold {fold} ======')

        set_global_seed(seed + fold)

        x_train_outer_raw, x_final_val_raw = x[train_idx], x[val_idx]
        y_train_outer, y_final_val = y[train_idx], y[val_idx]

        x_train_inner_raw, x_evo_val_raw, y_train_inner, y_evo_val = train_test_split(
            x_train_outer_raw,
            y_train_outer,
            test_size=inner_evo_val_size,
            random_state=seed + fold,
            stratify=y_train_outer
        )

        scaler = StandardScaler()

        x_train_inner = scaler.fit_transform(x_train_inner_raw).astype(np.float32)
        x_evo_val = scaler.transform(x_evo_val_raw).astype(np.float32)
        x_final_val = scaler.transform(x_final_val_raw).astype(np.float32)

        print(
            'Split sizes | '
            f'train_inner={len(x_train_inner)} | '
            f'evo_val={len(x_evo_val)} | '
            f'final_val={len(x_final_val)}'
        )

        if rge_n_steps_config is None:
            rge_n_steps = x_final_val.shape[1]
        else:
            rge_n_steps = int(rge_n_steps_config)

        input_dim = x_train_inner.shape[1]

        safe_models = {}

        print('\n--- QML Adam ---')

        qml_adam = QuantumClassifier(
            input_dim=input_dim,
            num_classes=n_classes,
            n_qubits=n_qubits,
            n_layers=n_layers
        ).to(device)

        qml_adam = train_model(
            qml_adam,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            epochs=epochs,
            batch_size=batch_size_train,
            lr=lr_torch,
            device=device,
            label_prefix='QML Adam'
        )

        adam_evo_loss = evaluate_qml_log_loss(
            qml_adam,
            x_evo_val,
            y_evo_val,
            device
        )

        adam_final_loss = evaluate_qml_log_loss(
            qml_adam,
            x_final_val,
            y_final_val,
            device
        )

        prob_adam = predict_proba_qml(
            qml_adam,
            x_final_val,
            device
        )

        acc, f1m, mse = add_standard_result(
            results,
            'QML Adam',
            y_final_val,
            prob_adam
        )

        safe_models['QML Adam'] = {
            'model': qml_adam,
            'prob': prob_adam
        }

        print(
            f'QML Adam | ACC={acc:.4f} | F1={f1m:.4f} | '
            f'MSE={mse:.6f} | evo_log_loss={adam_evo_loss:.4f} | '
            f'final_log_loss={adam_final_loss:.4f}'
        )

        print('\n--- QML EVOVAQ ---')

        qml_evovaq = QuantumClassifier(
            input_dim=input_dim,
            num_classes=n_classes,
            n_qubits=n_qubits,
            n_layers=n_layers
        ).to(device)

        qml_evovaq = train_model(
            qml_evovaq,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            epochs=epochs,
            batch_size=batch_size_train,
            lr=lr_torch,
            device=device,
            trainable_params=list(qml_evovaq.pre_pqc.parameters()) + list(qml_evovaq.fc.parameters()),
            freeze_params=qml_evovaq.q_layer.parameters(),
            label_prefix='QML EVOVAQ initial training'
        )

        initial_evo_loss = evaluate_qml_log_loss(
            qml_evovaq,
            x_evo_val,
            y_evo_val,
            device
        )

        initial_final_loss = evaluate_qml_log_loss(
            qml_evovaq,
            x_final_val,
            y_final_val,
            device
        )

        qml_evovaq, evovaq_info = train_quantum_with_evovaq(
            qml_evovaq,
            x_evo_val,
            y_evo_val,
            seed=seed + fold,
            n_layers=n_layers,
            n_qubits=n_qubits,
            evovaq_cfg=evovaq_cfg,
            device=device
        )

        qml_after_evovaq = copy.deepcopy(qml_evovaq)

        evovaq_evo_loss = evaluate_qml_log_loss(
            qml_after_evovaq,
            x_evo_val,
            y_evo_val,
            device
        )

        evovaq_final_loss = evaluate_qml_log_loss(
            qml_after_evovaq,
            x_final_val,
            y_final_val,
            device
        )

        prob_evovaq = predict_proba_qml(
            qml_after_evovaq,
            x_final_val,
            device
        )

        acc, f1m, mse = add_standard_result(
            results,
            'QML EVOVAQ',
            y_final_val,
            prob_evovaq
        )

        safe_models['QML EVOVAQ'] = {
            'model': qml_after_evovaq,
            'prob': prob_evovaq
        }

        print(
            f'QML EVOVAQ | ACC={acc:.4f} | F1={f1m:.4f} | '
            f'MSE={mse:.6f} | evo_log_loss={evovaq_evo_loss:.4f} | '
            f'final_log_loss={evovaq_final_loss:.4f}'
        )

        qml_evovaq = train_model(
            qml_evovaq,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            epochs=evovaq_adapt_epochs,
            batch_size=batch_size_train,
            lr=evovaq_adapt_lr,
            device=device,
            trainable_params=list(qml_evovaq.pre_pqc.parameters()) + list(qml_evovaq.fc.parameters()),
            freeze_params=qml_evovaq.q_layer.parameters(),
            label_prefix='QML EVOVAQ+Adaptation'
        )

        qml_evovaq_adapted = copy.deepcopy(qml_evovaq)

        adapted_evo_loss = evaluate_qml_log_loss(
            qml_evovaq_adapted,
            x_evo_val,
            y_evo_val,
            device
        )

        adapted_final_loss = evaluate_qml_log_loss(
            qml_evovaq_adapted,
            x_final_val,
            y_final_val,
            device
        )

        prob_adapted = predict_proba_qml(
            qml_evovaq_adapted,
            x_final_val,
            device
        )

        acc, f1m, mse = add_standard_result(
            results,
            'QML EVOVAQ+Adaptation',
            y_final_val,
            prob_adapted
        )

        safe_models['QML EVOVAQ+Adaptation'] = {
            'model': qml_evovaq_adapted,
            'prob': prob_adapted
        }

        print(
            f'QML EVOVAQ+Adaptation | ACC={acc:.4f} | F1={f1m:.4f} | '
            f'MSE={mse:.6f} | evo_log_loss={adapted_evo_loss:.4f} | '
            f'final_log_loss={adapted_final_loss:.4f}'
        )

        detail_rows.append(
            {
                'fold': fold,
                'adam_evo_log_loss': adam_evo_loss,
                'adam_final_log_loss': adam_final_loss,
                'initial_evo_log_loss': initial_evo_loss,
                'initial_final_log_loss': initial_final_loss,
                'evovaq_evo_log_loss': evovaq_evo_loss,
                'evovaq_final_log_loss': evovaq_final_loss,
                'adapted_evo_log_loss': adapted_evo_loss,
                'adapted_final_log_loss': adapted_final_loss,
                'evovaq_best_fitness': evovaq_info['best_fitness'],
                'evovaq_nfev': evovaq_info['nfev'],
                'evovaq_gen': evovaq_info['gen'],
                'evovaq_time_sec': evovaq_info['time_sec'],
                'evovaq_bounds_delta': evovaq_info['bounds_delta'],
                'evovaq_cxpb': evovaq_info['cxpb'],
                'evovaq_mutpb': evovaq_info['mutpb'],
                'evovaq_mutation_sigma': evovaq_info['mutation_sigma'],
                'evovaq_mut_indpb': evovaq_info['mut_indpb']
            }
        )

        fold_safe_store = evaluate_qml_safe_metrics(
            safe_models=safe_models,
            x_final_val=x_final_val,
            y_final_val=y_final_val,
            feature_names=feature_names,
            class_order=class_order,
            n_segments=n_segments,
            noise_levels=noise_levels,
            rge_n_steps=rge_n_steps,
            rge_masking_method=rge_masking_method,
            rge_baseline=rge_baseline,
            fig_dir=fig_dir,
            fold=fold,
            seed=seed,
            device=device
        )

        for model_name in model_names:
            for key in safe_store_all[model_name]:
                safe_store_all[model_name][key].extend(
                    fold_safe_store[model_name][key]
                )

    print('\n================ OVERALL STANDARD ================')

    for model_name, metrics in results.items():
        print(
            f'{model_name:>10} | '
            f'ACC={np.mean(metrics["acc"]):.4f} | '
            f'F1={np.mean(metrics["f1"]):.4f} | '
            f'MSE={np.mean(metrics["mse"]):.6f}'
        )

    print('\n================ OVERALL SAFE ================')

    rows = []
    details_df = pd.DataFrame(detail_rows)

    for model_name in model_names:
        rga_mean = np.nanmean(safe_store_all[model_name]['rga_full'])
        aurga_mean = np.nanmean(safe_store_all[model_name]['aurga'])
        aurgr_mean = np.nanmean(safe_store_all[model_name]['aurgr'])
        aurge_mean = np.nanmean(safe_store_all[model_name]['aurge'])

        print(
            f'{model_name:>10} | '
            f'RGA={rga_mean:.4f} | '
            f'AURGA={aurga_mean:.4f} | '
            f'AURGR={aurgr_mean:.4f} | '
            f'AURGE={aurge_mean:.4f}'
        )

        row = {
            'model': model_name,
            'ACC': float(np.mean(results[model_name]['acc'])),
            'F1': float(np.mean(results[model_name]['f1'])),
            'MSE': float(np.mean(results[model_name]['mse'])),
            'RGA': float(rga_mean),
            'AURGA': float(aurga_mean),
            'AURGR': float(aurgr_mean),
            'AURGE': float(aurge_mean)
        }

        rows.append(row)

    selected_fold_suffix = f'_fold{args.fold}' if args.fold is not None else '_all_folds'

    summary_csv = csv_dir / f'hmda_qml_evovaq_summary{selected_fold_suffix}.csv'
    details_csv = csv_dir / f'hmda_qml_evovaq_details{selected_fold_suffix}.csv'

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    details_df.to_csv(details_csv, index=False)

    print(f'\nSaved summary to: {summary_csv}')
    print(f'Saved details to: {details_csv}')


if __name__ == '__main__':
    main()
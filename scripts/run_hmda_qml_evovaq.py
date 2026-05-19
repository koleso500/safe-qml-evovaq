import argparse
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import yaml
from sklearn.linear_model import LogisticRegression
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


def extend_safe_store(safe_store_all, fold_safe_store):
    for model_name, metrics in fold_safe_store.items():
        for key in safe_store_all[model_name]:
            safe_store_all[model_name][key].extend(metrics[key])


def mean_value(values):
    return float(np.nanmean(values))


def std_value(values):
    values = np.asarray(values, dtype=float)

    if len(values) <= 1:
        return 0.0

    return float(np.nanstd(values, ddof=1))


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
    evovaq_seeds = evovaq_cfg['seeds']

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
        'Logistic Regression',
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

        print('\n--- Logistic Regression ---')

        log_model = LogisticRegression(
            C=1.0,
            solver='lbfgs',
            class_weight='balanced',
            max_iter=1000,
            random_state=seed + fold
        )

        log_model.fit(x_train_inner, y_train_inner)
        prob_logreg = log_model.predict_proba(x_final_val)

        acc, f1m, mse = add_standard_result(
            results,
            'Logistic Regression',
            y_final_val,
            prob_logreg
        )

        print(
            f'Logistic Regression | ACC={acc:.4f} | F1={f1m:.4f} | '
            f'MSE={mse:.6f}'
        )

        base_model = QuantumClassifier(
            input_dim=input_dim,
            num_classes=n_classes,
            n_qubits=n_qubits,
            n_layers=n_layers
        ).to(device)

        print('\n--- QML Adam ---')

        qml_adam = copy.deepcopy(base_model)

        with qml.Tracker(qml_adam.q_device) as adam_tracker:
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

        print('Adam tracker total:', adam_tracker.totals)

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

        print(
            f'QML Adam | ACC={acc:.4f} | F1={f1m:.4f} | '
            f'MSE={mse:.6f} | evo_log_loss={adam_evo_loss:.4f} | '
            f'final_log_loss={adam_final_loss:.4f}'
        )

        safe_models_base = {
            'Logistic Regression': {
                'model': log_model,
                'prob': prob_logreg,
                'model_type': 'sklearn'
            },
            'QML Adam': {
                'model': qml_adam,
                'prob': prob_adam,
                'model_type': 'pytorch'
            }
        }

        fold_safe_store = evaluate_qml_safe_metrics(
            safe_models=safe_models_base,
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
            fold=f'{fold}_base',
            seed=seed,
            device=device
        )

        extend_safe_store(
            safe_store_all,
            fold_safe_store
        )

        print('\n--- QML EVOVAQ seed statistics ---')

        qml_evovaq_start = copy.deepcopy(base_model)

        qml_evovaq_start = train_model(
            qml_evovaq_start,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            epochs=epochs,
            batch_size=batch_size_train,
            lr=lr_torch,
            device=device,
            trainable_params=list(qml_evovaq_start.pre_pqc.parameters()) + list(qml_evovaq_start.fc.parameters()),
            freeze_params=qml_evovaq_start.q_layer.parameters(),
            label_prefix='QML EVOVAQ initial training'
        )

        initial_evovaq_state = copy.deepcopy(qml_evovaq_start.state_dict())

        initial_evo_loss = evaluate_qml_log_loss(
            qml_evovaq_start,
            x_evo_val,
            y_evo_val,
            device
        )

        initial_final_loss = evaluate_qml_log_loss(
            qml_evovaq_start,
            x_final_val,
            y_final_val,
            device
        )

        for evovaq_seed in evovaq_seeds:
            evovaq_seed = int(evovaq_seed)

            print(f'\n--- EVOVAQ seed {evovaq_seed} ---')

            qml_evovaq = copy.deepcopy(base_model)
            qml_evovaq.load_state_dict(initial_evovaq_state)

            qml_evovaq, evovaq_info = train_quantum_with_evovaq(
                qml_evovaq,
                x_evo_val,
                y_evo_val,
                seed=evovaq_seed,
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

            print(
                f'QML EVOVAQ seed {evovaq_seed} | '
                f'ACC={acc:.4f} | F1={f1m:.4f} | MSE={mse:.6f} | '
                f'evo_log_loss={evovaq_evo_loss:.4f} | '
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
                label_prefix=f'QML EVOVAQ+Adaptation seed {evovaq_seed}'
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

            acc_adapt, f1_adapt, mse_adapt = add_standard_result(
                results,
                'QML EVOVAQ+Adaptation',
                y_final_val,
                prob_adapted
            )

            print(
                f'QML EVOVAQ+Adaptation seed {evovaq_seed} | '
                f'ACC={acc_adapt:.4f} | F1={f1_adapt:.4f} | MSE={mse_adapt:.6f} | '
                f'evo_log_loss={adapted_evo_loss:.4f} | '
                f'final_log_loss={adapted_final_loss:.4f}'
            )

            safe_models_seed = {
                'QML EVOVAQ': {
                    'model': qml_after_evovaq,
                    'prob': prob_evovaq,
                    'model_type': 'pytorch'
                },
                'QML EVOVAQ+Adaptation': {
                    'model': qml_evovaq_adapted,
                    'prob': prob_adapted,
                    'model_type': 'pytorch'
                }
            }

            fold_safe_store = evaluate_qml_safe_metrics(
                safe_models=safe_models_seed,
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
                fold=f'{fold}_seed{evovaq_seed}',
                seed=seed,
                device=device
            )

            extend_safe_store(
                safe_store_all,
                fold_safe_store
            )

            evovaq_rga = fold_safe_store['QML EVOVAQ']['rga_full'][-1]
            evovaq_aurga = fold_safe_store['QML EVOVAQ']['aurga'][-1]
            evovaq_aurgr = fold_safe_store['QML EVOVAQ']['aurgr'][-1]
            evovaq_aurge = fold_safe_store['QML EVOVAQ']['aurge'][-1]

            adapted_rga = fold_safe_store['QML EVOVAQ+Adaptation']['rga_full'][-1]
            adapted_aurga = fold_safe_store['QML EVOVAQ+Adaptation']['aurga'][-1]
            adapted_aurgr = fold_safe_store['QML EVOVAQ+Adaptation']['aurgr'][-1]
            adapted_aurge = fold_safe_store['QML EVOVAQ+Adaptation']['aurge'][-1]

            detail_rows.append(
                {
                    'fold': fold,
                    'evovaq_seed': evovaq_seed,
                    'adam_evo_log_loss': adam_evo_loss,
                    'adam_final_log_loss': adam_final_loss,
                    'initial_evo_log_loss': initial_evo_loss,
                    'initial_final_log_loss': initial_final_loss,
                    'evovaq_acc': acc,
                    'evovaq_f1': f1m,
                    'evovaq_mse': mse,
                    'evovaq_evo_log_loss': evovaq_evo_loss,
                    'evovaq_final_log_loss': evovaq_final_loss,
                    'evovaq_rga': evovaq_rga,
                    'evovaq_aurga': evovaq_aurga,
                    'evovaq_aurgr': evovaq_aurgr,
                    'evovaq_aurge': evovaq_aurge,
                    'adapted_acc': acc_adapt,
                    'adapted_f1': f1_adapt,
                    'adapted_mse': mse_adapt,
                    'adapted_evo_log_loss': adapted_evo_loss,
                    'adapted_final_log_loss': adapted_final_loss,
                    'adapted_rga': adapted_rga,
                    'adapted_aurga': adapted_aurga,
                    'adapted_aurgr': adapted_aurgr,
                    'adapted_aurge': adapted_aurge,
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

    print('\n================ OVERALL STANDARD ================')

    for model_name, metrics in results.items():
        print(
            f'{model_name:>10} | '
            f'ACC={mean_value(metrics["acc"]):.4f} ± {std_value(metrics["acc"]):.4f} | '
            f'F1={mean_value(metrics["f1"]):.4f} ± {std_value(metrics["f1"]):.4f} | '
            f'MSE={mean_value(metrics["mse"]):.6f} ± {std_value(metrics["mse"]):.6f}'
        )

    print('\n================ OVERALL SAFE ================')

    rows = []
    details_df = pd.DataFrame(detail_rows)

    for model_name in results:
        rga_mean = mean_value(safe_store_all[model_name]['rga_full'])
        rga_std = std_value(safe_store_all[model_name]['rga_full'])

        aurga_mean = mean_value(safe_store_all[model_name]['aurga'])
        aurga_std = std_value(safe_store_all[model_name]['aurga'])

        aurgr_mean = mean_value(safe_store_all[model_name]['aurgr'])
        aurgr_std = std_value(safe_store_all[model_name]['aurgr'])

        aurge_mean = mean_value(safe_store_all[model_name]['aurge'])
        aurge_std = std_value(safe_store_all[model_name]['aurge'])

        print(
            f'{model_name:>10} | '
            f'RGA={rga_mean:.4f} ± {rga_std:.4f} | '
            f'AURGA={aurga_mean:.4f} ± {aurga_std:.4f} | '
            f'AURGR={aurgr_mean:.4f} ± {aurgr_std:.4f} | '
            f'AURGE={aurge_mean:.4f} ± {aurge_std:.4f}'
        )

        row = {
            'model': model_name,
            'ACC_mean': mean_value(results[model_name]['acc']),
            'ACC_std': std_value(results[model_name]['acc']),
            'F1_mean': mean_value(results[model_name]['f1']),
            'F1_std': std_value(results[model_name]['f1']),
            'MSE_mean': mean_value(results[model_name]['mse']),
            'MSE_std': std_value(results[model_name]['mse']),
            'RGA_mean': rga_mean,
            'RGA_std': rga_std,
            'AURGA_mean': aurga_mean,
            'AURGA_std': aurga_std,
            'AURGR_mean': aurgr_mean,
            'AURGR_std': aurgr_std,
            'AURGE_mean': aurge_mean,
            'AURGE_std': aurge_std
        }

        rows.append(row)

    if 'evovaq_seed' in details_df.columns:
        seed_stats = details_df[
            [
                'evovaq_acc',
                'evovaq_f1',
                'evovaq_mse',
                'evovaq_final_log_loss',
                'evovaq_rga',
                'evovaq_aurga',
                'evovaq_aurgr',
                'evovaq_aurge',
                'adapted_acc',
                'adapted_f1',
                'adapted_mse',
                'adapted_final_log_loss',
                'adapted_rga',
                'adapted_aurga',
                'adapted_aurgr',
                'adapted_aurge',
                'evovaq_best_fitness',
                'evovaq_time_sec'
            ]
        ].agg(['mean', 'std'])

        print('\n================ EVOVAQ SEED STATISTICS ================')
        print(seed_stats)

    selected_fold_suffix = f'_fold{args.fold}' if args.fold is not None else '_all_folds'

    summary_csv = csv_dir / f'hmda_qml_evovaq_summary{selected_fold_suffix}.csv'
    details_csv = csv_dir / f'hmda_qml_evovaq_details{selected_fold_suffix}.csv'

    pd.DataFrame(rows).to_csv(summary_csv, index=False)
    details_df.to_csv(details_csv, index=False)

    print(f'\nSaved summary to: {summary_csv}')
    print(f'Saved details to: {details_csv}')


if __name__ == '__main__':
    main()
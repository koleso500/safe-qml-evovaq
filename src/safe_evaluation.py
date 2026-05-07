import numpy as np
import torch

from safe.rga import compare_models_rga
from safe.rge import compare_models_rge_tabular
from safe.rgr import compare_models_rgr


def evaluate_qml_safe_metrics(
    safe_models,
    x_final_val,
    y_final_val,
    feature_names,
    class_order,
    n_segments,
    noise_levels,
    rge_n_steps,
    rge_masking_method,
    rge_baseline,
    fig_dir,
    fold,
    seed,
    device
):
    model_names = list(safe_models.keys())

    safe_store = {
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

    models_rga = {
        model_name: (model_info['prob'], class_order)
        for model_name, model_info in safe_models.items()
    }

    results_rga = compare_models_rga(
        models_rga,
        y_labels=y_final_val,
        n_segments=n_segments,
        fig_size=(8, 6),
        verbose=True,
        save_path=str(fig_dir / f'rga_fold{fold}_qml.png')
    )

    rga_dict = {
        model_name: float(results_rga[model_name]['rga_full'])
        for model_name in model_names
    }

    for model_name in model_names:
        safe_store[model_name]['rga_curve'].append(
            np.asarray(results_rga[model_name]['curve_model'], float)
        )
        safe_store[model_name]['rga_full'].append(
            float(results_rga[model_name]['rga_full'])
        )

        aurga_value = results_rga[model_name].get(
            'aurga_normalized_to_perfect',
            results_rga[model_name].get('aurga')
        )

        safe_store[model_name]['aurga'].append(float(aurga_value))

    x_final_val_t = torch.tensor(x_final_val, dtype=torch.float32)

    models_rgr = {
        model_name: (
            model_info['model'],
            x_final_val_t,
            model_info['prob'],
            class_order,
            'pytorch',
            device
        )
        for model_name, model_info in safe_models.items()
    }

    results_rgr = compare_models_rgr(
        models_dict=models_rgr,
        noise_levels=noise_levels,
        class_order=class_order,
        rga_dict=rga_dict,
        fig_size=(8, 6),
        verbose=True,
        random_seed=seed,
        save_path=str(fig_dir / f'rgr_fold{fold}_qml.png')
    )

    for model_name in model_names:
        safe_store[model_name]['rgr_curve'].append(
            np.asarray(results_rgr[model_name]['rgr_rescaled'], float)
        )
        safe_store[model_name]['aurgr'].append(
            float(results_rgr[model_name]['aurgr'])
        )

    models_rge = {
        model_name: (
            model_info['model'],
            x_final_val,
            feature_names,
            model_info['prob'],
            class_order,
            'pytorch',
            device
        )
        for model_name, model_info in safe_models.items()
    }

    results_rge = compare_models_rge_tabular(
        models_dict=models_rge,
        class_order=class_order,
        rga_dict=rga_dict,
        verbose=True,
        random_seed=seed,
        fig_size=(8, 6),
        save_path=str(fig_dir / f'rge_fold{fold}_qml.png'),
        masking_method=rge_masking_method,
        baseline=rge_baseline,
        n_steps=rge_n_steps
    )

    for model_name in model_names:
        safe_store[model_name]['rge_curve'].append(
            np.asarray(results_rge[model_name]['rge_rescaled'], float)
        )
        safe_store[model_name]['aurge'].append(
            float(results_rge[model_name]['aurge'])
        )

    return safe_store
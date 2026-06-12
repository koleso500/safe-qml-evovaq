import numpy as np
import torch

from safeai.rga import compare_rga
from safeai.rge import compare_rge
from safeai.rgr import compare_rgr


def evaluate_qml_safe_metrics(
    safe_models,
    x_final_test,
    y_final_test,
    feature_names,
    class_order,
    n_segments,
    noise_levels,
    rge_n_steps,
    rge_masking_method,
    rge_baseline,
    fig_dir,
    run_label,
    seed,
    device,
    rga_curve_method='removal'
):
    """
    Evaluation with SAFE-AI metrics.
    """
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
        model_name: (
            model_info['prob'],
            model_info.get('class_order', class_order)
        )
        for model_name, model_info in safe_models.items()
    }

    results_rga = compare_rga(
        models_rga,
        y_true=y_final_test,
        n_segments=n_segments,
        positive_class=1,
        curve_method=rga_curve_method,
        normalize_to_perfect=True,
        verbose=True,
        save_path=str(fig_dir / f'rga_{run_label}.png')
    )

    rga_dict = {
        model_name: float(results_rga[model_name]['rga'])
        for model_name in model_names
    }

    for model_name in model_names:
        safe_store[model_name]['rga_curve'].append(
            np.asarray(results_rga[model_name]['curve'], dtype=float)
        )
        safe_store[model_name]['rga_full'].append(
            float(results_rga[model_name]['rga'])
        )
        safe_store[model_name]['aurga'].append(
            float(results_rga[model_name]['aurga'])
        )

    x_final_test_t = torch.tensor(x_final_test, dtype=torch.float32)

    models_rgr = {}

    for model_name, model_info in safe_models.items():
        model_type = model_info.get('model_type', 'pytorch')

        if model_type == 'sklearn':
            models_rgr[model_name] = (
                model_info['model'],
                x_final_test,
                model_info['prob'],
                model_info['model'].classes_,
                'sklearn',
                None
            )

        elif model_type == 'pytorch':
            models_rgr[model_name] = (
                model_info['model'],
                x_final_test_t,
                model_info['prob'],
                model_info.get('class_order', class_order),
                'pytorch',
                device
            )

        else:
            raise ValueError(
                f'Unknown model_type for {model_name}: {model_type}'
            )

    results_rgr = compare_rgr(
        models=models_rgr,
        strengths=noise_levels,
        class_order=class_order,
        method='noise',
        rga_dict=rga_dict,
        fig_size=(8, 6),
        verbose=True,
        random_seed=seed,
        save_path=str(fig_dir / f'rgr_{run_label}.png')
    )

    for model_name in model_names:
        safe_store[model_name]['rgr_curve'].append(
            np.asarray(results_rgr[model_name]['rgr_rescaled'], dtype=float)
        )
        safe_store[model_name]['aurgr'].append(
            float(results_rgr[model_name]['aurgr'])
        )

    models_rge = {}

    for model_name, model_info in safe_models.items():
        model_type = model_info.get('model_type', 'pytorch')

        if model_type == 'sklearn':
            models_rge[model_name] = (
                model_info['model'],
                x_final_test,
                feature_names,
                model_info['prob'],
                model_info['model'].classes_,
                'sklearn',
                None
            )

        elif model_type == 'pytorch':
            models_rge[model_name] = (
                model_info['model'],
                x_final_test,
                feature_names,
                model_info['prob'],
                model_info.get('class_order', class_order),
                'pytorch',
                device
            )

        else:
            raise ValueError(
                f'Unknown model_type for {model_name}: {model_type}'
            )

    results_rge = compare_rge(
        method='tabular',
        models=models_rge,
        class_order=class_order,
        rga_dict=rga_dict,
        verbose=True,
        random_seed=seed,
        fig_size=(8, 6),
        save_path=str(fig_dir / f'rge_{run_label}.png'),
        masking_method=rge_masking_method,
        baseline=rge_baseline,
        n_steps=rge_n_steps
    )

    for model_name in model_names:
        safe_store[model_name]['rge_curve'].append(
            np.asarray(results_rge[model_name]['rge_rescaled'], dtype=float)
        )
        safe_store[model_name]['aurge'].append(
            float(results_rge[model_name]['aurge'])
        )

    return safe_store
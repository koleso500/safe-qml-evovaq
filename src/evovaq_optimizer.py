import random
import time
from functools import partial

import numpy as np
import pennylane as qml
from evovaq.GeneticAlgorithm import GA
from evovaq.problem import Problem
from evovaq.tools.operators import cx_uniform, mut_gaussian, sel_tournament
from sklearn.metrics import log_loss

from src.qml_model import get_quantum_weights_flat, predict_proba_qml


def evovaq_cost_function(params, qml_model, eval_data, eval_labels, device, evovaq_cfg):
    old_weights = get_quantum_weights_flat(qml_model).copy()

    try:
        qml_model.set_quantum_weights_flat(params)
        qml_model.eval()

        probs_clean = predict_proba_qml(qml_model, eval_data, device)

        clean_loss = float(
            log_loss(
                eval_labels,
                probs_clean,
                labels=np.arange(qml_model.fc.out_features)
            )
        )

        fitness_mode = evovaq_cfg['fitness_mode']

        if fitness_mode == 'log_loss':
            return clean_loss

        if fitness_mode == 'robust_log_loss':
            alpha = float(evovaq_cfg['robustness_alpha'])
            sigma = float(evovaq_cfg['robustness_sigma'])

            noise = np.random.normal(
                loc=0.0,
                scale=sigma,
                size=eval_data.shape
            ).astype(np.float32)

            eval_data_noisy = eval_data + noise
            probs_noisy = predict_proba_qml(qml_model, eval_data_noisy, device)

            stability_penalty = float(
                np.mean((probs_clean - probs_noisy) ** 2)
            )

            return clean_loss + alpha * stability_penalty

        raise ValueError(f'Unknown EVOVAQ fitness_mode: {fitness_mode}')

    finally:
        qml_model.set_quantum_weights_flat(old_weights)


def make_initial_pop_around_current_weights(model, pop_size, sigma, param_bounds=None):
    center = get_quantum_weights_flat(model).astype(np.float32)
    n_params = center.size

    initial_pop = np.zeros((pop_size, n_params), dtype=np.float32)
    initial_pop[0] = center

    for i in range(1, pop_size):
        noise = np.random.normal(0.0, sigma, size=n_params).astype(np.float32)
        initial_pop[i] = center + noise

        if param_bounds is not None:
            lows = np.array([b[0] for b in param_bounds], dtype=np.float32)
            highs = np.array([b[1] for b in param_bounds], dtype=np.float32)
            initial_pop[i] = np.clip(initial_pop[i], lows, highs)

    return initial_pop


def train_quantum_with_evovaq(
    qml_model,
    x_evo_val,
    y_evo_val,
    seed,
    n_layers,
    n_qubits,
    evovaq_cfg,
    device
):
    for param in qml_model.pre_pqc.parameters():
        param.requires_grad = False

    for param in qml_model.fc.parameters():
        param.requires_grad = False

    n_params = n_layers * n_qubits * 3

    pop_size = int(evovaq_cfg['population_size'])
    max_gen = int(evovaq_cfg['max_gen'])
    init_sigma = float(evovaq_cfg['init_sigma'])
    bounds_delta = float(evovaq_cfg['bounds_delta'])

    cxpb = float(evovaq_cfg['crossover_probability'])
    cx_indpb = float(evovaq_cfg['crossover_individual_probability'])
    mutpb = float(evovaq_cfg['mutation_probability'])
    mutation_mu = float(evovaq_cfg['mutation_mu'])
    mutation_sigma = float(evovaq_cfg['mutation_sigma'])
    mut_indpb = float(evovaq_cfg['mutation_individual_probability'])
    tournsize = int(evovaq_cfg['tournament_size'])

    max_nfev = evovaq_cfg['max_nfev']
    if max_nfev is not None:
        max_nfev = int(max_nfev)

    cost_func = partial(
        evovaq_cost_function,
        qml_model=qml_model,
        eval_data=x_evo_val,
        eval_labels=y_evo_val,
        device=device,
        evovaq_cfg=evovaq_cfg
    )

    center = get_quantum_weights_flat(qml_model).astype(np.float32)

    param_bounds = [
        (float(c - bounds_delta), float(c + bounds_delta))
        for c in center
    ]

    problem = Problem(
        n_params=n_params,
        param_bounds=param_bounds,
        obj_function=cost_func
    )

    optimizer = GA(
        selection=sel_tournament,
        crossover=cx_uniform,
        mutation=mut_gaussian,
        elitism=True,
        cxpb=cxpb,
        mutpb=mutpb,
        tournsize=tournsize,
        cx_indpb=cx_indpb,
        mu=mutation_mu,
        sigma=mutation_sigma,
        mut_indpb=mut_indpb
    )

    np.random.seed(seed)
    random.seed(seed)

    initial_pop = make_initial_pop_around_current_weights(
        qml_model,
        pop_size=pop_size,
        sigma=init_sigma,
        param_bounds=param_bounds
    )

    start = time.time()

    with qml.Tracker(qml_model.q_device) as tracker:
        res = optimizer.optimize(
            problem,
            pop_size=pop_size,
            initial_pop=initial_pop,
            max_nfev=max_nfev,
            max_gen=max_gen,
            seed=seed,
            n_run=1,
            verbose=True
        )

    end = time.time()

    print('PennyLane tracker total:', tracker.totals)

    qml_model.set_quantum_weights_flat(res.x)

    evovaq_info = {
        'best_fitness': float(res.fun),
        'nfev': int(res.nfev),
        'gen': int(res.gen),
        'time_sec': float(end - start),
        'bounds_delta': float(bounds_delta),
        'cxpb': float(cxpb),
        'cx_indpb': float(cx_indpb),
        'mutpb': float(mutpb),
        'mutation_sigma': float(mutation_sigma),
        'mut_indpb': float(mut_indpb),
        'fitness_mode': evovaq_cfg['fitness_mode'],
        'robustness_alpha': float(evovaq_cfg['robustness_alpha']),
        'robustness_sigma': float(evovaq_cfg['robustness_sigma'])
    }

    for param in qml_model.pre_pqc.parameters():
        param.requires_grad = True

    for param in qml_model.fc.parameters():
        param.requires_grad = True

    print(f'EVOVAQ final best fitness: {float(res.fun):.6f}')
    print(f'EVOVAQ nfev: {res.nfev}')
    print(f'EVOVAQ generations: {res.gen}')
    print(f'EVOVAQ training time: {end - start:.3f} sec')

    return qml_model, evovaq_info
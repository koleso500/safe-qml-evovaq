import copy
import random
import time
import argparse
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
import torch
import torch.nn as nn
import yaml
from evovaq.GeneticAlgorithm import GA
from evovaq.problem import Problem
from evovaq.tools.operators import cx_uniform, mut_gaussian, sel_tournament
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_squared_error
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from safe.rga import compare_models_rga
from safe.rge import compare_models_rge_tabular
from safe.rgr import compare_models_rgr


PROJECT_ROOT = Path(__file__).resolve().parent
config_path = PROJECT_ROOT / 'config.yaml'

with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

SEED = int(config['seed'])
N_SPLITS = int(config['cross_validation']['n_splits'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

FIG_DIR = PROJECT_ROOT / config['paths']['fig_hmda_dir']
FIG_DIR.mkdir(parents=True, exist_ok=True)

CSV_DIR = PROJECT_ROOT / config['paths']['csv_dir']
CSV_DIR.mkdir(parents=True, exist_ok=True)


EPOCHS = int(config.get('training', {}).get('epochs', 20))
BATCH_SIZE_TRAIN = int(config.get('training', {}).get('batch_size', 64))
LR_TORCH = float(config.get('training', {}).get('learning_rate', 3e-3))

N_QUBITS = int(config.get('quantum', {}).get('n_qubits', 4))
N_LAYERS = int(config.get('quantum', {}).get('n_layers', 2))

evovaq_cfg = config.get('evovaq', {})

EVOVAQ_POP_SIZE = int(evovaq_cfg.get('population_size', 16))
EVOVAQ_MAX_GEN = int(evovaq_cfg.get('max_gen', 20))
EVOVAQ_BATCH_SIZE = int(evovaq_cfg.get('batch_size', BATCH_SIZE_TRAIN))
EVOVAQ_ACCEPT_ONLY_IF_BETTER = bool(evovaq_cfg.get('accept_only_if_better', True))
EVOVAQ_INIT_SIGMA = float(evovaq_cfg.get('init_sigma', 0.03))

EVOVAQ_CXPB = float(evovaq_cfg.get('crossover_probability', 0.0))
EVOVAQ_CX_INDPB = float(evovaq_cfg.get('crossover_individual_probability', 0.5))
EVOVAQ_MUTPB = float(evovaq_cfg.get('mutation_probability', 0.5))
EVOVAQ_MUT_MU = float(evovaq_cfg.get('mutation_mu', 0.0))
EVOVAQ_MUT_SIGMA = float(evovaq_cfg.get('mutation_sigma', 0.03))
EVOVAQ_MUT_INDPB = float(evovaq_cfg.get('mutation_individual_probability', 0.15))
EVOVAQ_TOURNSIZE = int(evovaq_cfg.get('tournament_size', 3))

EVOVAQ_MAX_NFEV = evovaq_cfg.get('max_nfev', None)
if EVOVAQ_MAX_NFEV is not None:
    EVOVAQ_MAX_NFEV = int(EVOVAQ_MAX_NFEV)

EVOVAQ_READAPT_EPOCHS = int(evovaq_cfg.get('readapt_epochs', 5))
EVOVAQ_READAPT_LR = float(evovaq_cfg.get('readapt_lr', LR_TORCH * 0.5))
INNER_EVO_VAL_SIZE = float(evovaq_cfg.get('inner_evo_val_size', 0.2))

N_SEGMENTS = 10
noise_levels = np.linspace(0.0, 1.0, 11)

RGE_N_STEPS = config.get('rge_tabular_n_steps', None)
RGE_MASKING_METHOD = config.get('rge_tabular_masking_method', 'greedy')
RGE_BASELINE = config.get('rge_tabular_baseline', 'mean')

dev = qml.device('default.qubit', wires=N_QUBITS)


@qml.qnode(dev, interface='torch')
def vqc_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), normalize=True)
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]


weight_shapes = {'weights': (N_LAYERS, N_QUBITS, 3)}


class QuantumClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.pre_pqc = nn.Linear(input_dim, 2 ** N_QUBITS)
        self.q_layer = qml.qnn.TorchLayer(vqc_circuit, weight_shapes)
        self.fc = nn.Linear(N_QUBITS, num_classes)

    def forward(self, inputs):
        inputs = self.pre_pqc(inputs)
        inputs = torch.nn.functional.gelu(inputs)
        inputs = self.q_layer(inputs)
        return self.fc(inputs)

    def set_quantum_weights_flat(self, flat_params):
        weights = np.asarray(flat_params, dtype=np.float32).reshape(
            N_LAYERS,
            N_QUBITS,
            3,
        )

        with torch.no_grad():
            self.q_layer.weights.copy_(
                torch.tensor(
                    weights,
                    dtype=self.q_layer.weights.dtype,
                    device=self.q_layer.weights.device,
                )
            )


def predict_proba_qml(model, x_data):
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(x_data, dtype=torch.float32, device=device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    return probs


def compute_metrics(y_true: np.ndarray, probs: np.ndarray):
    preds = np.argmax(probs, axis=1)
    acc_value = accuracy_score(y_true, preds)
    f1_value = f1_score(y_true, preds, average='macro')
    onehot = np.eye(probs.shape[1], dtype=float)[y_true]
    mse_value = mean_squared_error(onehot, probs)
    return acc_value, f1_value, mse_value


def evaluate_torch_loss(model, x_eval, y_eval):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        x_tensor = torch.tensor(x_eval, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y_eval, dtype=torch.long, device=device)
        loss = loss_fn(model(x_tensor), y_tensor)

    return float(loss.item())


def evaluate_qml_log_loss(model, x_eval, y_eval):
    probs = predict_proba_qml(model, x_eval)

    return float(
        log_loss(
            y_eval,
            probs,
            labels=np.arange(probs.shape[1]),
        )
    )


def print_loss_diagnostics(
    model,
    x_train_inner,
    y_train_inner,
    x_evo_val,
    y_evo_val,
    x_final_val,
    y_final_val,
    label,
):
    train_ce = evaluate_torch_loss(model, x_train_inner, y_train_inner)
    evo_ce = evaluate_torch_loss(model, x_evo_val, y_evo_val)
    final_ce = evaluate_torch_loss(model, x_final_val, y_final_val)

    train_log = evaluate_qml_log_loss(model, x_train_inner, y_train_inner)
    evo_log = evaluate_qml_log_loss(model, x_evo_val, y_evo_val)
    final_log = evaluate_qml_log_loss(model, x_final_val, y_final_val)

    print(
        f'{label} CE losses | '
        f'train_inner={train_ce:.4f} | '
        f'evo_val={evo_ce:.4f} | '
        f'final_val={final_ce:.4f}'
    )

    print(
        f'{label} log_loss | '
        f'train_inner={train_log:.4f} | '
        f'evo_val={evo_log:.4f} | '
        f'final_val={final_log:.4f}'
    )

    return {
        'train_ce': train_ce,
        'evo_ce': evo_ce,
        'final_ce': final_ce,
        'train_log_loss': train_log,
        'evo_log_loss': evo_log,
        'final_log_loss': final_log,
    }


def train_torch_model(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    epochs=EPOCHS,
    lr=LR_TORCH,
    label_prefix='Adam',
):
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

            running += loss.item()

        train_loss = running / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.long, device=device)
            val_loss = loss_fn(model(xv), yv).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'{label_prefix} Epoch {epoch + 1:02d} | '
                f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}'
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def train_classical_parts_only(
    model,
    x_train,
    y_train,
    x_val,
    y_val,
    *,
    epochs=EPOCHS,
    lr=LR_TORCH,
    label_prefix='Classical',
):
    for param in model.q_layer.parameters():
        param.requires_grad = False

    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(x_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        ),
        batch_size=BATCH_SIZE_TRAIN,
        shuffle=True,
    )

    opt = torch.optim.Adam(
        list(model.pre_pqc.parameters()) + list(model.fc.parameters()),
        lr=lr,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        running = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

            running += loss.item()

        train_loss = running / max(len(train_loader), 1)

        model.eval()
        with torch.no_grad():
            xv = torch.tensor(x_val, dtype=torch.float32, device=device)
            yv = torch.tensor(y_val, dtype=torch.long, device=device)
            val_loss = loss_fn(model(xv), yv).item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f'{label_prefix} Epoch {epoch + 1:02d} | '
                f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}'
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    for param in model.q_layer.parameters():
        param.requires_grad = True

    return model


def get_quantum_weights_flat(model):
    return model.q_layer.weights.detach().cpu().numpy().reshape(-1)


def evovaq_cost_function(params, qml_model, eval_data, eval_labels):
    old_weights = get_quantum_weights_flat(qml_model).copy()

    try:
        qml_model.set_quantum_weights_flat(params)
        qml_model.eval()

        probs_full = predict_proba_qml(qml_model, eval_data)

        return log_loss(
            eval_labels,
            probs_full,
            labels=np.arange(qml_model.fc.out_features),
        )

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


def train_quantum_with_evovaq(qml_model, x_evo_val, y_evo_val, seed=SEED):
    for param in qml_model.pre_pqc.parameters():
        param.requires_grad = False

    for param in qml_model.fc.parameters():
        param.requires_grad = False

    n_params = N_LAYERS * N_QUBITS * 3

    cost_func = partial(
        evovaq_cost_function,
        qml_model=qml_model,
        eval_data=x_evo_val,
        eval_labels=y_evo_val,
    )

    center = get_quantum_weights_flat(qml_model).astype(np.float32)
    bounds_delta = float(evovaq_cfg.get('bounds_delta', np.pi))

    param_bounds = [
        (float(c - bounds_delta), float(c + bounds_delta))
        for c in center
    ]

    problem = Problem(
        n_params=n_params,
        param_bounds=param_bounds,
        obj_function=cost_func,
    )

    optimizer = GA(
        selection=sel_tournament,
        crossover=cx_uniform,
        mutation=mut_gaussian,
        elitism=True,
        cxpb=EVOVAQ_CXPB,
        mutpb=EVOVAQ_MUTPB,
        tournsize=EVOVAQ_TOURNSIZE,
        cx_indpb=EVOVAQ_CX_INDPB,
        mu=EVOVAQ_MUT_MU,
        sigma=EVOVAQ_MUT_SIGMA,
        mut_indpb=EVOVAQ_MUT_INDPB,
    )

    initial_pop = make_initial_pop_around_current_weights(
        qml_model,
        pop_size=EVOVAQ_POP_SIZE,
        sigma=EVOVAQ_INIT_SIGMA,
        param_bounds=param_bounds,
    )

    current_center = get_quantum_weights_flat(qml_model)
    max_center_diff = np.max(np.abs(initial_pop[0] - current_center))
    print(f'Initial population center max diff: {max_center_diff:.8f}')

    initial_center_fitness = float(cost_func(initial_pop[0]))
    initial_best_fitness = float(np.min([cost_func(ind) for ind in initial_pop]))

    start = time.time()

    res = optimizer.optimize(
        problem,
        pop_size=EVOVAQ_POP_SIZE,
        initial_pop=initial_pop,
        max_nfev=EVOVAQ_MAX_NFEV,
        max_gen=EVOVAQ_MAX_GEN,
        seed=seed,
        n_run=1,
        verbose=True,
    )

    end = time.time()

    qml_model.set_quantum_weights_flat(res.x)

    evovaq_info = {
        'initial_center_fitness': initial_center_fitness,
        'initial_best_fitness': initial_best_fitness,
        'best_fitness': float(res.fun),
        'nfev': int(res.nfev),
        'gen': int(res.gen),
        'time_sec': float(end - start),
        'initial_center_max_diff': float(max_center_diff),
        'bounds_delta': float(bounds_delta),
    }

    print(f'EVOVAQ initial center fitness: {initial_center_fitness:.6f}')
    print(f'EVOVAQ initial best fitness:   {initial_best_fitness:.6f}')
    print(f'EVOVAQ final best fitness:     {float(res.fun):.6f}')
    print(f'EVOVAQ nfev:                   {res.nfev}')
    print(f'EVOVAQ generations:            {res.gen}')
    print(f'EVOVAQ training time:          {end - start:.3f} sec')

    if max_center_diff > 1e-8:
        print(
            'WARNING: initial_pop[0] is not exactly equal to current quantum weights. '
            f'Max diff = {max_center_diff:.8f}.'
        )

    if float(res.fun) > initial_center_fitness + 1e-6:
        print(
            'WARNING: EVOVAQ best fitness is worse than the initial checkpoint. '
            'This should not happen because initial_pop[0] is the current checkpoint.'
        )

    for param in qml_model.pre_pqc.parameters():
        param.requires_grad = True

    for param in qml_model.fc.parameters():
        param.requires_grad = True

    return qml_model, evovaq_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', type=int, required=True)
    args = parser.parse_args()
    selected_fold = args.fold

    hmda_cfg = config.get('hmda', {})
    raw_csv = PROJECT_ROOT / hmda_cfg.get('raw_csv', 'data/raw/hmda.csv')
    target_col = hmda_cfg.get('target', 'action_taken')
    create_sample = bool(hmda_cfg.get('create_sample', False))
    sample_fraction = float(hmda_cfg.get('sample_fraction', 0.1))
    sample_size = hmda_cfg.get('sample_size', None)

    if sample_size is not None:
        sample_size = int(sample_size)

    print('Loading HMDA:', raw_csv)

    df_raw = load_raw_hmda(raw_csv)
    df = clean_hmda_binary(df_raw, target_col=target_col)

    df = maybe_subsample_stratified(
        df,
        target_col,
        create_sample=create_sample,
        sample_fraction=sample_fraction,
        sample_size=sample_size,
        random_state=SEED,
    )

    y = df[target_col].astype(int).to_numpy()
    x_df = df.drop(columns=[target_col])
    feature_names = x_df.columns.to_list()
    x = x_df.to_numpy(dtype=np.float32)

    class_order = np.array([0, 1], dtype=int)
    n_classes = 2

    print('HMDA cleaned shape:', df.shape)
    print('X shape:', x.shape, '| y shape:', y.shape)

    k_fold = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=SEED,
    )

    model_names = ['QML Simple', 'QML EVOVAQ']

    results = {
        m: {'acc': [], 'f1': [], 'mse': []}
        for m in model_names
    }

    safe_store = {
        m: {
            'rga_curve': [],
            'rga_full': [],
            'aurga': [],
            'rgr_curve': [],
            'aurgr': [],
            'rge_curve': [],
            'aurge': [],
        }
        for m in model_names
    }

    diagnostic_rows = []

    for fold, (train_idx, val_idx) in enumerate(k_fold.split(x, y), 1):
        if fold != selected_fold:
            continue

        print(f'\n====== Fold {fold} ======')

        torch.manual_seed(SEED + fold)
        np.random.seed(SEED + fold)
        random.seed(SEED + fold)

        x_train_outer_raw, x_final_val_raw = x[train_idx], x[val_idx]
        y_train_outer, y_final_val = y[train_idx], y[val_idx]

        x_train_inner_raw, x_evo_val_raw, y_train_inner, y_evo_val = train_test_split(
            x_train_outer_raw,
            y_train_outer,
            test_size=INNER_EVO_VAL_SIZE,
            random_state=SEED + fold,
            stratify=y_train_outer,
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

        input_dim = x_train_inner.shape[1]

        print('\n--- QML Simple ---')

        qml_plain = QuantumClassifier(
            input_dim=input_dim,
            num_classes=n_classes,
        ).to(device)

        qml_plain = train_torch_model(
            qml_plain,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            epochs=EPOCHS,
            lr=LR_TORCH,
            label_prefix='QML Simple Adam',
        )

        simple_losses = print_loss_diagnostics(
            qml_plain,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            x_final_val,
            y_final_val,
            label='QML Simple',
        )

        prob_plain = predict_proba_qml(qml_plain, x_final_val)

        acc, f1m, mse = compute_metrics(y_final_val, prob_plain)
        results['QML Simple']['acc'].append(acc)
        results['QML Simple']['f1'].append(f1m)
        results['QML Simple']['mse'].append(mse)

        print(f'QML Simple | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')

        print('\n--- QML EVOVAQ ---')

        qml_evovaq = QuantumClassifier(
            input_dim=input_dim,
            num_classes=n_classes,
        ).to(device)

        print('Stage 1: classical-only Adam warm-up')

        qml_evovaq = train_classical_parts_only(
            qml_evovaq,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            epochs=EPOCHS,
            lr=LR_TORCH,
            label_prefix='Warm-up',
        )

        classical_state = copy.deepcopy(qml_evovaq.state_dict())

        classical_losses = print_loss_diagnostics(
            qml_evovaq,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            x_final_val,
            y_final_val,
            label='Before EVOVAQ',
        )

        classical_evo_loss = classical_losses['evo_log_loss']
        classical_final_loss = classical_losses['final_log_loss']

        print('Stage 2: EVOVAQ quantum fine-tuning')

        qml_evovaq, evovaq_info = train_quantum_with_evovaq(
            qml_evovaq,
            x_evo_val,
            y_evo_val,
            seed=SEED + fold,
        )

        raw_evovaq_losses = print_loss_diagnostics(
            qml_evovaq,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            x_final_val,
            y_final_val,
            label='After raw EVOVAQ',
        )

        raw_evovaq_evo_loss = raw_evovaq_losses['evo_log_loss']

        if raw_evovaq_evo_loss > classical_evo_loss + 1e-6:
            print(
                'WARNING: raw EVOVAQ worsened evo log-loss compared with the checkpoint. '
                'Check fitness consistency or EVOVAQ implementation.'
            )

        if abs(raw_evovaq_evo_loss - evovaq_info['best_fitness']) > 1e-5:
            print(
                'WARNING: raw EVOVAQ diagnostic loss does not match EVOVAQ best fitness. '
                f'diagnostic={raw_evovaq_evo_loss:.6f}, '
                f'res.fun={evovaq_info["best_fitness"]:.6f}'
            )

        print('Stage 3: post-EVOVAQ classical readaptation')

        qml_evovaq = train_classical_parts_only(
            qml_evovaq,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            epochs=EVOVAQ_READAPT_EPOCHS,
            lr=EVOVAQ_READAPT_LR,
            label_prefix='Readapt',
        )

        readapt_losses = print_loss_diagnostics(
            qml_evovaq,
            x_train_inner,
            y_train_inner,
            x_evo_val,
            y_evo_val,
            x_final_val,
            y_final_val,
            label='After EVOVAQ + readaptation',
        )

        readapt_evo_loss = readapt_losses['evo_log_loss']
        readapt_final_loss = readapt_losses['final_log_loss']

        if EVOVAQ_ACCEPT_ONLY_IF_BETTER and readapt_evo_loss > classical_evo_loss:
            print(
                'EVOVAQ rejected: evo_val log-loss worsened '
                f'from {classical_evo_loss:.4f} to {readapt_evo_loss:.4f}. '
                f'Final-val log-loss would be {classical_final_loss:.4f} -> {readapt_final_loss:.4f}.'
            )

            qml_evovaq.load_state_dict(classical_state)
            evovaq_accepted = False
            final_evovaq_losses = classical_losses

        else:
            print(
                'EVOVAQ accepted: evo_val log-loss changed '
                f'from {classical_evo_loss:.4f} to {readapt_evo_loss:.4f}. '
                f'Final-val log-loss changed {classical_final_loss:.4f} -> {readapt_final_loss:.4f}.'
            )

            evovaq_accepted = True
            final_evovaq_losses = readapt_losses

        prob_evovaq = predict_proba_qml(qml_evovaq, x_final_val)

        acc, f1m, mse = compute_metrics(y_final_val, prob_evovaq)
        results['QML EVOVAQ']['acc'].append(acc)
        results['QML EVOVAQ']['f1'].append(f1m)
        results['QML EVOVAQ']['mse'].append(mse)

        print(f'QML EVOVAQ | ACC={acc:.4f}  F1={f1m:.4f}  MSE={mse:.6f}')

        diagnostic_rows.append(
            {
                'fold': fold,
                'evovaq_accepted': evovaq_accepted,

                'simple_train_ce': simple_losses['train_ce'],
                'simple_evo_ce': simple_losses['evo_ce'],
                'simple_final_ce': simple_losses['final_ce'],
                'simple_train_log_loss': simple_losses['train_log_loss'],
                'simple_evo_log_loss': simple_losses['evo_log_loss'],
                'simple_final_log_loss': simple_losses['final_log_loss'],

                'classical_train_ce': classical_losses['train_ce'],
                'classical_evo_ce': classical_losses['evo_ce'],
                'classical_final_ce': classical_losses['final_ce'],
                'classical_train_log_loss': classical_losses['train_log_loss'],
                'classical_evo_log_loss': classical_losses['evo_log_loss'],
                'classical_final_log_loss': classical_losses['final_log_loss'],

                'raw_evovaq_train_ce': raw_evovaq_losses['train_ce'],
                'raw_evovaq_evo_ce': raw_evovaq_losses['evo_ce'],
                'raw_evovaq_final_ce': raw_evovaq_losses['final_ce'],
                'raw_evovaq_train_log_loss': raw_evovaq_losses['train_log_loss'],
                'raw_evovaq_evo_log_loss': raw_evovaq_losses['evo_log_loss'],
                'raw_evovaq_final_log_loss': raw_evovaq_losses['final_log_loss'],

                'readapt_train_ce': readapt_losses['train_ce'],
                'readapt_evo_ce': readapt_losses['evo_ce'],
                'readapt_final_ce': readapt_losses['final_ce'],
                'readapt_train_log_loss': readapt_losses['train_log_loss'],
                'readapt_evo_log_loss': readapt_losses['evo_log_loss'],
                'readapt_final_log_loss': readapt_losses['final_log_loss'],

                'final_evovaq_train_ce': final_evovaq_losses['train_ce'],
                'final_evovaq_evo_ce': final_evovaq_losses['evo_ce'],
                'final_evovaq_final_ce': final_evovaq_losses['final_ce'],
                'final_evovaq_train_log_loss': final_evovaq_losses['train_log_loss'],
                'final_evovaq_evo_log_loss': final_evovaq_losses['evo_log_loss'],
                'final_evovaq_final_log_loss': final_evovaq_losses['final_log_loss'],

                'evovaq_initial_center_fitness': evovaq_info['initial_center_fitness'],
                'evovaq_initial_best_fitness': evovaq_info['initial_best_fitness'],
                'evovaq_best_fitness': evovaq_info['best_fitness'],
                'evovaq_nfev': evovaq_info['nfev'],
                'evovaq_gen': evovaq_info['gen'],
                'evovaq_time_sec': evovaq_info['time_sec'],
                'evovaq_initial_center_max_diff': evovaq_info['initial_center_max_diff'],
            }
        )

        models_rga = {
            'QML Simple': (prob_plain, class_order),
            'QML EVOVAQ': (prob_evovaq, class_order),
        }

        results_rga = compare_models_rga(
            models_rga,
            y_labels=y_final_val,
            n_segments=N_SEGMENTS,
            fig_size=(8, 6),
            verbose=True,
            save_path=str(FIG_DIR / f'rga_fold{fold}_qml.png'),
        )

        rga_dict = {
            m: float(results_rga[m]['rga_full'])
            for m in model_names
        }

        for m in model_names:
            safe_store[m]['rga_curve'].append(
                np.asarray(results_rga[m]['curve_model'], float)
            )
            safe_store[m]['rga_full'].append(
                float(results_rga[m]['rga_full'])
            )

            aurga_value = results_rga[m].get(
                'aurga_normalized_to_perfect',
                results_rga[m].get('aurga'),
            )

            safe_store[m]['aurga'].append(float(aurga_value))

        x_final_val_t = torch.tensor(x_final_val, dtype=torch.float32)

        models_rgr = {
            'QML Simple': (
                qml_plain,
                x_final_val_t,
                prob_plain,
                class_order,
                'pytorch',
                device,
            ),
            'QML EVOVAQ': (
                qml_evovaq,
                x_final_val_t,
                prob_evovaq,
                class_order,
                'pytorch',
                device,
            ),
        }

        results_rgr = compare_models_rgr(
            models_dict=models_rgr,
            noise_levels=noise_levels,
            class_order=class_order,
            rga_dict=rga_dict,
            fig_size=(8, 6),
            verbose=True,
            random_seed=SEED,
            save_path=str(FIG_DIR / f'rgr_fold{fold}_qml.png'),
        )

        for m in model_names:
            safe_store[m]['rgr_curve'].append(
                np.asarray(results_rgr[m]['rgr_rescaled'], float)
            )
            safe_store[m]['aurgr'].append(
                float(results_rgr[m]['aurgr'])
            )

        rge_n_steps = (
            int(RGE_N_STEPS)
            if RGE_N_STEPS is not None
            else x_final_val.shape[1]
        )

        models_rge = {
            'QML Simple': (
                qml_plain,
                x_final_val,
                feature_names,
                prob_plain,
                class_order,
                'pytorch',
                device,
            ),
            'QML EVOVAQ': (
                qml_evovaq,
                x_final_val,
                feature_names,
                prob_evovaq,
                class_order,
                'pytorch',
                device,
            ),
        }

        results_rge = compare_models_rge_tabular(
            models_dict=models_rge,
            class_order=class_order,
            rga_dict=rga_dict,
            verbose=True,
            random_seed=SEED,
            fig_size=(8, 6),
            save_path=str(FIG_DIR / f'rge_fold{fold}_qml.png'),
            masking_method=RGE_MASKING_METHOD,
            baseline=RGE_BASELINE,
            n_steps=rge_n_steps,
        )

        for m in model_names:
            safe_store[m]['rge_curve'].append(
                np.asarray(results_rge[m]['rge_rescaled'], float)
            )
            safe_store[m]['aurge'].append(
                float(results_rge[m]['aurge'])
            )

    print('\n================ OVERALL 5-FOLD (STANDARD) ================')

    for model, m in results.items():
        print(
            f'{model:>10} | '
            f'ACC={np.mean(m["acc"]):.4f} '
            f'F1={np.mean(m["f1"]):.4f} '
            f'MSE={np.mean(m["mse"]):.6f}'
        )

    print('\n================ OVERALL 5-FOLD (SAFE) ================')

    rows = []
    diagnostic_df = pd.DataFrame(diagnostic_rows)

    for m in model_names:
        rga_mean = np.nanmean(safe_store[m]['rga_full'])
        aurga_mean = np.nanmean(safe_store[m]['aurga'])
        aurgr_mean = np.nanmean(safe_store[m]['aurgr'])
        aurge_mean = np.nanmean(safe_store[m]['aurge'])

        print(
            f'{m:>10} | '
            f'RGA={rga_mean:.4f}  '
            f'AURGA={aurga_mean:.4f}  '
            f'AURGR={aurgr_mean:.4f}  '
            f'AURGE={aurge_mean:.4f}'
        )

        row = {
            'model': m,
            'ACC': float(np.mean(results[m]['acc'])),
            'F1': float(np.mean(results[m]['f1'])),
            'MSE': float(np.mean(results[m]['mse'])),
            'RGA': float(rga_mean),
            'AURGA': float(aurga_mean),
            'AURGR': float(aurgr_mean),
            'AURGE': float(aurge_mean),
        }

        if m == 'QML EVOVAQ' and not diagnostic_df.empty:
            row['EVOVAQ_ACCEPTED'] = bool(diagnostic_df['evovaq_accepted'].iloc[-1])

            row['classical_evo_log_loss'] = float(diagnostic_df['classical_evo_log_loss'].iloc[-1])
            row['raw_evovaq_evo_log_loss'] = float(diagnostic_df['raw_evovaq_evo_log_loss'].iloc[-1])
            row['readapt_evo_log_loss'] = float(diagnostic_df['readapt_evo_log_loss'].iloc[-1])
            row['final_evovaq_evo_log_loss'] = float(diagnostic_df['final_evovaq_evo_log_loss'].iloc[-1])

            row['classical_final_log_loss'] = float(diagnostic_df['classical_final_log_loss'].iloc[-1])
            row['raw_evovaq_final_log_loss'] = float(diagnostic_df['raw_evovaq_final_log_loss'].iloc[-1])
            row['readapt_final_log_loss'] = float(diagnostic_df['readapt_final_log_loss'].iloc[-1])
            row['final_evovaq_final_log_loss'] = float(diagnostic_df['final_evovaq_final_log_loss'].iloc[-1])

            row['evovaq_initial_center_fitness'] = float(
                diagnostic_df['evovaq_initial_center_fitness'].iloc[-1]
            )
            row['evovaq_initial_best_fitness'] = float(
                diagnostic_df['evovaq_initial_best_fitness'].iloc[-1]
            )
            row['evovaq_best_fitness'] = float(diagnostic_df['evovaq_best_fitness'].iloc[-1])
            row['evovaq_nfev'] = int(diagnostic_df['evovaq_nfev'].iloc[-1])
            row['evovaq_gen'] = int(diagnostic_df['evovaq_gen'].iloc[-1])
            row['evovaq_time_sec'] = float(diagnostic_df['evovaq_time_sec'].iloc[-1])
            row['evovaq_initial_center_max_diff'] = float(
                diagnostic_df['evovaq_initial_center_max_diff'].iloc[-1]
            )

        rows.append(row)

    df_summary = pd.DataFrame(rows)

    out_csv = CSV_DIR / f'hmda_qml_evovaq_summary_fold{selected_fold}.csv'
    df_summary.to_csv(out_csv, index=False)

    print(f'\nSaved summary to: {out_csv}')

    if not diagnostic_df.empty:
        diag_csv = CSV_DIR / f'hmda_qml_evovaq_diagnostics_fold{selected_fold}.csv'
        diagnostic_df.to_csv(diag_csv, index=False)
        print(f'Saved diagnostics to: {diag_csv}')


if __name__ == '__main__':
    main()
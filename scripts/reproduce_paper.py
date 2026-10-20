"""Regenerate manuscript tables/curves from archived runs, without training."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'paper_results'
METRICS = ['acc', 'f1', 'mse', 'rga', 'aurga', 'aurgr', 'aurge']
NAMES = {'Random Forest': 'RF', 'QML Adam': 'E2E-QML',
         'QML EVOVAQ': 'SWO-QML-1', 'QML EVOVAQ+Adaptation': 'SWO-QML-2'}
KEYS = ['n_qubits', 'n_layers', 'model', 'run_seed']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_runs(path):
    data = pd.read_csv(path)
    data = data[data.model.isin(NAMES)].copy()
    require(not data.duplicated(KEYS).any(), f'Duplicate run keys: {path}')
    require(np.isfinite(data[METRICS].to_numpy()).all(), f'Invalid metrics: {path}')
    require((data.n_quantum_params == 3 * data.n_qubits * data.n_layers).all(),
            f'Incorrect parameter counts: {path}')
    return data


def check_grid(data, configs):
    expected = {(q, l, m, seed) for q, l in configs for m in NAMES
                for seed in range(42, 47)}
    require(set(data[KEYS].itertuples(index=False, name=None)) == expected,
            'Missing or unexpected configurations, models, or seeds')


def compare_candidate(path, reference, atol):
    candidate = read_runs(path).set_index(KEYS)
    require(len(candidate) > 0, 'Candidate contains no paper models')
    ref = reference.set_index(KEYS)
    require(candidate.index.isin(ref.index).all(), 'Unknown candidate run keys')
    delta = (candidate[METRICS] - ref.loc[candidate.index, METRICS]).abs()
    require(np.isfinite(delta.to_numpy()).all(), 'Invalid candidate/reference join')
    print(f'Fresh-run comparison: {len(candidate)} model rows; '
          f'max absolute metric difference = {delta.to_numpy().max():.9g}')
    require((delta <= atol).all().all(),
            f'Candidate differs from archived metrics (tolerance {atol}):\n{delta}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'results/paper_rebuilt')
    parser.add_argument('--candidate', type=Path, help='Compare completed fresh per-run CSV')
    parser.add_argument('--atol', type=float, default=1e-6)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    depth = read_runs(ARCHIVE / 'depth_per_run.csv')
    qubits = read_runs(ARCHIVE / 'qubits_per_run.csv')
    check_grid(depth, [(5, l) for l in range(1, 6)])
    check_grid(qubits, [(q, 2) for q in range(3, 8)])
    overlap = depth[depth.n_layers == 2].set_index(KEYS)[METRICS].sort_index()
    require(np.allclose(overlap, qubits[qubits.n_qubits == 5].set_index(KEYS)[METRICS]
                        .sort_index(), rtol=0, atol=1e-14), 'Sweep overlap disagrees')
    unique = pd.concat([depth, qubits]).drop_duplicates(KEYS)
    reference = pd.read_csv(ARCHIVE / 'table_ii_reference.csv')
    require(len(reference) == 31, 'Expected 31 manuscript rows')
    table = []
    reverse = {v: k for k, v in NAMES.items()}
    checked = 0
    for _, row in reference.iterrows():
        data = depth if row.scenario in ('depth', 'baseline') else qubits
        values = data[(data.n_qubits == row.n_qubits) & (data.n_layers == row.n_layers)
                      & (data.model == reverse[row.model])][METRICS]
        result = row[['scenario', 'n_qubits', 'n_layers', 'model']].to_dict()
        for metric in METRICS:
            for stat, value in [('mean', values[metric].mean()),
                                ('std', values[metric].std(ddof=1))]:
                require(abs(value - row[f'{metric}_{stat}']) <= 0.0000500001,
                        f'Table II mismatch: {row.model}, Q{row.n_qubits}, '
                        f'L{row.n_layers}, {metric} {stat}')
                result[f'{metric}_{stat}'] = value
                checked += 1
        table.append(result)
    pd.DataFrame(table).to_csv(out / 'table_ii.csv', index=False)
    print(f'Table II: all {checked} printed means/standard deviations match (4 decimals).')

    # Section V-C budget: quantum parameters only; no hardware execution implied.
    configs = [(5, l) for l in range(1, 6)] + [(q, 2) for q in [3, 4, 6, 7]]
    budget = pd.DataFrame([{'q': q, 'L': l, 'N': 3*q*l,
                           'N_E2E': 20*13718*(2*3*q*l+1),
                           'N_SWO_upper_bound': 10*31*13718} for q, l in configs])
    expected_e2e = [8505160, 16735960, 24966760, 33197560, 41428360,
                    10151320, 13443640, 20028280, 23320600]
    require(budget.N_E2E.tolist() == expected_e2e, 'Table III E2E mismatch')
    require((budget.N_SWO_upper_bound == 4252580).all(), 'Table III SWO mismatch')
    budget.to_csv(out / 'table_iii.csv', index=False)
    print('Table III: all 9 configurations match the manuscript formulas.')

    curves = pd.read_csv(ARCHIVE / 'q5_l5_safe_curves.csv')
    curve_keys = ['run_seed', 'model', 'curve', 'step']
    require(not curves.duplicated(curve_keys).any(), 'Duplicate curve points')
    require(np.isfinite(curves[['x', 'value']]).all().all(), 'Invalid curve points')
    require(set(curves.model) == set(NAMES.values()), 'Missing curve models')
    require(set(zip(curves.n_qubits, curves.n_layers)) == {(5, 5)}, 'Wrong curve config')
    require(set(curves.curve) == {'RGA', 'RGR', 'RGE'}, 'Missing curve types')
    for (name, curve), points in curves.groupby(['model', 'curve']):
        require(set(points.run_seed) == set(range(42, 47)), f'Missing curve seeds: {name}')
        grids = [g.sort_values('step').x.to_numpy() for _, g in points.groupby('run_seed')]
        require(all(np.array_equal(grids[0], g) for g in grids), 'Curve grids disagree')
        require(np.all(np.diff(grids[0]) > 0), 'Non-increasing curve grid')
    mean = curves.groupby(['model', 'curve', 'x']).value.agg(['mean', 'std']).reset_index()
    mean.to_csv(out / 'figure_2_mean_curves.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    labels = [('RGA', 'Removed sample fraction'), ('RGR', 'Gaussian noise level'),
              ('RGE', 'Removed feature fraction')]
    for ax, (kind, xlabel) in zip(axes, labels):
        for name, marker in zip(NAMES.values(), ['o', 's', '^', 'D']):
            d = mean[(mean.curve == kind) & (mean.model == name)].sort_values('x')
            ax.plot(d.x, d['mean'], marker=marker, markersize=3, label=name)
        ax.set(xlabel=xlabel, ylabel=f'{kind} score')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'figure_2.png', dpi=200)
    plt.close(fig)
    print('Figure 2: rebuilt three mean curves from five seeds per model.')
    if args.candidate:
        compare_candidate(args.candidate, unique, args.atol)
    print(f'Outputs: {out}')


if __name__ == '__main__':
    main()

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


VALUES_TO_REMOVE = {
    'action_taken': [2, 4, 5, 6, 7, 8],
    'loan_type': [3, 4],
    'applicant_race_1': [1, 2, 4, 6, 7],
    'lien_status': [3, 4],
    'applicant_sex': [3, 4],
    'co_applicant_sex': [3, 4],
    'co_applicant_race_1': [1, 2, 4, 6, 7],
    'applicant_ethnicity': [3, 4],
    'co_applicant_ethnicity': [3, 4]
}

COLUMNS_TO_REMOVE = [
    'as_of_year', 'agency_name', 'agency_abbr', 'agency_code', 'property_type_name',
    'property_type', 'owner_occupancy_name', 'owner_occupancy', 'preapproval_name',
    'preapproval', 'state_name', 'state_abbr', 'state_code', 'applicant_race_name_2',
    'applicant_race_2', 'applicant_race_name_3', 'applicant_race_3',
    'applicant_race_name_4', 'applicant_race_4', 'applicant_race_name_5',
    'applicant_race_5', 'co_applicant_race_name_2', 'co_applicant_race_2',
    'co_applicant_race_name_3', 'co_applicant_race_3', 'co_applicant_race_name_4',
    'co_applicant_race_4', 'co_applicant_race_name_5', 'co_applicant_race_5',
    'purchaser_type_name', 'purchaser_type', 'denial_reason_name_1',
    'denial_reason_1', 'denial_reason_name_2', 'denial_reason_2',
    'denial_reason_name_3', 'denial_reason_3', 'rate_spread', 'hoepa_status_name',
    'hoepa_status', 'edit_status_name', 'edit_status', 'sequence_number',
    'application_date_indicator', 'respondent_id', 'loan_type_name',
    'loan_purpose_name', 'action_taken_name', 'msamd_name', 'county_name',
    'applicant_ethnicity_name', 'co_applicant_ethnicity_name',
    'applicant_race_name_1', 'co_applicant_race_name_1', 'applicant_sex_name',
    'co_applicant_sex_name', 'lien_status_name'
]


def load_raw_hmda(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, low_memory=False)


def clean_hmda(df: pd.DataFrame, target_col: str = 'action_taken') -> pd.DataFrame:
    df = df.copy()

    for col, bad_values in VALUES_TO_REMOVE.items():
        if col in df.columns:
            df = df[~df[col].isin(bad_values)]

    cols_to_drop = [c for c in COLUMNS_TO_REMOVE if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    df = df.dropna(axis='index')

    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col!r} not found')

    df.loc[:, target_col] = df[target_col].map({1: 0, 3: 1})
    df = df.dropna(axis='index')
    df.loc[:, target_col] = df[target_col].astype(int)

    return df


def stratified_subsample(
    df: pd.DataFrame,
    target_col: str,
    create_sample: bool,
    sample_fraction: float = 0.1,
    random_state: int = 42
) -> pd.DataFrame:
    if not create_sample:
        return df

    if target_col not in df.columns:
        raise ValueError(f'Target column {target_col!r} not found for subsample')

    if sample_fraction <= 0 or sample_fraction >= 1:
        raise ValueError('sample_fraction must be between 0 and 1')

    _, sample_df = train_test_split(
        df,
        test_size=sample_fraction,
        random_state=random_state,
        stratify=df[target_col],
    )

    return sample_df


def load_clean_hmda_from_config(project_root: Path, config: dict, seed: int):
    hmda_cfg = config.get('hmda', {})
    raw_csv = project_root / hmda_cfg.get('raw_csv', 'data/hmda_2017_ny_all-records_labels.csv')
    target_col = hmda_cfg.get('target', 'action_taken')
    create_sample = bool(hmda_cfg.get('create_sample', True))
    sample_fraction = float(hmda_cfg.get('sample_fraction', 0.1))

    print('Loading HMDA:', raw_csv)

    df_raw = load_raw_hmda(raw_csv)
    df = clean_hmda(df_raw, target_col=target_col)

    df = stratified_subsample(
        df,
        target_col,
        create_sample=create_sample,
        sample_fraction=sample_fraction,
        random_state=seed,
    )

    y = df[target_col].astype(int).to_numpy()
    x_df = df.drop(columns=[target_col])
    feature_names = x_df.columns.to_list()
    x = x_df.to_numpy(dtype='float32')

    print('HMDA cleaned shape:', df.shape)
    print('x shape:', x.shape, '| y shape:', y.shape)

    return x, y, feature_names
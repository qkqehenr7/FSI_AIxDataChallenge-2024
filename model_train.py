import pandas as pd
import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import numpy as np


def prepare_dataset_for_model(
        train_total: pd.DataFrame,
        additional_dup_for_others: int = 2
):
    train_df = train_total.copy()

    # Fraud_Type이 'm'인 데이터
    type_m = train_df[train_df['Fraud_Type'] == 'm']
    m_sample = type_m.sample(n=1100, random_state=42)

    # 나머지 Fraud_Type
    other_types = train_df[train_df['Fraud_Type'] != 'm']

    # m을 제외한 클래스를 여러 배로 복제
    merged = [m_sample]
    for _ in range(additional_dup_for_others):
        merged.append(other_types)
    train_data = pd.concat(merged, ignore_index=True)

    # 행 순서 섞기
    train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)
    return train_data


def encode_features(train_data: pd.DataFrame, test_data: pd.DataFrame, target_col: str = 'Fraud_Type'):
    """
    - 레이블인코딩 (target)
    - OrdinalEncoder (일부 범주형 변수)
    - One-Hot 인코딩 (특정 칼럼만)
    - 최종 train_x, train_y, test_x 반환
    """
    train_x = train_data.drop(columns=[target_col])
    train_y = train_data[target_col]

    # 레이블 인코딩
    le_subclass = LabelEncoder()
    train_y_encoded = le_subclass.fit_transform(train_y)

    # 분류에 사용할 수 있도록 test_data에서도 동일 컬럼 제거
    test_x = test_data.drop(columns=['ID'])

    # 원핫 인코딩할 칼럼
    one_hot_columns = ['Customer_loan_type']

    # 범주형 컬럼 중 원핫 인코딩 제외 목록(Ordinal Encoding 대상)
    cat_cols = train_x.select_dtypes(include=['object', 'category']).columns
    categorical_columns = cat_cols.difference(one_hot_columns)

    # Ordinal Encoding
    ordinal_encoder = OrdinalEncoder(
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    train_x[categorical_columns] = ordinal_encoder.fit_transform(train_x[categorical_columns])
    test_x[categorical_columns] = ordinal_encoder.transform(test_x[categorical_columns])

    # 원핫 인코딩
    train_x = pd.get_dummies(train_x, columns=one_hot_columns, drop_first=True)
    test_x = pd.get_dummies(test_x, columns=one_hot_columns, drop_first=True)

    # train, test 컬럼 통일
    feature_order = train_x.columns.tolist()
    test_x = test_x.reindex(columns=feature_order, fill_value=0)

    # 타입 일치화
    for col in feature_order:
        test_x[col] = test_x[col].astype(train_x[col].dtype)

    return train_x, train_y_encoded, test_x, le_subclass


def tune_xgb_hyperparams(X: pd.DataFrame, y: np.array, n_trials: int = 50, random_seed: int = 42):

    def objective(trial):
        xgb_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'n_estimators': trial.suggest_int('n_estimators', 200, 2000),
            'max_depth': trial.suggest_int('max_depth', 4, 30),
            'gamma': trial.suggest_float('gamma', 0.0, 0.9),
            'min_child_weight': trial.suggest_float('min_child_weight', 1, 30),
            'subsample': trial.suggest_float('subsample', 0.2, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 1.0),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 10.0),
            'max_delta_step': trial.suggest_float('max_delta_step', 0, 10),
        }

        xgb = XGBClassifier(
            **xgb_params,
            device='cuda',
            random_state=42,
            class_weight='balanced'
        )

        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        f1_scores = []

        for train_index, val_index in kf.split(X, y):
            X_train_fold = X.iloc[train_index]
            X_val_fold = X.iloc[val_index]
            y_train_fold = y[train_index]
            y_val_fold = y[val_index]

            xgb.fit(X_train_fold, y_train_fold, verbose=False)
            y_pred = xgb.predict(X_val_fold)
            f1 = f1_score(y_val_fold, y_pred, average='macro')
            f1_scores.append(f1)

        return sum(f1_scores) / len(f1_scores)

    sampler = optuna.samplers.TPESampler(seed=random_seed)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def train_final_model(X: pd.DataFrame, y: np.array, best_params: dict = None):
    """
    XGBoost 모델을 최적 파라미터로 학습 후, 학습된 모델을 반환합니다.
    """
    if best_params is None:
        # 만약 튜닝을 안 했다면 기본값 지정
        best_params = {
            'learning_rate': 0.02,
            'n_estimators': 1000,
            'max_depth': 10,
            'gamma': 0.0,
            'min_child_weight': 1.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_lambda': 1.0,
            'reg_alpha': 0.0,
            'scale_pos_weight': 1.0,
            'max_delta_step': 0
        }

    model = XGBClassifier(
        **best_params,
        device='cuda',
        class_weight='balanced',
        random_state=42
    )
    model.fit(X, y)
    return model

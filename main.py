import os
import warnings

warnings.filterwarnings('ignore')

import pandas as pd

from data_load import load_data
from preprocessing import (
    preprocess_train_data,
    preprocess_test_data,
    refine_data_for_model
)
from synthetic_generation import (
    generate_synthetic_data,
    postprocess_synthetic_data
)
from model_train import (
    prepare_dataset_for_model,
    encode_features,
    tune_xgb_hyperparams,
    train_final_model
)
from submission import (
    predict_and_create_submission,
    create_syn_submission,
    make_zipfile
)


def main():
    train_path = "data/train.csv"
    test_path = "data/test.csv"
    sample_submission_path = "data/sample_submission.csv"

    train_all, test_all = load_data(train_path, test_path)

    # ===== 전처리 =====
    train_processed = preprocess_train_data(train_all)
    test_processed = preprocess_test_data(test_all)

    # ===== 합성 데이터 생성 =====
    all_synthetic_data = generate_synthetic_data(
        train_processed, n_sample=100, n_cls_per_gen=2000
    )
    synthetic_data = postprocess_synthetic_data(all_synthetic_data)

    # ===== 원본 train + 합성데이터 병합 =====
    origin_train = train_all.drop(columns="ID").copy()
    # Time_difference 전처리
    origin_train['Time_difference'] = pd.to_timedelta(origin_train['Time_difference']).dt.total_seconds()
    origin_train['Time_difference'] = origin_train['Time_difference'].clip(lower=1)

    train_total = pd.concat([origin_train, synthetic_data])

    # ===== 모델 학습용 데이터 전처리 =====
    # 날짜 컬럼 제거 & 기타 전처리
    train_total, test_processed = refine_data_for_model(train_total, test_processed)

    train_data = prepare_dataset_for_model(train_total, additional_dup_for_others=2)

    train_x, train_y_encoded, test_x, le_subclass = encode_features(
        train_data, test_processed, target_col='Fraud_Type'
    )

    # ===== 하이퍼 파라미터 튜닝 =====
    # 시간이 오래 걸릴 수 있으니 필요 시 주석 해제
    # best_params = tune_xgb_hyperparams(train_x, train_y_encoded, n_trials=20, random_seed=42)
    # print("Best hyperparameters:", best_params)

    # 이미 최적값을 알고 있다면 직접 입력
    best_params = {
        'learning_rate': 0.019935063753921546,
        'n_estimators': 1209,
        'max_depth': 21,
        'gamma': 0.13021237049313528,
        'min_child_weight': 1.0037029366354757,
        'subsample': 0.9397452843795323,
        'colsample_bytree': 0.5483858560356617,
        'reg_lambda': 0.09062574910114087,
        'reg_alpha': 0.23614577547483448,
        'scale_pos_weight': 5.421487621314032,
        'max_delta_step': 9.323657555341919
    }


    model = train_final_model(train_x, train_y_encoded, best_params)

    predict_and_create_submission(
        model=model,
        test_x=test_x,
        label_encoder=le_subclass,
        sample_submission_path=sample_submission_path
    )

    create_syn_submission(synthetic_data)

    make_zipfile(zip_name="FSI_result.zip")

    print("[Info] Done.")


if __name__ == "__main__":
    main()

import pandas as pd
import numpy as np


def preprocess_train_data(train_all: pd.DataFrame):
    """
    train.csv를 로드한 뒤 필요한 전처리를 수행합니다.
    """
    train = train_all.drop(columns="ID").copy()

    # Time_difference를 초 단위로 변환
    train['Time_difference'] = pd.to_timedelta(train['Time_difference']).dt.total_seconds()
    # 0 이하값을 1로 대치
    train['Time_difference'] = train['Time_difference'].clip(lower=1)

    # velocity 컬럼 생성
    train['velocity'] = train['Distance'] / train['Time_difference']

    return train


def preprocess_test_data(test_all: pd.DataFrame):
    """
    test.csv를 로드한 뒤 필요한 전처리를 수행합니다.
    """
    test_data = test_all.copy()
    # Time_difference를 초 단위로 변환
    test_data['Time_difference'] = pd.to_timedelta(test_data['Time_difference']).dt.total_seconds()
    test_data['Time_difference'] = test_data['Time_difference'].clip(lower=0)  # 0 이하 대치

    return test_data


def refine_data_for_model(train_data: pd.DataFrame, test_data: pd.DataFrame):
    """
    - 공통적으로 제거할 컬럼들 제거
    - 날짜 -> 이진변수 변환
    - Location 전처리
    - IP_Address 전처리
    - 잘 안쓰이거나 값이 거의 없는 컬럼 제거
    """
    # ========== Train 데이터 전처리 ==========
    # 날짜 컬럼들 삭제
    drop_cols = [
        'Account_creation_datetime',
        'Transaction_Datetime',
        'Last_atm_transaction_datetime',
        'Last_bank_branch_transaction_datetime',
        'Transaction_resumed_date'
    ]
    for c in drop_cols:
        if c in train_data.columns:
            train_data.drop(c, axis=1, inplace=True)

    # 날짜 -> 이진변수
    if 'Customer_registration_datetime' in train_data.columns:
        train_data['Customer_registration_datetime'] = (
                train_data['Customer_registration_datetime'] > '2013-01-01'
        ).astype(int)

    # Location 전처리
    if 'Location' in train_data.columns:
        train_data['Location'] = train_data['Location'].apply(lambda x: x.split(' ')[0])

    # 이체 한도를 순서형 범주처럼 변환
    if 'Account_amount_daily_limit' in train_data.columns:
        train_data['Account_amount_daily_limit'] = (
                train_data['Account_amount_daily_limit'] / 1_000_000
        ).astype(int)

    # IP 주소 앞 두자리만 사용
    if 'IP_Address' in train_data.columns:
        train_data['IP_Address'] = train_data['IP_Address'].apply(lambda x: '.'.join(x.split('.')[:2]))

    # 고유값이 1개이거나 거의 1개인 변수 제거
    to_drop = [
        'Another_Person_Account',
        'Account_indicator_Openbanking',
        'First_time_iOS_by_vulnerable_user'
    ]
    for c in to_drop:
        if c in train_data.columns:
            train_data.drop(c, axis=1, inplace=True)

    # 사용 안할 컬럼 제거
    if 'Customer_personal_identifier' in train_data.columns:
        train_data.drop('Customer_personal_identifier', axis=1, inplace=True)

    # ========== Test 데이터 전처리 ==========

    for c in drop_cols:
        if c in test_data.columns:
            test_data.drop(c, axis=1, inplace=True)

    if 'Customer_registration_datetime' in test_data.columns:
        test_data['Customer_registration_datetime'] = (
                test_data['Customer_registration_datetime'] > '2013-01-01'
        ).astype(int)

    if 'Location' in test_data.columns:
        test_data['Location'] = test_data['Location'].apply(lambda x: x.split(' ')[0])

    if 'Account_amount_daily_limit' in test_data.columns:
        test_data['Account_amount_daily_limit'] = (
                test_data['Account_amount_daily_limit'] / 1_000_000
        ).astype(int)

    if 'IP_Address' in test_data.columns:
        test_data['IP_Address'] = test_data['IP_Address'].apply(lambda x: '.'.join(x.split('.')[:2]))

    for c in to_drop:
        if c in test_data.columns:
            test_data.drop(c, axis=1, inplace=True)

    if 'Customer_personal_identifier' in test_data.columns:
        test_data.drop('Customer_personal_identifier', axis=1, inplace=True)

    return train_data, test_data

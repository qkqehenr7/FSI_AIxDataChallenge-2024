import pandas as pd
from tqdm import tqdm
from sdv.metadata import SingleTableMetadata
from sdv.single_table import CTGANSynthesizer


def generate_synthetic_data(train: pd.DataFrame, n_sample=100, n_cls_per_gen=2000):

    fraud_types = train['Fraud_Type'].unique()
    all_synthetic_data = pd.DataFrame()

    for fraud_type in tqdm(fraud_types):
        # 해당 Fraud_Type만 샘플링
        subset = train[train["Fraud_Type"] == fraud_type].sample(n=n_sample, random_state=42)

        # 메타데이터
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(subset)
        metadata.set_primary_key(None)

        # 필요한 컬럼 타입 지정 (사용자 필요에 맞게 수정)
        column_sdtypes = {
            'Account_initial_balance': 'numerical',
            'Account_balance': 'numerical',
            'Customer_identification_number': 'categorical',
            'Customer_personal_identifier': 'categorical',
            'Account_account_number': 'categorical',
            'IP_Address': 'ipv4_address',
            'Location': 'categorical',
            'Recipient_Account_Number': 'categorical',
            'Fraud_Type': 'categorical',
            'Time_difference': 'numerical',
            'Customer_Birthyear': 'numerical'
        }

        for column, sdtype in column_sdtypes.items():
            if column in metadata.columns:
                metadata.update_column(
                    column_name=column,
                    sdtype=sdtype
                )

        synthesizer = CTGANSynthesizer(
            metadata,
            epochs=100
        )
        synthesizer.fit(subset)

        synthetic_subset = synthesizer.sample(num_rows=n_cls_per_gen)
        all_synthetic_data = pd.concat([all_synthetic_data, synthetic_subset], ignore_index=True)

    # 생성된 데이터에 Distance 복원 (velocity x Time_difference)
    if 'velocity' in all_synthetic_data.columns and 'Time_difference' in all_synthetic_data.columns:
        all_synthetic_data['Distance'] = (
                all_synthetic_data['velocity'] * all_synthetic_data['Time_difference']
        )
        all_synthetic_data.drop('velocity', axis=1, inplace=True)

    return all_synthetic_data


def postprocess_synthetic_data(all_synthetic_data: pd.DataFrame):
    """
    합성 데이터에 대한 후처리 (특정 로직에 맞춰 행 제거, Fraud_Type별 샘플 수 확보 등).
    """
    df = all_synthetic_data.copy()
    # Mobile이 아닌 기기 중 로밍 데이터 제거
    df = df[~((df['Channel'] != 'mobile') & (df['Customer_mobile_roaming_indicator'] == 1))]
    # Others에서 OS가 Windows, Others가 아닌 데이터 삭제
    exclude_condition1 = (
            (df['Channel'] == 'Others') &
            (df['Operating_System'].isin(['iOS', 'Android', 'Linux', 'macOS']))
    )
    df = df[~exclude_condition1]

    # ATM에서 OS가 Windows, Others가 아닌 데이터 삭제
    exclude_condition2 = (
            (df['Channel'] == 'ATM') &
            (df['Operating_System'].isin(['iOS', 'Android', 'Linux', 'macOS']))
    )
    df = df[~exclude_condition2]

    # mobile에서 OS가 Windows, Linux, macOS인 데이터 삭제
    exclude_condition3 = (
            (df['Channel'] == 'mobile') &
            (df['Operating_System'].isin(['Windows', 'Linux', 'macOS']))
    )
    df = df[~exclude_condition3]

    # internet에서 OS가 iOS, Android인 데이터 삭제
    exclude_condition4 = (
            (df['Channel'] == 'internet') &
            (df['Operating_System'].isin(['iOS', 'Android']))
    )
    df = df[~exclude_condition4]

    # 클래스별 1000개씩
    synthetic_data = df.groupby('Fraud_Type').apply(
        lambda x: x.sample(n=1000, random_state=42) if len(x) >= 1000 else x
    ).reset_index(drop=True)

    return synthetic_data

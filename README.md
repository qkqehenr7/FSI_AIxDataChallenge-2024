# FSI AIxData Challenge 2024

- 주최 / 주관 : 금융보안원
- 후원 : 금융위원회, KB국민은행, 하나은행, 미래에셋증권, 생명보험협회
- 운영 : 데이콘
- https://dacon.io/competitions/official/236297/overview/description

- 주제 : 이상금융거래 데이터셋으로 분류 AI모델을 구현하고, 오픈소스 생성형 AI 모델을 응용/활용하여 분류 AI모델의 성능을 개선

  1) 클래스 불균형이 심한 데이터셋의 특성을 고려하여 분류 AI모델 개발

  2) 제공하는 데이터셋을 오픈소스 생성형 AI 모델 등 AI 기술에 응용 

  3) 이를 분류 AI모델에 활용함으로써 분류 AI모델의 성능을 개선
 

# Preprocessing
- 파생변수 생성 : velocity >> Synthetic Data 생성 후 Distance 재생성으로 반영
- 불균형 클래스 처리 : m 1100개, 나머지 3300개 로 단순 복사
- 입출금 기기에 따른 OS 필터링
- 고객 가입일자 이진 변환
- 컬럼 삭제 : 가입일자 제외 날짜 데이터, Another_Person_Account, Account_indicator_Openbanking, First_time_iOS_by_vulnerable_user

# Synthetic Data
- Library : CTGANSynthesizer
- epoch : 100
  
  ```python
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
  ```


# Hyperparameter Tuning
- Library : optuna

```python
Best hyperparameters: {'learning_rate': 0.019935063753921546, 'n_estimators': 1209, 'max_depth': 21, 'gamma': 0.13021237049313528, 'min_child_weight': 1.0037029366354757, 'subsample': 0.9397452843795323, 'colsample_bytree': 0.5483858560356617, 'reg_lambda': 0.09062574910114087, 'reg_alpha': 0.23614577547483448, 'scale_pos_weight': 5.421487621314032, 'max_delta_step': 9.323657555341919}
```
- Model : XGBoost

```python
model = XGBClassifier(**best_params,
                      device='cuda',
                      class_weight='balanced',
                      random_state=42)

model.fit(train_x_encoded[feature_order], train_y_encoded)
```

# Prediction

```python
m    60667
l     7869
e     6753
f     5829
b     5820
k     5384
j     5332
a     4916
i     4794
h     3594
d     3445
g     3261
c     2336
Name: count, dtype: int64
```

# Result

Public : 0.7287, Private : 0.73052, 141팀 중 18위

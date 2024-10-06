Distance와 Time_difference, 두 Feature 간 상관관계를 반영할 수 있는 파생변수 Velocity를 만들어 데이터를 생성하였다.

생성된 데이터셋에서 velocity를 이용하여 Distance 칼럼을 재생성하였으므로 그 결과가 반영되었을 것이다.

이후 파생변수 칼럼은 velocity는 다중공선성 등을 고려하여 삭제하였다.

Target Feature인 Fraud_Type의 클래스 별 샘플 수를 조정하였다.

m 1100개, 나머지 클래스는 1100개의 샘플을 단순복사하여 3300개로 만들어 모델에 반영했다.

Channel과 Operating_System 칼럼에서 필터링을 진행하였다.  또한 모바일이 아닌 기기에서 Customer_mobile_roaming_indicator 에 해당되는 데이터가 있었다.

그 역시 필터링 대상이다. 필터링 과정에서 발생하는 데이터 손실을 고려해 2000개의 데이터를 생성하였다.

날짜데이터 중 Customer_registration_datetime를 2013-01-01 이전, 이후에 따라 0과 1의 값을 갖는 이진 변수로 변환하였다.

그를 제외한 다른 모든 날짜 데이터는 삭제하였다.

Time_difference는 파생변수를 생성하는 과정 이전에 이상치 처리 및 형변환을 진행했는데, 전부 초 단위 데이터로 만들어 정수값을 갖도록 변환하였으며

음수값을 갖는 데이터를 이상치라 판단하고 모두 1로 대치하였다.

Location의 경우 너무 많은 범주를 갖고 분류에 기여하는 Feature가 되지 못한다고 판단해 가장 큰 범주인 도 단위 주소만 남겨 사용했다.

이체 한도인 Account_amount_daily_limit는 연속형 변수임에도 5가지의 값만 가졌다. 따라서 1000000으로 나누어 마치 Score를 부여한 순서형 범주처럼 변환했다.

IP_Address의 경우도 Location과 마찬가지로 맨 앞 두자리 주소만 추출하여 사용했다.

Another_Person_Account, Account_indicator_Openbanking, First_time_iOS_by_vulnerable_user 의 경우 대부분의 값이 하나의 범주에 몰려있거나

단 하나의 값만 갖는 케이스로, 분류에 도움이 되지 않는다고 판단해 삭제하였다.

Customer_personal_identifier는 고객의 이름인데, 이미 주민번호가 있으므로 삭제하였다.

Customer_loan_type은 대출 유형으로, 순서형 범주로서 변환할 수 없다고 판단하여 One-Hot 인코딩으로 처리하였다.

optuna를 이용하여 하이퍼파라미터 튜닝을 진행하였다. Trial은 200회.

계산된 최적의 파라미터는 다음과 같다.

```python
Best hyperparameters: {'learning_rate': 0.019935063753921546, 'n_estimators': 1209, 'max_depth': 21, 'gamma': 0.13021237049313528, 'min_child_weight': 1.0037029366354757, 'subsample': 0.9397452843795323, 'colsample_bytree': 0.5483858560356617, 'reg_lambda': 0.09062574910114087, 'reg_alpha': 0.23614577547483448, 'scale_pos_weight': 5.421487621314032, 'max_delta_step': 9.323657555341919}
```

모델은 XGBoost를 사용하였다. 여전히 데이터 불균형이 존재하므로 가중치를 주어 완화하고자 했다.

```python
model = XGBClassifier(**best_params,
                      device='cuda',
                      class_weight='balanced',
                      random_state=42)

model.fit(train_x_encoded[feature_order], train_y_encoded)
```

예측된 클래스의 값들은 다음과 같다.

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

제출 결과

Public : 0.7287, Private : 0.73052

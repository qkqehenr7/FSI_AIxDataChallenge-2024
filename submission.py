import os
import zipfile
import pandas as pd

def predict_and_create_submission(
    model,
    test_x,
    label_encoder,
    sample_submission_path: str,
    submission_zip_name: str = "FSI_result.zip"
):

    predictions = model.predict(test_x)
    predictions_label = label_encoder.inverse_transform(predictions)

    clf_submission = pd.read_csv(sample_submission_path)
    clf_submission["Fraud_Type"] = predictions_label
    clf_submission.to_csv("clf_submission.csv", encoding='utf-8-sig', index=False)

    print("[Info] clf_submission.csv 생성 완료.")

def create_syn_submission(
    synthetic_data: pd.DataFrame
):
    
    synthetic_data.to_csv("syn_submission.csv", encoding='utf-8-sig', index=False)
    print("[Info] syn_submission.csv 생성 완료.")

def make_zipfile(zip_name="FSI_result.zip"):
    with zipfile.ZipFile(zip_name, 'w') as submission:
        submission.write('clf_submission.csv')
        submission.write('syn_submission.csv')
    print(f"[Info] {zip_name} 생성 완료.")

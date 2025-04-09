import pandas as pd

def load_data(train_path: str, test_path: str):

    train_all = pd.read_csv(train_path)
    test_all = pd.read_csv(test_path)
    return train_all, test_all

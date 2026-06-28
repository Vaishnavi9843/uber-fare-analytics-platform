import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV.
    """

    df = pd.read_csv(path)

    return df

def prepare_features(df: pd.DataFrame):
    """
    Separate features and target.
    """

    drop_columns = [
        "key",
        "pickup_datetime",
        "pickup_day"
    ]

    X = df.drop(
        columns=drop_columns + ["fare_amount"]
    )

    y = df["fare_amount"]

    return X, y

def split_dataset(X, y,test_size=0.2,random_state=42):

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )


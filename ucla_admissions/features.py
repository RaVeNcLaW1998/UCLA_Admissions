import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging


logger = logging.getLogger(__name__)


def preprocess_training_data(df):
    if "Serial_No" in df.columns:
        df = df.drop(columns=["Serial_No"])

    df["Admit_Chance"] = (df["Admit_Chance"] >= 0.8).astype(int)
    df["University_Rating"] = df["University_Rating"].astype("object")
    df["Research"] = df["Research"].astype("object")
    df = pd.get_dummies(df, columns=["University_Rating", "Research"], dtype="int")

    X = df.drop("Admit_Chance", axis=1)
    y = df["Admit_Chance"]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns.tolist()


def preprocess_user_input(user_df, expected_columns, scaler):
    user_df["University_Rating"] = user_df["University_Rating"].astype("object")
    user_df["Research"] = user_df["Research"].astype("object")
    user_df = pd.get_dummies(
        user_df, columns=["University_Rating", "Research"], dtype="int"
    )

    # Add missing columns with zeros
    for col in expected_columns:
        if col not in user_df.columns:
            user_df[col] = 0

    # Ensure the column order is the same
    user_df = user_df[expected_columns]

    # Scale using existing scaler
    X_user_scaled = scaler.transform(user_df)
    return X_user_scaled

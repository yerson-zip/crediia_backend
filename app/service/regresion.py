import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pandas import DataFrame, read_csv, read_excel
from fastapi import UploadFile
from app.const.value import COLUMNAS, PATH_REGRESION_M, PATH_REGRESION_S


model : LogisticRegression = joblib.load(PATH_REGRESION_M)
scaler : StandardScaler    = joblib.load(PATH_REGRESION_S)


def prediction(df: DataFrame)->dict:


    X_scaler = scaler.transform(df)

    predict = model.predict(X_scaler)[0]
    proba   = model.predict_proba(X_scaler)[0][1]

    return {
        "prediction": int(predict),
        "probability": float(proba)
    }


def processing_data(data:dict)->dict:

    df = DataFrame([data])

    df["loan_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1e-6)
    df["loan_assets_ratio"] = df["loan_amount"] / (df["total_assets"] + 1e-6)

    df=df.reindex(columns=COLUMNAS, fill_value=0)

    return prediction(df)


def processing_data_file(file_path:UploadFile):

    if file_path.filename.endswith(".csv"):
        df =  read_csv(file_path.file)

    elif file_path.filename.endswith(".xlsx"):
        df = read_excel(file_path.file)

    else:
        return {"Message":"Formato no valido"}

    df["loan_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1e-6)
    df["loan_assets_ratio"] = df["loan_amount"] / (df["total_assets"] + 1e-6)

    df = df.reindex(columns=COLUMNAS, fill_value=0)

    X = scaler.transform(df)

    df["prediction"] = model.predict(X)
    df["probability"] = model.predict_proba(X)[:,1]

    return df.to_dict(orient="records")
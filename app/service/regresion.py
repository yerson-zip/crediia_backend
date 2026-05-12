import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
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

    df.drop(columns=["loan_status"], errors="ignore", inplace=True)

    df["loan_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1e-6)
    df["loan_assets_ratio"] = df["loan_amount"] / (df["total_assets"] + 1e-6)

    df = df.reindex(columns=COLUMNAS, fill_value=0)
    X = scaler.transform(df)

    df["prediction"] = model.predict(X)
    df["probability"] = model.predict_proba(X)[:,1]

    return df.to_dict(orient="records")

def evaluate_file(file_path:UploadFile)->dict:
    if file_path.filename.endswith(".csv"):
        df = read_csv(file_path.file)

    elif file_path.filename.endswith(".xlsx"):
        df = read_excel(file_path.file)

    else:
        return {"Message": "Formato no valido"}

    if "loan_status" not in df.columns:
        return {"error:""El archivo debe contener la columna 'loan_status'"}

    y_true = df["loan_status"]

    df["loan_income_ratio"] = df["loan_amount"] / (df["income_annum"] + 1e-6)
    df["loan_assets_ratio"] = df["loan_amount"] / (df["total_assets"] + 1e-6)
    df_feat = df.reindex(columns=COLUMNAS, fill_value=0)

    X = scaler.transform(df_feat)
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:,1]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1": round(f1_score(y_true, y_pred), 4),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp)
        },
        "total": len(y_true),
        "approved": int(tp + fn),  # reales positivos
        "rejected": int(tn + fp),
    }
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
import pandas as pd
import mlflow
import pickle


mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("homework-experiment")


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def transform(df:pd.DataFrame, **kwargs):

    with mlflow.start_run():

        categorical = ['PULocationID', 'DOLocationID']
        numerical = ['trip_distance']
        train_dicts = df[categorical + numerical].to_dict(orient='records')
        dv = DictVectorizer()
        
        X_train = dv.fit_transform(train_dicts)

        target = 'duration'
        y_train = df[target].values

        lr = LinearRegression()
        lr.fit(X_train, y_train)
    
        with open("./preprocessor_mage.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("./preprocessor_mage.b", artifact_path="preprocessor")

        mlflow.sklearn.log_model(lr, artifact_path="models_mlflow")



    return lr,dv


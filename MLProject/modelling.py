import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import mlflow
import mlflow.sklearn

df = pd.read_csv('telco_preprocessing/clean_data.csv')

X_train = df.drop(columns=['Churn'])
y_train = df['Churn']

mlflow.autolog()

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
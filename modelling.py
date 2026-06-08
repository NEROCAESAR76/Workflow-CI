import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import os

# Set tracking ke lokal (wajib buat bikin Docker nanti)
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))
mlflow.set_experiment("Telco_Churn_CI")

print("Membaca data...")
df = pd.read_csv('telco_churn_clean.csv')

X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)

with mlflow.start_run():
    print("Melatih model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    mlflow.log_metric("accuracy", acc)
    # Nama folder modelnya: random_forest_model
    mlflow.sklearn.log_model(model, "random_forest_model")
    
    print(f"Selesai! Akurasi: {acc:.2f}")
import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Mengatur lokasi tracking server (Hanya set URI jika dijalankan secara lokal di laptop)
    # Di GitHub Actions, MLflow akan otomatis mencatat ke folder mlruns lokal server
    if not os.environ.get("GITHUB_ACTIONS"):
        mlflow.set_tracking_uri("http://127.0.0.1:5000") 
        
    mlflow.set_experiment("Eksperimen_Model_Telco")

    # 2. Aktifkan Autolog dari MLflow agar parameter & metrik tercatat otomatis
    mlflow.sklearn.autolog()

# Memuat data bersih (clean_data.csv) dengan jaring pencarian yang lebih luas
    possible_paths = [
        "telco_preprocessing/clean_data.csv",
        "MLProject/telco_preprocessing/clean_data.csv",
        "namadataset_preprocessing/clean_data.csv",
        "MLProject/namadataset_preprocessing/clean_data.csv",
        "preprocessing/clean_data.csv",
        "clean_data.csv"
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path is not None:
        print(f"✅ Berhasil menemukan file data di: {data_path}")
        data = pd.read_csv(data_path)
    else:
        # Hentikan program secara paksa jika data tidak ada agar GitHub Actions tahu ada yang salah!
        raise FileNotFoundError("❌ ERROR FATAL: File 'clean_data.csv' tidak ditemukan! Pastikan file dataset sudah terdorong (ter-push) ke GitHub.")

    # 4. Memisahkan fitur (X) dengan target Churn (y)
    X = data.drop(columns=['Churn']) 
    y = data['Churn']

    # Membagi data menjadi data Train dan data Test (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Mulai eksperimen run dengan parameter tetap (TANPA GridSearchCV sesuai ketentuan)
    # Jika berjalan di GitHub Actions (MLflow Project), gunakan active run yang sudah ada
    active_run = mlflow.active_run()
    
    if active_run is not None:
        # Jika dipicu lewat MLflow Project di CI
        print(f"🚀 Melatih model menggunakan Active Run ID dari MLProject: {active_run.info.run_id}")
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
    else:
        # Jika dijalankan manual di lokal laptop kamu
        with mlflow.start_run(run_name="Model_Standar_RandomForest"):
            model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            print("🚀 Melatih model utama dengan autolog...")
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            acc = accuracy_score(y_test, predictions)
            
    # Bagian evaluasi cetak laporan tetap diletakkan di luar blok agar selalu tereksekusi
    print("\n--- Hasil Evaluasi Model ---")
    print(f"Akurasi Model Standar: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))
    print("✅ Metrik dan model berhasil direkam secara otomatis!")
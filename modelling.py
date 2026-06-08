import os
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    # 1. Mengatur lokasi tracking server ke MLflow lokal komputer (Bukan DagsHub)
    mlflow.set_tracking_uri("http://127.0.0.1:5000") 
    mlflow.set_experiment("Eksperimen_Model_Telco")

    # 2. Aktifkan Autolog dari MLflow agar parameter & metrik tercatat otomatis
    mlflow.sklearn.autolog()

    # 3. Memuat data bersih (clean_data.csv) menggunakan pendeteksian path otomatis
    possible_paths = [
        "preprocessing/clean_data.csv",
        "clean_data.csv",
        "../preprocessing/clean_data.csv",
        "Eksperimen_SML_Nero/preprocessing/clean_data.csv"
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
        print("❌ Error: File 'clean_data.csv' benar-benar tidak ditemukan di folder lokal kamu.")
        print(f"Posisi aktif terminal kamu saat ini berada di: {os.getcwd()}")
        print("Silakan periksa kembali apakah file clean_data.csv sudah ada di laptop kamu.")
        return

    # 4. Memisahkan fitur (X) dengan target Churn (y)
    X = data.drop(columns=['Churn']) 
    y = data['Churn']

    # Membagi data menjadi data Train dan data Test (80:20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. Mulai eksperimen run dengan parameter tetap (TANPA GridSearchCV sesuai ketentuan)
    with mlflow.start_run(run_name="Model_Standar_RandomForest"):
        # Menggunakan parameter tetap bawaan, tidak ada modul tuning di file ini
        model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        
        print("🚀 Melatih model utama dengan autolog lokal...")
        model.fit(X_train, y_train)
        
        # Evaluasi model pada data testing
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        
        print("\n--- Hasil Evaluasi Model ---")
        print(f"Akurasi Model Standar: {acc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        print("✅ Metrik dan model berhasil direkam secara otomatis di MLflow lokal kamu!")

if __name__ == "__main__":
    main()
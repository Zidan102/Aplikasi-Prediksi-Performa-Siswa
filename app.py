import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Aplikasi Prediksi Performa Siswa - Kelompok 4",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk memuat dan memproses data secara otomatis
@st.cache_data
def load_data():
    # Pastikan file 'StudentPerformanceFactors.csv' berada di folder yang sama
    df = pd.read_csv("StudentPerformanceFactors.csv")
    
    # Menangani Missing Values dengan Mode/Imputasi Sederhana sesuai info kelompok 4
    df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
    df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
    df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])
    
    return df

try:
    df_clean = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# Struktur Sidebar Navigasi
st.sidebar.title("Navigation")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard & Evaluasi Model", "Prediksi Nilai Siswa"])

# ====================================================================
# HALAMAN 1: DASHBOARD & EVALUASI MODEL
# ====================================================================
if page == "Dashboard & Evaluasi Model":
    st.title("🎓 Dashboard Faktor Performa Siswa (Analisis Kelompok 4)")
    st.write("Aplikasi ini menganalisis faktor-faktor akademis dan lingkungan yang memengaruhi nilai ujian siswa.")
    
    if data_loaded:
        st.subheader("📊 Sekilas Dataset Awal")
        st.dataframe(df_clean.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Baris Data", df_clean.shape[0])
            st.metric("Jumlah Fitur/Kolom", df_clean.shape[1])
        with col2:
            st.write("**Catatan Penanganan Missing Values (Kelompok 4):**")
            st.write("- `Teacher_Quality` (78 missing values) -> Diimputasi")
            st.write("- `Parental_Education_Level` (90 missing values) -> Diimputasi")
            st.write("- `Distance_from_Home` (67 missing values) -> Diimputasi")
    else:
        st.warning("Silakan letakkan file `StudentPerformanceFactors.csv` di direktori yang sama dengan skrip ini.")

    # TAHAP 4: HASIL EVALUASI MODEL KELOMPOK 4
    st.markdown("---")
    st.subheader("🤖 Hasil Evaluasi Model Machine Learning")
    st.write("Perbandingan performa model regresi berdasarkan nilai MAE dan R2 Score:")
    
    metrics_data = {
        'Model': ['Random Forest', 'XGBoost'],
        'MAE': [1.142194, 0.826520],
        'R2 Score': [0.656468, 0.710897]
    }
    df_eval = pd.DataFrame(metrics_data)
    st.table(df_eval)
    st.success("Kesimpulan: Model XGBoost memberikan performa terbaik dengan MAE terkecil (0.82) dan R2 Score tertinggi (0.71).")

# ====================================================================
# HALAMAN 2: PREDIKSI NILAI SISWA
# ====================================================================
elif page == "Prediksi Nilai Siswa":
    st.title("🔮 Form Prediksi Nilai Ujian Siswa")
    st.write("Masukkan indikator siswa di bawah ini untuk mengestimasi nilai ujian (`Exam_Score`).")
    
    if not data_loaded:
        st.error("Model membutuhkan data `StudentPerformanceFactors.csv` untuk training awal di latar belakang.")
    else:
        # Menyiapkan Model di Latar Belakang menggunakan XGBoost
        @st.cache_resource
        def train_app_model(df):
            X = df.drop(columns=['Exam_Score'])
            y = df['Exam_Score']
            
            # Label Encoding untuk kolom kategorikal
            le_dict = {}
            for col in X.select_dtypes(include=['object']).columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                le_dict[col] = le
                
            model_xgb = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
            model_xgb.fit(X, y)
            return model_xgb, le_dict, X.columns

        model_xgb, le_dict, feature_columns = train_app_model(df_clean)

        # Membuat Form Input untuk Pengguna
        with st.form("prediction_form"):
            st.subheader("📝 Input Data Karakteristik Siswa")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                hours_studied = st.number_input("Hours Studied (Jam Belajar/Minggu)", min_value=0, max_value=50, value=20)
                attendance = st.number_input("Attendance (Persentase Kehadiran %)", min_value=0, max_value=100, value=85)
                previous_scores = st.number_input("Previous Scores (Nilai Ujian Sebelumnya)", min_value=0, max_value=100, value=75)
                sleep_hours = st.slider("Sleep Hours (Jam Tidur Rata-rata)", 4, 12, 7)
                
            with col2:
                parental_involvement = st.selectbox("Parental Involvement", ["Low", "Medium", "High"])
                motivation_level = st.selectbox("Motivation Level", ["Low", "Medium", "High"])
                family_income = st.selectbox("Family Income", ["Low", "Medium", "High"])
                teacher_quality = st.selectbox("Teacher Quality", ["Low", "Medium", "High"])
                
            with col3:
                internet_access = st.selectbox("Internet Access", ["Yes", "No"])
                extracurricular = st.selectbox("Extracurricular Activities", ["Yes", "No"])
                tutoring_sessions = st.number_input("Tutoring Sessions (Jumlah Sesi Les)", min_value=0, max_value=10, value=1)
                gender = st.selectbox("Gender", ["Male", "Female"])

            # Submit Button
            submitted = st.form_submit_button("Prediksi Nilai")
            
            if submitted:
                # Membuat data dictionary dari input
                input_data = {
                    'Hours_Studied': hours_studied,
                    'Attendance': attendance,
                    'Parental_Involvement': parental_involvement,
                    'Access_to_Resources': 'Medium', # default fallback untuk efisiensi form
                    'Extracurricular_Activities': extracurricular,
                    'Sleep_Hours': sleep_hours,
                    'Previous_Scores': previous_scores,
                    'Motivation_Level': motivation_level,
                    'Internet_Access': internet_access,
                    'Tutoring_Sessions': tutoring_sessions,
                    'Family_Income': family_income,
                    'Teacher_Quality': teacher_quality,
                    'School_Type': 'Public', 
                    'Peer_Influence': 'Neutral',
                    'Physical_Activity': 3,
                    'Learning_Disabilities': 'No',
                    'Parental_Education_Level': 'College',
                    'Distance_from_Home': 'Near',
                    'Gender': gender
                }
                
                # Mengubah input menjadi DataFrame
                input_df = pd.DataFrame([input_data])
                
                # Menyamakan Encoding dengan Training Data
                for col in input_df.columns:
                    if col in le_dict:
                        le = le_dict[col]
                        # Menangani jika ada kelas baru yang tidak dikenal saat input
                        if input_df[col].values[0] in le.classes_:
                            input_df[col] = le.transform(input_df[col])
                        else:
                            input_df[col] = 0
                
                # Memastikan urutan kolom sesuai dengan saat training
                input_df = input_df[feature_columns]
                
                # Prediksi menggunakan XGBoost
                predicted_score = model_xgb.predict(input_df)[0]
                
                # Menampilkan Hasil Prediksi
                st.markdown("---")
                st.subheader("🎯 Hasil Estimasi Skor")
                st.metric(label="Prediksi Skor Exam Score", value=f"{predicted_score:.2f}")
                st.info("Prediksi ini diproses menggunakan algoritma XGBoost Regressor Kelompok 4.")

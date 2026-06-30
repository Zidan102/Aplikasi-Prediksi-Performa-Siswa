import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Konfigurasi Halaman Utama
st.set_page_config(
    page_title="Aplikasi Prediksi Performa Siswa - Kelompok 4",
    page_icon="🎓",
    layout="wide"
)

# Fungsi untuk memuat data dan membersihkan missing value
@st.cache_data
def load_data():
    df = pd.read_csv("StudentPerformanceFactors.csv")
    
    # Imputasi Missing Values sesuai analisis Kelompok 4
    df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
    df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
    df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])
    
    return df

try:
    df_clean = load_data()
    data_loaded = True
except FileNotFoundError:
    data_loaded = False

# Navigasi Menu
st.sidebar.title("Navigasi Proyek")
page = st.sidebar.radio("Pilih Halaman:", ["Dashboard & Analisis Data", "Form Input Data Baru"])

# ====================================================================
# PROSES PRE-TRAINING (GLOBAL) AGAR ENCODER KONSISTEN
# ====================================================================
if data_loaded:
    @st.cache_resource
    def train_and_prepare_model(df):
        # Salin dataframe agar tidak merusak data asli
        X = df.drop(columns=['Exam_Score'])
        y = df['Exam_Score']
        
        le_dict = {}
        # Melatih LabelEncoder pada data teks asli secara permanen
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le
            
        # Latih Model XGBoost
        model = xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
        model.fit(X, y)
        return model, le_dict, X.columns

    model_xgb, le_dict, feature_columns = train_and_prepare_model(df_clean)


# ====================================================================
# HALAMAN 1: DASHBOARD
# ====================================================================
if page == "Dashboard & Analisis Data":
    st.title("🎓 Dashboard Faktor Performa Siswa (Analisis Kelompok 4)")
    st.write("Halaman ini menampilkan gambaran dataset yang digunakan untuk melatih model prediksi.")
    
    if data_loaded:
        st.subheader("📊 Sampel Data Eksploratif")
        st.dataframe(df_clean.head(10))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Baris Data", df_clean.shape[0])
            st.metric("Jumlah Fitur/Kolom", df_clean.shape[1])
        with col2:
            st.write("**Hasil Evaluasi Model Regresi:**")
            st.text("- Random Forest MAE: 1.142 | R2: 0.656")
            st.text("- XGBoost Regressor MAE: 0.826 | R2: 0.710 (Terbaik)")
    else:
        st.warning("Letakkan file `StudentPerformanceFactors.csv` di folder yang sama untuk mengaktifkan dashboard.")

# ====================================================================
# HALAMAN 2: FORM INPUT DATA BARU (SUDAH DIPERBAIKI)
# ====================================================================
elif page == "Form Input Data Baru":
    st.title("🔮 Prediksi Interaktif Input Data Baru")
    st.write("Silakan masukkan data karakteristik siswa baru di bawah ini.")
    
    if not data_loaded:
        st.error("Gagal memuat dataset. Silakan pastikan file `StudentPerformanceFactors.csv` tersedia.")
    else:
        with st.form("student_form"):
            st.subheader("📝 Formulir Isian Siswa Baru")
            
            col1, col2, col3 = st.columns(3)
            
            # --- KOLOM 1: INPUT ANGKA ---
            with col1:
                hours_studied = st.number_input("Hours Studied (Jam Belajar per Minggu)", 
                                                min_value=int(df_clean['Hours_Studied'].min()), 
                                                max_value=int(df_clean['Hours_Studied'].max()), value=15)
                attendance = st.slider("Attendance (Persentase Kehadiran %)", 
                                       min_value=int(df_clean['Attendance'].min()), 
                                       max_value=int(df_clean['Attendance'].max()), value=80)
                previous_scores = st.number_input("Previous Scores (Nilai Ujian Sebelumnya)", 
                                                  min_value=int(df_clean['Previous_Scores'].min()), 
                                                  max_value=int(df_clean['Previous_Scores'].max()), value=70)
                sleep_hours = st.slider("Sleep Hours (Rata-rata Jam Tidur)", 
                                        min_value=int(df_clean['Sleep_Hours'].min()), 
                                        max_value=int(df_clean['Sleep_Hours'].max()), value=7)
                physical_activity = st.slider("Physical Activity (Frekuensi Olahraga/Minggu)", 
                                              min_value=int(df_clean['Physical_Activity'].min()), 
                                              max_value=int(df_clean['Physical_Activity'].max()), value=3)

            # --- KOLOM 2: INPUT TEKS KATEGORI ---
            with col2:
                parental_involvement = st.selectbox("Parental Involvement", df_clean['Parental_Involvement'].dropna().unique())
                access_resources = st.selectbox("Access to Resources", df_clean['Access_to_Resources'].dropna().unique())
                extracurricular = st.selectbox("Extracurricular Activities", df_clean['Extracurricular_Activities'].dropna().unique())
                motivation_level = st.selectbox("Motivation Level", df_clean['Motivation_Level'].dropna().unique())
                internet_access = st.selectbox("Internet Access", df_clean['Internet_Access'].dropna().unique())
                family_income = st.selectbox("Family Income", df_clean['Family_Income'].dropna().unique())
                teacher_quality = st.selectbox("Teacher Quality", df_clean['Teacher_Quality'].dropna().unique())

            # --- KOLOM 3: INPUT TEKS KATEGORI LANJUTAN ---
            with col3:
                school_type = st.selectbox("School Type", df_clean['School_Type'].dropna().unique())
                peer_influence = st.selectbox("Peer Influence", df_clean['Peer_Influence'].dropna().unique())
                learning_disabilities = st.selectbox("Learning Disabilities", df_clean['Learning_Disabilities'].dropna().unique())
                parental_education = st.selectbox("Parental Education Level", df_clean['Parental_Education_Level'].dropna().unique())
                distance_home = st.selectbox("Distance from Home", df_clean['Distance_from_Home'].dropna().unique())
                gender = st.selectbox("Gender", df_clean['Gender'].dropna().unique())
                tutoring_sessions = st.number_input("Tutoring Sessions (Sesi Les Tambahan)", 
                                                    min_value=int(df_clean['Tutoring_Sessions'].min()), 
                                                    max_value=int(df_clean['Tutoring_Sessions'].max()), value=0)

            # Tombol kirim data baru
            submit_btn = st.form_submit_button("Hitung Prediksi Nilai")
            
            if submit_btn:
                # Mengumpulkan semua data baru dari form ke bentuk dict
                new_data = {
                    'Hours_Studied': hours_studied,
                    'Attendance': attendance,
                    'Parental_Involvement': parental_involvement,
                    'Access_to_Resources': access_resources,
                    'Extracurricular_Activities': extracurricular,
                    'Sleep_Hours': sleep_hours,
                    'Previous_Scores': previous_scores,
                    'Motivation_Level': motivation_level,
                    'Internet_Access': internet_access,
                    'Tutoring_Sessions': tutoring_sessions,
                    'Family_Income': family_income,
                    'Teacher_Quality': teacher_quality,
                    'School_Type': school_type,
                    'Peer_Influence': peer_influence,
                    'Physical_Activity': physical_activity,
                    'Learning_Disabilities': learning_disabilities,
                    'Parental_Education_Level': parental_education,
                    'Distance_from_Home': distance_home,
                    'Gender': gender
                }
                
                # Konversi menjadi DataFrame 1 baris
                new_df = pd.DataFrame([new_data])
                
                # MENGGUNAKAN ENCODER YANG SAMA KETIKA TRAINING (Mencegah Error Dimensi)
                for col in new_df.columns:
                    if col in le_dict:
                        encoder = le_dict[col]
                        # Ambil nilai teksnya
                        val_str = str(new_df[col].values[0])
                        # Transformasikan berdasarkan kelas yang sudah terdaftar
                        new_df[col] = encoder.transform([val_str])[0]
                
                # Menyelaraskan urutan kolom agar identik dengan model saat training
                new_df = new_df[feature_columns]
                
                # Menghitung Prediksi nilai dengan XGBoost
                hasil_prediksi = model_xgb.predict(new_df)[0]
                
                # Tampilkan hasil akhir ke layar
                st.markdown("---")
                st.subheader("🎯 Hasil Estimasi Skor Nilai")
                st.metric(label="Prediksi Exam Score Siswa Baru", value=f"{hasil_prediksi:.2f}")
                st.success("Berhasil memproses input data baru menggunakan model regresi Kelompok 4 tanpa error!")

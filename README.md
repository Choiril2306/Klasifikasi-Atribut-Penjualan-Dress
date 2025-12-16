# 📘 Judul Proyek
*Prediksi Rekomendasi Dress Menggunakan Model Klasifikasi Machine Learning dan Deep Learning pada Dataset Atribut Penjualan Dress*

## 👤 Informasi
- **Nama:** CHOIRIL ANWAR FAUZY  
- **Repo:** https://github.com/Choiril2306/Klasifikasi-Atribut-Penjualan-Dress  
- **Video:** https://youtu.be/28zPRi2vqHQ  

---

# 1. 🎯 Ringkasan Proyek
Proyek ini bertujuan untuk memprediksi apakah sebuah dress direkomendasikan (Recommendation = 1) atau tidak (0) berdasarkan atribut-atribut produk fashion seperti Style, Price, Size, Season, NeckLine, SleeveLength, Material, Rating, dan lainnya.
- Menyelesaikan permasalahan sesuai domain  
- Melakukan data preparation  
- Membangun 3 model: Baseline, Advanced, Deep Learning  
- Melakukan evaluasi dan menentukan model terbaik  

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
- Dataset Dresses Attribute Sales memiliki banyak atribut produk (style, price, rating, season, dsb.), sehingga perlu diketahui fitur mana yang benar‑benar berpengaruh terhadap tingkat penjualan.  
- Model klasifikasi perlu mampu memprediksi kategori penjualan (sales) dengan akurasi yang baik agar hasilnya dapat digunakan untuk pengambilan keputusan.
- Diperlukan perbandingan performa antara model sederhana, model machine learning, dan model deep learning untuk mengetahui pendekatan mana yang paling efektif.
- Data memiliki variasi nilai dan potensi noise, sehingga perlu preprocessing dan pemilihan fitur agar model lebih stabil dan tidak mudah overfitting.
  
**Goals:**  
- Membangun model klasifikasi untuk memprediksi tingkat penjualan produk fashion berdasarkan atribut produk.
- Melakukan feature selection untuk menentukan fitur yang paling relevan dan meningkatkan performa model.
- Membandingkan performa tiga model: baseline (Logistic Regression), advanced (Random Forest), dan deep learning (MLP).
- Menentukan model terbaik berdasarkan metrik evaluasi seperti accuracy, precision, recall, dan F1‑score. 

---
## 📁 Struktur Folder
```
project/
│
├── data/               
│   ├── Dress Sales.xlsx
│   └── Attribute DataSet.xlsx
│
├── notebooks/            
│   └── 234311036_Choiril Anwar F_ UAS Data Science.ipynb
│
├── src/                    # Source code
│   
├── models/           
│   ├── model_logistic_regression.pkl
│   ├── model_random_forest.pkl
│   └── model_deep_learning(mlp).pkl
│
├── images/    
│   ├── Bivariate Analysis (Hubungan Fitur dengan Target).png
│   ├── BOXPLOT FITUR NUMERIK vs TARGET.png
│   ├── Confusion Matrix-Deep Learning.png
│   ├── Confusion Matrix-Logistic Regression.png
│   ├── Confusion Matrix-Random Forest.png
│   ├── DISTRIBUSI FITUR NUMERIK.png
│   ├── Feature Importance.png
│   ├── Heatmap Korelasi Antar Fitur Numerik.png
│   ├── Training History (Accuracy).png
│   ├── Training History (Loss).png           
│   └── Univariate Analysis (Distribusi Setiap Fitur).png
│
├── requirements.txt        # Dependencies
├── .gitignore
├── LICENSE
├── Checklist Submit.md
├── 234311036_Choiril Anwar Fauzy_Laporan Dress Data.docx
└── README.md
```
---

# 3. 📊 Dataset
- **Sumber:** UCI Machine Learning Repository - https://archive.ics.uci.edu/dataset/289/dresses+attribute+sales  
- **Jumlah Data:** 501 instances → setelah digabungkan dan dibersihkan menjadi sekitar 479 baris → 133 instances (setelah dropna final) 
- **Jumlah Kolom:** 13 atribut utama (termasuk Dress_ID dan Recommendation) + kolom sales harian
- **Ukuran Dataset:** Sekitar 5.6 MB (file RAR/ZIP asli dari UCI)
- **Format File:** Excel (.xlsx) – terdiri dari dua file: "Attribute DataSet.xlsx" dan "Dress Sales.xlsx", digabung berdasarkan Dress_ID
- **Tipe:** Tabular  

## 📌 Fitur Utama Dataset Dress Sales

| Nama Fitur        | Tipe Data    | Deskripsi                                                         | Contoh Nilai                                   |
|------------------|-------------|-------------------------------------------------------------------|-----------------------------------------------|
| Dress_ID         | Integer     | ID unik untuk setiap dress / item produk                          | 1006032852, 1212192089, 1190380701             |
| Style            | Categorical | Gaya atau model dress                                             | Sexy, Casual, Vintage, Brief, Cute            |
| Price            | Categorical | Kategori harga dress                                              | Low, Average, Medium, High, Very-High          |
| Rating           | Float       | Rating atau penilaian dress (skala 0–5)                           | 4.6, 3.5, 4.0, 0.0                            |
| Size             | Categorical | Ukuran dress                                                      | S, M, L, XL, Free                             |
| Season           | Categorical | Musim yang cocok untuk dress                                      | Autumn, Winter, Spring, Summer                |
| NeckLine         | Categorical | Jenis garis leher (neckline)                                      | O-neck, V-neck, Sweetheart, Scoop             |
| SleeveLength     | Categorical | Panjang lengan                                                    | Full, Short, Sleeveless, Half                 |
| Waistline        | Categorical | Jenis garis pinggang (waistline)                                  | Natural, Empire, Dropped, Princess            |
| Material         | Categorical | Bahan utama dress                                                 | Cotton, Polyester, Silk, Mix                  |
| FabricType       | Categorical | Jenis kain / fabric                                               | Chiffon, Satin, Jersey, Knitted               |
| Decoration       | Categorical | Elemen dekorasi pada dress                                        | Bow, Ruffles, Embroidery, Beading             |
| Pattern_Type     | Categorical | Jenis pola atau motif dress                                       | Solid, Print, Dot, Animal                     |
| Recommendation   | Binary      | Label target: dress direkomendasikan (1) atau tidak (0)           | 0, 1                                         |

---

# 4. 🔧 Data Preparation
- Cleaning : Pengecekan missing values, duplicates, outliers, noise
- Transformasi : Melakukan encoding, scaling, feature selection
- Splitting : Train / test / stratified split untuk maintain class distribution

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Logistic Regression 
- **Model 2 – Advanced ML:** Random Forest Classifier  
- **Model 3 – Deep Learning:** Multilayer Perceptron (MLP)

---

# 6. 🧪 Evaluation
**Metrik:** Accuracy / F1 / MAE / MSE (pilih sesuai tugas)

### Hasil Singkat
| Model | Score | Catatan |
|-------|--------|---------|
| Baseline (LR) | 0.7692 |Model terbaik - performa optimal dengan efisiensi tertinggi, cocok untuk deployment production |
| Advanced (RF)| 0.7308 |Performa lebih rendah tanpa feature selection |
| Deep Learning (MLP) | 0.7692 |Performa setara baseline, tapi 1500x lebih lambat |

---

# 7. 🏁 Kesimpulan
- Model terbaik: Logistic Regression (Baseline) adalah model terbaik untuk deployment.
- Alasan: Model paling sederhana, mudah diinterpretasi dan maintain. Serta performa optimal dengan efisiensi tertinggi, 1500x lebih cepat dari MLP 
sambil mempertahankan akurasi yang sama.  
- Insight penting: Kompleksitas tidak selalu lebih baik. Model linear sederhana 
setara dengan deep learning pada dataset kecil. Feature selection krusial (100+ → 50 fitur). 
Recall 90.91% sangat reliable untuk sistem rekomendasi.  

---

# 8. 🔮 Future Work
Data : 
- [ ] Mengumpulkan lebih banyak data
- [x] Menambah variasi data
- [x] Feature engineering lebih lanjut 

Model :
- [x] Mencoba arsitektur DL yang lebih kompleks
- [x] Hyperparameter tuning lebih ekstensif 
- [ ] Ensemble methods (combining models)
- [ ] Transfer learning dengan model yang lebih besar
- [x] Model compression (pruning, quantization)

Deployment :
- [x] Membuat API (Flask/FastAPI) 
- [x] Membuat web application (Streamlit/Gradio) 
- [ ] Containerization dengan Docker
- [ ] Deploy ke cloud (Heroku, GCP, AWS)

Optimization : 
- [x] Model compression (pruning, quantization)
- [ ] Improving inference speed
- [ ] Reducing model size 
---

# 9. 🔁 Reproducibility
🧪 Environment
- Python Version: 3.10+
- Platform: Google Colab / Local Machine
- Hardware: CPU (no GPU required)

📦 Libraries & Dependencies
- txtnumpy==1.24.3
- pandas==2.0.3
- scikit-learn==1.3.0
- imbalanced-learn==0.11.0
- tensorflow==2.14.0
- keras==2.14.0
- matplotlib==3.7.2
- seaborn==0.12.2
- joblib==1.3.2
- tabulate==0.9.0

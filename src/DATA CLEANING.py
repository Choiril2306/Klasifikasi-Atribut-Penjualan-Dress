# **DATA CLEANING**

# CEK SEMUA KOLOM YANG ADA
print("Daftar kolom asli:")
df_raw = df.copy()
print(df_raw.columns.tolist())

#cek tipedata
df.dtypes

print(df.head())

# DAFTAR FITUR ATRIBUT YANG RELEVAN UNTUK MODELING
fitur_atribut = [
    'Style', 'Price', 'Rating', 'Size', 'Season',
    'NeckLine', 'SleeveLength', 'Material',
    'FabricType', 'Decoration', 'Recommendation'
]

# FILTER HANYA KOLOM YANG BENAR-BENAR ADA DI DATASET
fitur_valid = [col for col in fitur_atribut if col in df_raw.columns]

print("\nKolom atribut yang ditemukan:")
print(fitur_valid)

# MEMBUAT DATASET ATRIBUT SAJA
# (menghapus kolom tanggal, Dress_ID, pivot penjualan, dll)
df_clean = df_raw[fitur_valid].copy()

print("\nDataset atribut berhasil dibuat!")
print("Ukuran dataset:", df_clean.shape)
print("Kolom:", df_clean.columns.tolist())

# CEK MISSING VALUE SEBELUM CLEANING
print("MISSING VALUE SEBELUM CLEANING")
print(df_raw.isnull().sum())
print("\nTotal missing value:", df_raw.isnull().sum().sum())


# TANGANI MISSING VALUE
df_clean = df_clean.dropna()

print("\nSetelah menangani missing value:")
print(df_clean.isnull().sum())


# Cek duplikasi
print("Jumlah duplikasi:", df.duplicated().sum())
df = df.drop_duplicates()
print("Duplikasi setelah dibersihkan:", df.duplicated().sum())

# Cek outlier (IQR) untuk kolom numerik
# Ambil kolom numerik
kolom_numerik = df_clean.select_dtypes(include=['int64', 'float64']).columns
print("Kolom numerik:", kolom_numerik.tolist())
for kolom in kolom_numerik:
    Q1 = df_clean[kolom].quantile(0.25)
    Q3 = df_clean[kolom].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outlier_count = df_clean[(df_clean[kolom] < lower) | (df_clean[kolom] > upper)].shape[0]
    print(f"\nOutlier pada kolom {kolom}: {outlier_count} baris")

# Cek imbalance target
print("CEK IMBALANCE TARGET")
print(df_clean['Recommendation'].value_counts())
print("\nPersentase:")
print(df_clean['Recommendation'].value_counts(normalize=True) * 100)

# Cek noise (nilai aneh)
for kolom in df_clean.select_dtypes(include=['object']).columns:
    print(f"Nilai unik kolom {kolom}")
    print(df_clean[kolom].unique())

# Normalisasi
scaler = MinMaxScaler()
df_clean['Rating_norm'] = scaler.fit_transform(df_clean[['Rating']])
print("\nRating setelah normalisasi:")
print(df_clean[['Rating', 'Rating_norm']].head())

# HASIL AKHIR DATA CLEANING
print("DATA CLEANING SELESAI")
print("Ukuran final dataset:", df_clean.shape)
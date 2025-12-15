# **FEATURE ENGINEERING**

# 1. CREATING NEW FEATURES
print("1. CREATING NEW FEATURES")

# Initialize df_fe from df_clean
df_fe = df_clean.copy()

# Membuat fitur baru: Rating Category
# Kategorisasi Rating menjadi Low, Medium, High
def categorize_rating(rating):
    if rating < 3.5:
        return 'Low'
    elif rating < 4.5:
        return 'Medium'
    else:
        return 'High'

df_fe['Rating_Category'] = df_fe['Rating'].apply(categorize_rating)
print("Fitur baru 'Rating_Category' berhasil dibuat")
print(f"  Kategori: {df_fe['Rating_Category'].unique()}")

# Membuat fitur baru: Price Level (kombinasi Price dengan informasi lain)
df_fe['Price_Season'] = df_fe['Price'].astype(str) + '_' + df_fe['Season'].astype(str)
print("Fitur baru 'Price_Season' berhasil dibuat")
print(f"  Jumlah kombinasi: {df_fe['Price_Season'].nunique()}")

print(f"Total fitur setelah creating: {df_fe.shape[1]} kolom")

# 2. FEATURE EXTRACTION
print("\n2. FEATURE EXTRACTION")

# Perbaiki nama kolom yang salah ejaan
print("Memperbaiki nama kolom yang typo...")
df_fe.rename(columns={
    "waiseline": "WaistLine",
    "Pattern Type": "PatternType"
}, inplace=True)
print("  - 'waiseline' → 'WaistLine'")
print("  - 'Pattern Type' → 'PatternType'")

# Standarisasi kapitalisasi kategori
print("Standarisasi kapitalisasi pada fitur kategorikal...")
categorical_cols = [
    "Style", "Price", "Size", "Season", "NeckLine",
    "SleeveLength", "WaistLine", "Material",
    "FabricType", "Decoration", "PatternType"
]

for col in categorical_cols:
    if col in df_fe.columns:
        df_fe[col] = df_fe[col].astype(str).str.strip().str.title()

print(f"  - {len(categorical_cols)} kolom kategorikal telah distandarisasi")

# Pastikan Rating numerik
print("Konversi Rating ke tipe numerik...")
df_fe["Rating"] = pd.to_numeric(df_fe["Rating"], errors="coerce")
print(f"Tipe data Rating: {df_fe['Rating'].dtype}")

print(f"Total fitur setelah extraction: {df_fe.shape[1]} kolom")

# 3. DIMENSIONALITY REDUCTION
print("\n3. DIMENSIONALITY REDUCTION")

# DROP KOLOM TANGGAL yang tidak relevan
print("Menghapus kolom tanggal yang tidak relevan...")
date_cols = [
    col for col in df_fe.columns if isinstance(col, str) and "/" in col
] + [
    col for col in df_fe.columns if not isinstance(col, str)
]

df_clean = df_fe.drop(columns=date_cols)
print(f"  - Jumlah kolom tanggal dihapus: {len(date_cols)}")

# Filter hanya kolom atribut yang relevan
print("Filter kolom atribut yang relevan untuk modeling...")
fitur_atribut = [
    'Style', 'Price', 'Rating', 'Size', 'Season',
    'NeckLine', 'SleeveLength', 'WaistLine', 'Material',
    'FabricType', 'Decoration', 'PatternType', 'Recommendation',
    'Rating_Category', 'Price_Season'  # Fitur baru
]

fitur_valid = [col for col in fitur_atribut if col in df_clean.columns]
df_clean = df_clean[fitur_valid].copy()

print(f"Jumlah fitur yang dipertahankan: {len(fitur_valid)}")
print(f"Total fitur setelah reduction: {df_clean.shape[1]} kolom")

# 4. FEATURE SELECTION (akan dilakukan setelah encoding)
print("\n4. FEATURE SELECTION")
print("Feature Selection akan dilakukan setelah One-Hot Encoding")
print("  - Metode: SelectKBest dengan Chi-Square")
print("  - Target: Pilih 50 fitur terbaik")

# RINGKASAN FEATURE ENGINEERING
print("FEATURE ENGINEERING SELESAI")
print(f"Ukuran dataset akhir: {df_clean.shape}")
print(f"Jumlah fitur: {df_clean.shape[1]} kolom")
print(f"Jumlah baris: {df_clean.shape[0]} baris")
print("\nKolom yang tersedia:")
for i, col in enumerate(df_clean.columns, 1):
    print(f"  {i}. {col}")

# Tampilkan 5 baris pertama
print("\n5 Baris Pertama Dataset:")
print(df_clean.head())
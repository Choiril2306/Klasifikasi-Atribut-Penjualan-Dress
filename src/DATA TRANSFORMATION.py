# **DATA TRANSFORMATION**

# DROP KOLOM TANGGAL
date_cols = [
    col for col in df_fe.columns if isinstance(col, str) and "/" in col
] + [
    col for col in df_fe.columns if not isinstance(col, str) and not isinstance(col, pd.Timestamp)
]

df_clean = df_fe.drop(columns=date_cols, errors='ignore')

print("Kolom tanggal dihapus:", len(date_cols))

# Normalisasi Rating dan penambahan Rating_norm
scaler = MinMaxScaler()
df_clean['Rating_norm'] = scaler.fit_transform(df_clean[['Rating']])

print("\nRating setelah normalisasi:")
print(df_clean[['Rating', 'Rating_norm']].head())

#  Handle original Rating column
if 'Rating' in df_clean.columns and 'Rating_norm' in df_clean.columns:
    df_clean = df_clean.drop(columns=['Rating'])
    print("\nDropped original 'Rating' column from df_clean as 'Rating_norm' is available and will be used.")

# --- DATA SPLITTING ---
# X = semua fitur kecuali target
X = df_clean.drop("Recommendation", axis=1)
# y = target / label
y = df_clean["Recommendation"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nData Splitting selesai.")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)
# --- END DATA SPLITTING ---


# IDENTIFIKASI FITUR UNTUK PREPROCESSOR
numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print("\nKolom numerikal yang dipakai untuk pipeline:", numerical_cols)
print("Kolom kategorikal yang dipakai untuk pipeline:", categorical_cols)

# PIPELINE NUMERIK
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', MinMaxScaler())
])

# PIPELINE KATEGORIKAL
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# GABUNGKAN PIPELINE
preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# TRANSFORMASI
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("\nData Transformation selesai.")
print("Jumlah fitur setelah OHE:", X_train_processed.shape[1])
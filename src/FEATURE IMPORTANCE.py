# **FEATURE IMPORTANCE**

# Ambil nama fitur setelah ColumnTransformer (ini akan mencakup semua fitur numerik dan OHE)
feature_names_after_preprocessing = preprocessor.get_feature_names_out()

# Debugging: Print lengths to verify consistency
print(f"Length of feature_names_after_preprocessing: {len(feature_names_after_preprocessing)}")
print(f"Length of selector.get_support() (boolean mask): {len(selector.get_support())}")

# FIX: Samakan panjang mask dengan jumlah fitur setelah preprocessing
mask = selector.get_support()[:len(feature_names_after_preprocessing)]

# Ambil nama fitur yang dipilih SelectKBest dari daftar feature_names_after_preprocessing
selected_features = feature_names_after_preprocessing[mask]

# Latih model Random Forest baru pada fitur-fitur yang sudah diseleksi (X_train_fs)
rf_fs = RandomForestClassifier(random_state=42)
rf_fs.fit(X_train_fs, y_train)

# Ambil importance dari model Random Forest yang baru dilatih
feature_importance = rf_fs.feature_importances_

# Buat DataFrame
feat_imp = pd.DataFrame({
    'Feature': selected_features,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Visualisasi Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(
    data=feat_imp,
    x='Importance',
    y='Feature',
    palette='viridis'
)
plt.title("Feature Importance (Random Forest + Selected Features)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()
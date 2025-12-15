# **MEMISAHKAN FITUR DAN TARGET (LABEL)**

# X = semua fitur kecuali target
X = df_clean.drop("Recommendation", axis=1)
# y = target / label
y = df_clean["Recommendation"]

print("PEMISAHAN FITUR & TARGET BERHASIL")
print("Ukuran X:", X.shape)
print("Ukuran y:", y.shape)
print("\n5 baris pertama X:")
print(X.head())
print("\n5 baris pertama y:")
print(y.head())
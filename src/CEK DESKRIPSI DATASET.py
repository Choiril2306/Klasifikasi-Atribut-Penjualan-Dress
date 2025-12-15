# **CEK DESKRIPSI DATASET**

target = "Recommendation"
fitur = X.columns.tolist()   # X = fitur setelah cleaning

print(f"Targetnya    : {target}")
print(f"Nama Fitur   : {fitur}")
print(f"Jumlah Fitur : {len(fitur)} kolom")
print(f"Jumlah baris : {len(df_clean)} baris")
print(f"Ukuran Data  : {df_clean.shape}")

# Cek tipe data
print("\nCek Tipe Data :")
display(df_clean.dtypes)
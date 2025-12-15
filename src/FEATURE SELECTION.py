# **FEATURE SELECTION**

selector = SelectKBest(score_func=chi2, k=50)

X_train_fs = selector.fit_transform(X_train_processed, y_train)
X_test_fs = selector.transform(X_test_processed)

print("Feature Selection selesai.")
print("Jumlah fitur terpilih:", X_train_fs.shape[1])
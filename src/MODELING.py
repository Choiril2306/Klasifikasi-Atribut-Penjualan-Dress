# **MODELING**

# EVALUASI
def evaluate_model(y_true, y_pred, model_name, train_time):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Training Time (s): {train_time:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    return acc, prec, rec, f1

# VISUALISASI MODEL
def show_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix — {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# Logistic Regression
start = time.time()
logreg = LogisticRegression(max_iter=500)
logreg.fit(X_train_fs, y_train)
time_logreg = time.time() - start

y_pred_logreg = logreg.predict(X_test_fs)
evaluate_model(y_test, y_pred_logreg, "Logistic Regression", time_logreg)
show_confusion_matrix(y_test, y_pred_logreg, "Logistic Regression")

# Calculate and store metrics for comparison
acc_logreg = accuracy_score(y_test, y_pred_logreg)
prec_logreg = precision_score(y_test, y_pred_logreg)
rec_logreg = recall_score(y_test, y_pred_logreg)
f1_logreg = f1_score(y_test, y_pred_logreg)

# Save the trained Logistic Regression model
joblib.dump(logreg, "model_logistic_regression.pkl")

# Download the saved model file
files.download("model_logistic_regression.pkl")

# Random Forest
start = time.time()

rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    bootstrap=True,
    random_state=42
)

rf.fit(X_train_processed, y_train)
time_rf = time.time() - start

# Prediksi
y_pred_rf = rf.predict(X_test_processed)

# Evaluasi
evaluate_model(y_test, y_pred_rf, "Random Forest", time_rf)
show_confusion_matrix(y_test, y_pred_rf, "Random Forest")

# Simpan metrik untuk perbandingan
acc_rf = accuracy_score(y_test, y_pred_rf)
prec_rf = precision_score(y_test, y_pred_rf)
rec_rf = recall_score(y_test, y_pred_rf)
f1_rf = f1_score(y_test, y_pred_rf)

# Save the trained Random Forest model
joblib.dump(rf, "model_random_forest.pkl")

# Download the saved model file
files.download("model_random_forest.pkl")

# Model MLP
start = time.time()
model_fs = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_fs.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
model_fs.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
history_fs = model_fs.fit(
    X_train_fs, y_train,
    validation_split=0.2,
    epochs=60,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)
train_time_fs = time.time() - start

# Model Summary
print("Model Summary for model_fs")
model_fs.summary()

# Plot Accuracy
plt.plot(history_fs.history['accuracy'], label='Train Acc')
plt.plot(history_fs.history['val_accuracy'], label='Val Acc')
plt.title("Training History (Accuracy) — FS")
plt.legend()
plt.show()

# Plot Loss
plt.plot(history_fs.history['loss'], label='Train Loss')
plt.plot(history_fs.history['val_loss'], label='Val Loss')
plt.title("Training History (Loss) — FS")
plt.legend()
plt.show()

# Evaluasi
y_pred_fs = (model_fs.predict(X_test_fs) > 0.5).astype(int)
evaluate_model(y_test, y_pred_fs, "Deep Learning (MLP) — Feature Selection", train_time_fs)
show_confusion_matrix(y_test, y_pred_rf, "")

# Save the trained Random Forest model
joblib.dump(rf, "model_deep_learning(mlp).pkl")

# Download the saved model file
files.download("model_deep_learning(mlp).pkl")

# PERBANDINGAN SEMUA MODEL
acc_mlp = accuracy_score(y_test, y_pred_fs)
prec_mlp = precision_score(y_test, y_pred_fs)
rec_mlp = recall_score(y_test, y_pred_fs)
f1_mlp = f1_score(y_test, y_pred_fs)

comparison_data = {
    "Model": [
        "Baseline – Logistic Regression",
        "Advanced – Random Forest",
        "Deep Learning – MLP (Feature Selection)"
    ],
    "Accuracy": [ acc_logreg, acc_rf, acc_mlp ],
    "Precision": [ prec_logreg, prec_rf, prec_mlp ],
    "Recall": [ rec_logreg, rec_rf, rec_mlp ],
    "F1-Score": [ f1_logreg, f1_rf, f1_mlp ],
    "Training Time (s)": [ time_logreg, time_rf, train_time_fs ]
}

df_compare = pd.DataFrame(comparison_data)
print("PERBANDINGAN SEMUA MODEL")
print(tabulate(df_compare, headers='keys', tablefmt='grid', showindex=False))

# MENAMPILKAN PREDIKSI
# Ambil probabilitas mentah dari model
y_prob_fs = model_fs.predict(X_test_fs).flatten()
y_pred_fs = (y_prob_fs > 0.5).astype(int)

# Buat DataFrame hasil
results = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred_fs,
    'Probability (Class 1)': y_prob_fs.round(4)
})

# Tampilkan 10 baris pertama
print("\nContoh Hasil Prediksi Deep Learning:")
print(results.head(10).to_markdown(index=False))
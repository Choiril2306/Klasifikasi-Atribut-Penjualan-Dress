# **EXPLORATORY DATA ANALYSIS (EDA)**

# Univariate Analysis (Distribusi Setiap Fitur)
plt.figure(figsize=(14, 18))

# Distribusi Target Recommendation
plt.subplot(3, 2, 1)
sns.countplot(x="Recommendation", data=df)
plt.title("Distribusi Target Recommendation")

# Distribusi Season
plt.subplot(3, 2, 2)
sns.countplot(x="Season", data=df)
plt.title("Distribusi Season")

# Distribusi Price
plt.subplot(3, 2, 3)
sns.countplot(x="Price", data=df)
plt.title("Distribusi Price")

# Distribusi Style
plt.subplot(3, 2, 4)
sns.countplot(x="Style", data=df)
plt.title("Distribusi Style")
plt.xticks(rotation=45)

# Distribusi Material
plt.subplot(3, 2, 5)
sns.countplot(x="Material", data=df)
plt.title("Distribusi Material")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# Bivariate Analysis (Hubungan Fitur dengan Target)
plt.figure(figsize=(14, 18))  # ukuran besar agar tidak dempet

# Season vs Recommendation
plt.subplot(3, 2, 1)
sns.countplot(x="Season", hue="Recommendation", data=df)
plt.title("Season vs Recommendation")

# Price vs Recommendation
plt.subplot(3, 2, 2)
sns.countplot(x="Price", hue="Recommendation", data=df)
plt.title("Price vs Recommendation")

# Style vs Recommendation
plt.subplot(3, 2, 3)
sns.countplot(x="Style", hue="Recommendation", data=df)
plt.title("Style vs Recommendation")
plt.xticks(rotation=45);

# Material vs Recommendation
plt.subplot(3, 2, 4)
sns.countplot(x="Material", hue="Recommendation", data=df)
plt.title("Material vs Recommendation")
plt.xticks(rotation=45);

df_corr = df.select_dtypes(include='number')

plt.figure(figsize=(10, 6))
sns.heatmap(df_corr.corr(), cmap='coolwarm', annot=True, fmt=".2f")
plt.title("Heatmap Korelasi Antar Fitur Numerik")
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# BOXPLOT FITUR NUMERIK vs TARGET
plt.figure(figsize=(6,4))
sns.boxplot(x="Recommendation", y="Rating", data=df)
plt.title("Rating vs Recommendation (Boxplot)")
plt.show()

# DISTRIBUSI FITUR NUMERIK
if "Sales" in df.columns:
    plt.figure(figsize=(6,4))
    sns.histplot(df["Sales"], bins=10, kde=True)
    plt.title("Distribusi Sales")
    plt.show()
else:
    plt.figure(figsize=(6,4))
    sns.histplot(df["Rating"], bins=10, kde=True)
    plt.title("Distribusi Rating")
    plt.show()

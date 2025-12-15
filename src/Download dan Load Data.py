# **DOWNLOAD DATASET dan LOAD DATASET**

# Download file ZIP dari UCI dan simpan sebagai dresses.zip
wget https://archive.ics.uci.edu/static/public/289/dresses+attribute+sales.zip -O dresses.zip

# Ekstrak ZIP → hasilnya file RAR
!unzip dresses.zip -d dresses_data

# Install unrar (kalau belum ada)
!apt-get install unrar -y

# Ekstrak file RAR → hasilnya file Excel/CSV
!unrar x dresses_data/Dresses_Attribute_Sales.rar dresses_data/

# LOAD DATASET

# Load file atribut
df_attr = pd.read_excel("/content/dresses_data/Dresses_Attribute_Sales/Attribute DataSet.xlsx")

# Load file sales
df_sales = pd.read_excel("/content/dresses_data/Dresses_Attribute_Sales/Extra files/Dress Sales.xlsx")

# Gabungkan dataset berdasarkan Dress_ID
df = pd.merge(df_attr, df_sales, on="Dress_ID", how="inner")
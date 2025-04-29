# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Veriyi yükle
df = pd.read_csv('data/bike_rentals.csv')

# Giriş ve çıkış değişkenlerini ayır
X = df[['Sıcaklık (°C)', 'Nem (%)', 'Rüzgar (km/h)', 'Gun']]
y = df['Kiralanan Bisiklet']

# Veriyi eğitim ve test kümelerine böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli oluştur ve eğit
model = LinearRegression()
model.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = model.predict(X_test)

# Model performansını yazdır
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Modeli kaydet
joblib.dump(model, 'models/bike_rental_model.pkl')
print('Model saved as models/bike_rental_model.pkl')

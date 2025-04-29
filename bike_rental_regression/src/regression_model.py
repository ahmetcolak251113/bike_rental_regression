# regression_model.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Read Data
df = pd.read_csv('/Users/ahmetcolak/Desktop/Yapay_Zeka_ve_Bilgisayarlı_Goru/3 Öğrenen Makineler/bike_rental_regression/data/bike_rentals.csv')

# Virtualize
sns.pairplot(df)
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# Inputs and Outputs
X = df[['Sıcaklık (°C)', 'Nem (%)', 'Rüzgar (km/h)', 'Gun']]
y = df['Kiralanan Bisiklet']

# Train and test datas
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Model Performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
print(f'R² Score: {r2:.2f}')

# Real and prediction values
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel('Gerçek Kiralanan Bisiklet Sayısı')
plt.ylabel('Tahmin Edilen Kiralanan Bisiklet Sayısı')
plt.title('Gerçek vs Tahmin Edilen Değerler')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.show()

# Save Model
joblib.dump(model, '/Users/ahmetcolak/Desktop/Yapay_Zeka_ve_Bilgisayarlı_Goru/3 Öğrenen Makineler/bike_rental_regression/models/bike_rental_model.pkl')
print('Model kaydedildi: models/bike_rental_model.pkl')

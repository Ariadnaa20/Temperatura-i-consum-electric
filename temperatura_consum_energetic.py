# ==============================================================
# Projecte integrador — Temperatura i consum elèctric
# Autor: Ariadna Pasqual
# Descripció: Anàlisi de la relació entre la temperatura i el
# consum elèctric a Espanya mitjançant regressió lineal i polinòmica.
# ==============================================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
import seaborn as sns
import math

# ==============================================================
# 1️⃣ Càrrega de dades
# ==============================================================

energy_path = "energy_dataset.csv"
weather_path = "weather_features.csv"

energy = pd.read_csv(energy_path)
weather = pd.read_csv(weather_path)

print("\n✅ Archivos cargados correctamente.")
print("\nColumnas de energy_dataset.csv:")
print(energy.columns)
print("\nColumnas de weather_features.csv:")
print(weather.columns)

# ==============================================================
# 2️⃣ Preparació de les dades
# ==============================================================

energy['time'] = pd.to_datetime(energy['time'], utc=True, errors='coerce')
weather['time'] = pd.to_datetime(weather['dt_iso'], utc=True, errors='coerce')

energy = energy.dropna(subset=['time'])
weather = weather.dropna(subset=['time'])

# Agreguem dades per dia
energy_daily = energy.groupby(energy['time'].dt.date)['total load actual'].sum().reset_index()
weather_daily = weather.groupby(weather['time'].dt.date)['temp'].mean().reset_index()

# Unim els dos conjunts
data = pd.merge(energy_daily, weather_daily, left_on='time', right_on='time')
data.columns = ['Date', 'EnergyConsumption', 'Temperature']

print("\n✅ Dades processades correctament:")
print(data.head())

# ==============================================================
# 3️⃣ Exploració inicial de les dades
# ==============================================================

print(f"\nEl DataFrame té {data.shape[0]} files i {data.shape[1]} columnes.")

plt.figure(figsize=(8,5))
plt.scatter(data['Temperature'], data['EnergyConsumption'], alpha=0.6)
plt.title("Temperatura vs Consum elèctric")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Consum elèctric (MWh)")
plt.grid(True)
plt.show()

# ==============================================================
# 4️⃣ Entrenament del model de regressió lineal
# ==============================================================

X = data[['Temperature']]
y = data['EnergyConsumption']

model = LinearRegression()
model.fit(X, y)

w = model.coef_[0]
b = model.intercept_

print(f"\nCoeficient (pendent w): {w:.2f}")
print(f"Intercept (b): {b:.2f}")

# ==============================================================
# 5️⃣ Avaluació del model
# ==============================================================

y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
rmse = math.sqrt(mse)

print("\n📊 AVALUACIÓ DEL MODEL LINEAL:")
print(f" - MSE (Error cuadrático medio): {mse:.2f}")
print(f" - RMSE: {rmse:.2f} MWh")
print(f" - R² (Coeficiente de determinación): {r2:.4f}")

# Interpretació automàtica
if r2 < 0.3:
    print("   ➤ El model lineal explica molt poca variabilitat del consum. La relació pot no ser lineal.")
elif r2 < 0.7:
    print("   ➤ El model explica part de la variabilitat, però encara hi ha molts factors externs.")
else:
    print("   ➤ El model explica bé la variació del consum segons la temperatura.")

# ==============================================================
# 6️⃣ Visualització dels resultats — Model Lineal
# ==============================================================

plt.figure(figsize=(8,5))
plt.scatter(X, y, label='Dades reals', alpha=0.6)
plt.plot(X, y_pred, color='red', label='Model lineal')
plt.title("Ajust de regressió lineal")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Consum elèctric (MWh)")
plt.legend()

text_metrics = f"RMSE = {rmse:.2f}     R² = {r2:.4f}"
plt.text(
    0.5, -0.18, text_metrics,
    ha='center', va='center', transform=plt.gca().transAxes,
    fontsize=11, color='black',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
)

plt.tight_layout()
plt.show()

# ==============================================================
# 7️⃣ Regressió polinòmica (grau 2)
# ==============================================================

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly_pred = poly_model.predict(X_poly)

mse_poly = mean_squared_error(y, y_poly_pred)
rmse_poly = math.sqrt(mse_poly)
r2_poly = r2_score(y, y_poly_pred)

print("\n📊 AVALUACIÓ DEL MODEL POLINÒMIC (GRAU 2):")
print(f" - MSE: {mse_poly:.2f}")
print(f" - RMSE: {rmse_poly:.2f} MWh")
print(f" - R²: {r2_poly:.4f}")

plt.figure(figsize=(8,5))
plt.scatter(X, y, alpha=0.6, label='Dades reals')
plt.plot(
    np.sort(X.values, axis=0),
    y_poly_pred[np.argsort(X.values[:, 0])],
    color='orange', label='Model polinòmic (grau 2)'
)
plt.title("Regressió polinòmica (grau 2)")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Consum elèctric (MWh)")
plt.legend()

# 🔹 Mostrem RMSE i R² dins del gràfic
text_metrics_poly = f"RMSE = {rmse_poly:.2f}     R² = {r2_poly:.4f}"
plt.text(
    0.5, -0.18, text_metrics_poly,
    ha='center', va='center', transform=plt.gca().transAxes,
    fontsize=11, color='black',
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
)

plt.tight_layout()
plt.show()

# ==============================================================
# 8️⃣ Avaluació avançada i residus
# ==============================================================

pred_0 = model.predict([[0]])[0]
print(f"\nPredicció del consum si Temperatura = 0°C: {pred_0:.2f} MWh")

residuals = y - y_pred
print(f"\nEstadístiques dels residus:")
print(f" mitjana: {residuals.mean():.2f}")
print(f" desviació estàndard: {residuals.std():.2f}")
print(f" mínim: {residuals.min():.2f}")
print(f" màxim: {residuals.max():.2f}")

plt.figure(figsize=(8,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residus vs Prediccions")
plt.xlabel("Predicció (MWh)")
plt.ylabel("Residus (MWh)")
plt.grid(True)
plt.show()

plt.figure(figsize=(8,4))
plt.hist(residuals, bins=40, edgecolor='k', alpha=0.7)
plt.title("Histograma de residus")
plt.xlabel("Residual (MWh)")
plt.ylabel("Freqüència")
plt.show()

plt.figure(figsize=(8,4))
plt.scatter(data['Temperature'], residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.title("Residus vs Temperatura")
plt.xlabel("Temperatura (°C)")
plt.ylabel("Residus (MWh)")
plt.grid(True)
plt.show()

# ==============================================================
# 9️⃣ Validació creuada
# ==============================================================

kf = KFold(n_splits=5, shuffle=True, random_state=42)
neg_mse_scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=kf)
mse_cv = -neg_mse_scores.mean()
rmse_cv = math.sqrt(mse_cv)
r2_cv_scores = cross_val_score(model, X, y, scoring='r2', cv=kf)
r2_cv = r2_cv_scores.mean()

print(f"\nValidació creuada (5-fold):")
print(f" RMSE_cv: {rmse_cv:.2f} MWh")
print(f" R²_cv: {r2_cv:.4f}")

# ==============================================================
# 🔟 Sugeriments segons rendiment
# ==============================================================

print("\nSugeriment:")
if r2 < 0.4:
    print(" - R² baix: considera afegir més variables (estacionalitat, dia de la setmana), features polinòmiques o models no lineals.")
elif r2 < 0.7:
    print(" - R² moderat: el model captura part de la variabilitat, però pot millorar-se amb més dades o variables.")
else:
    print(" - R² alt: el model lineal simple captura bé la relació principal.")

# ==============================================================
# 📦 Exportar resultats
# ==============================================================

out = data.copy()
out['Predicted'] = y_pred
out['Residual'] = residuals
out.to_csv("predictions_and_residuals.csv", index=False)
print("\n📁 S'ha guardat 'predictions_and_residuals.csv' amb prediccions i residus.")

print("\n✅ Anàlisi completada correctament.")

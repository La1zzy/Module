import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

file_name = "medical_data_200_patients.csv"
df = pd.read_csv(file_name)

X_col = 'Стабілізована глюкоза (ммоль/л)'
Y_col = 'Гемоглобін (г/л)'

df[X_col] = pd.to_numeric(df[X_col], errors='coerce')
df[Y_col] = pd.to_numeric(df[Y_col], errors='coerce')
df = df.dropna(subset=[X_col, Y_col])

X = df[X_col].values
Y = df[Y_col].values

def non_linear_func(X, a, b, c):
    """Нелінійна функція: Y = a * exp(b * X) + c"""
    return a * np.exp(b * X) + c

p0 = [5.0, 0.05, 135.0]

try:
    popt, pcov = curve_fit(non_linear_func, X, Y, p0=p0, maxfev=10000)
    a_opt, b_opt, c_opt = popt

    Y_pred = non_linear_func(X, a_opt, b_opt, c_opt)

    r_squared = r2_score(Y, Y_pred)

    X_fit = np.linspace(X.min(), X.max(), 100)
    Y_fit = non_linear_func(X_fit, a_opt, b_opt, c_opt)

    plt.figure(figsize=(10, 6))
    plt.scatter(X, Y, label='Фактичні дані', alpha=0.6)
    plt.plot(X_fit, Y_fit, 'r-', label=f'Нелінійна модель: $\\hat{{Y}} = {a_opt:.2f} e^{{{b_opt:.3f} X}} + {c_opt:.2f}$')

    plt.title(f'Нелінійна регресія: Зв\'язок між Глюкозою та Гемоглобіном\\n$R^2 = {r_squared:.4f}$')
    plt.xlabel('Стабілізована глюкоза (ммоль/л)')
    plt.ylabel('Гемоглобін (г/л)')
    plt.legend()
    plt.grid(True)
    plt.savefig('nonlinear_regression_glucose_hemoglobin.png')

except RuntimeError as e:
    print(f"Помилка при обчисленні параметрів нелінійної моделі: {e}")
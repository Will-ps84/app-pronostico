import streamlit as st
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import io

warnings.filterwarnings("ignore")

# ------------------------ Funciones ------------------------

def calcular_rmse(real, pred):
    real, pred = np.array(real), np.array(pred)
    min_len = min(len(real), len(pred))
    return np.sqrt(mean_squared_error(real[-min_len:], pred[-min_len:]))

def pronosticar_modelos(serie, pasos=12):
    resultados = {}
    n_eval = min(18, len(serie))

    # Holt-Winters (ETS)
    try:
        modelo_hw = ExponentialSmoothing(
            serie, trend="add", seasonal="add", seasonal_periods=12
        ).fit()
        pred_hw = np.clip(modelo_hw.forecast(pasos), 0, None)
        rmse_hw = calcular_rmse(serie[-n_eval:], modelo_hw.fittedvalues[-n_eval:])
        resultados["Holt-Winters"] = (rmse_hw, pred_hw)
    except:
        resultados["Holt-Winters"] = (np.inf, [serie[-1]] * pasos)

    # ARIMA (1,1,1)
    try:
        modelo_arima = ARIMA(serie, order=(1, 1, 1)).fit()
        pred_arima = np.clip(modelo_arima.forecast(steps=pasos), 0, None)
        rmse_arima = calcular_rmse(
            serie[-n_eval:], modelo_arima.predict(start=len(serie)-n_eval, end=len(serie)-1)
        )
        resultados["ARIMA"] = (rmse_arima, pred_arima)
    except:
        resultados["ARIMA"] = (np.inf, [serie[-1]] * pasos)

    # Prophet
    try:
        df_prophet = pd.DataFrame({
            "ds": pd.date_range(start="2000-01-01", periods=len(serie), freq="M"),
            "y": serie
        })
        modelo_prophet = Prophet()
        modelo_prophet.fit(df_prophet)
        futuro = modelo_prophet.make_future_dataframe(periods=pasos, freq="M")
        pronostico = modelo_prophet.predict(futuro)
        pred_prophet = np.clip(pronostico.tail(pasos).set_index("ds")["yhat"], 0, None)
        fitted_prophet = pronostico.iloc[:-pasos]["yhat"]
        rmse_prophet = calcular_rmse(serie[-n_eval:], fitted_prophet[-n_eval:])
        resultados["Prophet"] = (rmse_prophet, pred_prophet)
    except:
        resultados["Prophet"] = (np.inf, [serie[-1]] * pasos)

    return resultados

def elegir_mejor_modelo(resultados):
    mejor = min(resultados, key=lambda x: resultados[x][0])
    rmse, pronostico = resultados[mejor]
    return f"{mejor} - RMSE {rmse:.2f}", pronostico

def procesar_excel(file):
    df = pd.read_excel(file)
    id_cols = list(df.columns[:2])

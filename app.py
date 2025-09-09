import gradio as gr
import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore")

# ------------------ Funciones ------------------

def calcular_rmse(real, pred):
    real, pred = np.array(real), np.array(pred)
    min_len = min(len(real), len(pred))
    return np.sqrt(mean_squared_error(real[-min_len:], pred[-min_len:]))

def pronosticar(file, pasos=12):
    df = pd.read_excel(file)
    serie = df.iloc[:, -1]  # última columna como serie

    resultados = {}
    n_eval = min(18, len(serie))

    # Holt-Winters
    try:
        modelo_hw = ExponentialSmoothing(
            serie, trend="add", seasonal="add", seasonal_periods=12
        ).fit()
        pred_hw = modelo_hw.forecast(pasos)
        rmse_hw = calcular_rmse(serie[-n_eval:], modelo_hw.fittedvalues[-n_eval:])
        resultados["Holt-Winters"] = (rmse_hw, pred_hw.tolist())
    except:
        resultados["Holt-Winters"] = (np.inf, [serie.iloc[-1]] * pasos)

    # ARIMA
    try:
        modelo_arima = ARIMA(serie, order=(1, 1, 1)).fit()
        pred_arima = modelo_arima.forecast(steps=pasos)
        rmse_arima = calcular_rmse(
            serie[-n_eval:], modelo_arima.predict(start=len(serie)-n_eval, end=len(serie)-1)
        )
        resultados["ARIMA"] = (rmse_arima, pred_arima.tolist())
    except:
        resultados["ARIMA"] = (np.inf, [serie.iloc[-1]] * pasos)

    # Prophet
    try:
        df_prophet = pd.DataFrame({
            "ds": pd.date_range(start="2000-01-01", periods=len(serie), freq="M"),
            "y": serie.values
        })
        modelo_prophet = Prophet()
        modelo_prophet.fit(df_prophet)
        futuro = modelo_prophet.make_future_dataframe(periods=pasos, freq="M")
        pronostico = modelo_prophet.predict(futuro)
        pred_prophet = pronostico.tail(pasos)["yhat"].values
        rmse_prophet = calcular_rmse(serie[-n_eval:], pronostico["yhat"].iloc[:-pasos].values[-n_eval:])
        resultados["Prophet"] = (rmse_prophet, pred_prophet.tolist())
    except:
        resultados["Prophet"] = (np.inf, [serie.iloc[-1]] * pasos)

    # Elegir mejor modelo
    mejor_modelo = min(resultados, key=lambda x: resultados[x][0])
    rmse, forecast = resultados[mejor_modelo]

    return f"Mejor modelo: {mejor_modelo} (RMSE={rmse:.2f})", forecast

# ------------------ Interfaz ------------------

demo = gr.Interface(
    fn=pronosticar,
    inputs=[
        gr.File(label="Sube tu Excel con la serie temporal"),
        gr.Slider(1, 24, value=12, step=1, label="Meses a pronosticar")
    ],
    outputs=["text", "json"],
    title="📈 Pronosticador de Series Temporales",
    description="Pronostica usando Holt-Winters, ARIMA y Prophet. Suba un Excel con una serie temporal."
)

if __name__ == "__main__":
    demo.launch()

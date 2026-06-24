import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# ============================================================
# CARGA DE ARCHIVOS DEL MODELO
# Estos archivos los genera la libreta al final.
# Deben estar en la misma carpeta que app.py
# ============================================================
RUTA_MODELO = "modelo_final.pkl"
RUTA_ENCODER = "encoder.pkl"
RUTA_SCALER = "scaler_general.pkl"
RUTA_PCA = "pca.pkl"
RUTA_INFO = "info_modelo.pkl"

modelo = None
encoder = None
scaler_general = None
pca = None
info_modelo = {}
error_carga = None

try:
    modelo = joblib.load(RUTA_MODELO)
    encoder = joblib.load(RUTA_ENCODER)
    scaler_general = joblib.load(RUTA_SCALER)
    pca = joblib.load(RUTA_PCA)
    info_modelo = joblib.load(RUTA_INFO)
except Exception as e:
    error_carga = str(e)

# ============================================================
# CONFIGURACIÓN BASE SEGÚN LA LIBRETA
# ============================================================
features_seleccionadas = info_modelo.get(
    "features_seleccionadas",
    ["Rooms", "Type", "Distance", "Landsize", "Lattitude", "Longtitude"]
)

columnas_originales_usadas = info_modelo.get(
    "columnas_originales_usadas",
    [
        "Suburb", "Rooms", "Type", "Method", "Distance", "Postcode",
        "Bedroom2", "Bathroom", "Car", "Landsize", "BuildingArea",
        "YearBuilt", "CouncilArea", "Lattitude", "Longtitude",
        "Regionname", "Propertycount"
    ]
)

columnas_categoricas = info_modelo.get(
    "columnas_categoricas",
    ["Suburb", "Type", "Method", "CouncilArea", "Regionname"]
)

mejor_escenario = info_modelo.get("mejor_escenario", "Características seleccionadas")
mejor_modelo = info_modelo.get("mejor_modelo", "XGBoost")

# Valores base para columnas no pedidas en el formulario.
# No afectan al escenario de características seleccionadas porque después solo se toman las 6 variables elegidas.
valores_base = {
    "Suburb": "Abbotsford",
    "Rooms": 3,
    "Type": "h",
    "Method": "S",
    "Distance": 8.0,
    "Postcode": 3000,
    "Bedroom2": 3,
    "Bathroom": 1,
    "Car": 1,
    "Landsize": 250,
    "BuildingArea": 120,
    "YearBuilt": 1970,
    "CouncilArea": "Yarra",
    "Lattitude": -37.80,
    "Longtitude": 145.00,
    "Regionname": "Northern Metropolitan",
    "Propertycount": 5000
}

etiquetas = {
    "Suburb": "Suburbio",
    "Rooms": "Habitaciones",
    "Type": "Tipo de vivienda",
    "Method": "Método de venta",
    "Distance": "Distancia al centro (km)",
    "Postcode": "Código postal",
    "Bedroom2": "Recámaras registradas",
    "Bathroom": "Baños",
    "Car": "Espacios para auto",
    "Landsize": "Superficie del terreno (m²)",
    "BuildingArea": "Área construida (m²)",
    "YearBuilt": "Año de construcción",
    "CouncilArea": "Zona / consejo",
    "Lattitude": "Latitud",
    "Longtitude": "Longitud",
    "Regionname": "Región",
    "Propertycount": "Cantidad de propiedades en la zona"
}

ayuda = {
    "Rooms": "Ejemplo: 3",
    "Distance": "Ejemplo: 8.2",
    "Landsize": "Ejemplo: 250",
    "Lattitude": "Ejemplo: -37.80",
    "Longtitude": "Ejemplo: 144.99",
    "BuildingArea": "Ejemplo: 120",
    "YearBuilt": "Ejemplo: 1970"
}

type_opciones = [
    ("h", "Casa / house"),
    ("u", "Departamento / unit"),
    ("t", "Townhouse")
]

# Si el mejor escenario es PCA, se piden todas las columnas originales usadas.
# Si es características seleccionadas, se piden solo las seis variables finales.
def obtener_campos_formulario():
    if mejor_escenario == "PCA":
        return columnas_originales_usadas
    return features_seleccionadas


def convertir_numero(valor, nombre_campo):
    try:
        return float(valor)
    except Exception:
        raise ValueError(f"El campo {nombre_campo} debe ser numérico.")


def preparar_entrada(formulario):
    # Crear una fila con todas las columnas que espera el scaler.
    fila = valores_base.copy()

    campos = obtener_campos_formulario()

    for col in campos:
        valor = formulario.get(col)
        if col in columnas_categoricas:
            fila[col] = valor
        else:
            fila[col] = convertir_numero(valor, etiquetas.get(col, col))

    df_entrada = pd.DataFrame([fila])
    df_entrada = df_entrada[columnas_originales_usadas]

    # Codificar variables categóricas con el mismo encoder de la libreta.
    if encoder is not None and len(columnas_categoricas) > 0:
        df_entrada[columnas_categoricas] = encoder.transform(df_entrada[columnas_categoricas])

    # Escalar con el mismo scaler de la libreta.
    datos_escalados = scaler_general.transform(df_entrada)
    df_escalado = pd.DataFrame(datos_escalados, columns=columnas_originales_usadas)

    # Elegir representación según el mejor escenario.
    if mejor_escenario == "PCA":
        entrada_modelo = pca.transform(df_escalado)
    else:
        entrada_modelo = df_escalado[features_seleccionadas]

    return entrada_modelo


@app.route("/", methods=["GET", "POST"])
def index():
    prediccion = None
    error = None
    campos = obtener_campos_formulario()
    valores = {}

    if error_carga:
        error = (
            "No se pudieron cargar los archivos del modelo. "
            "Asegúrate de copiar modelo_final.pkl, encoder.pkl, scaler_general.pkl, "
            "pca.pkl e info_modelo.pkl en la misma carpeta que app.py. "
            f"Detalle: {error_carga}"
        )

    if request.method == "POST" and not error_carga:
        try:
            # Guardar los valores que escribió el usuario
            valores = request.form.to_dict()

            entrada_modelo = preparar_entrada(request.form)
            resultado = modelo.predict(entrada_modelo)[0]
            prediccion = f"${resultado:,.2f} AUD"

        except Exception as e:
            error = f"No se pudo generar la predicción: {e}"

    return render_template(
        "index.html",
        prediccion=prediccion,
        error=error,
        campos=campos,
        etiquetas=etiquetas,
        ayuda=ayuda,
        valores=valores,
        columnas_categoricas=columnas_categoricas,
        type_opciones=type_opciones,
        mejor_modelo=mejor_modelo,
        mejor_escenario=mejor_escenario
    )

if __name__ == "__main__":
    puerto = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=puerto, debug=False)

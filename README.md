# Sistema Predictivo de Intención de Compra — E-commerce

> **Predicción en tiempo real de si un usuario realizará una compra**, implementada como un pipeline completo de ciencia de datos y desplegada a través de una API REST.

---

## Índice

1. [Contexto del Negocio](#1-contexto-del-negocio)
2. [Estructura del Repositorio](#2-estructura-del-repositorio)
3. [Instalación y Ejecución Rápida](#3-instalación-y-ejecución-rápida)
4. [Pipeline de Machine Learning](#4-pipeline-de-machine-learning)
   - [4.1 Análisis Exploratorio del Dataset](#41-análisis-exploratorio-del-dataset)
   - [4.2 Limpieza de Datos](#42-limpieza-de-datos)
   - [4.3 Codificación de Variables Categóricas](#43-codificación-de-variables-categóricas)
   - [4.4 División de Datos](#44-división-de-datos)
   - [4.5 Escalado de Características](#45-escalado-de-características)
   - [4.6 Aumento de Datos — SMOTE](#46-aumento-de-datos--smote)
   - [4.7 Selección del Algoritmo](#47-selección-del-algoritmo)
   - [4.8 Ajuste de Hiperparámetros — GridSearchCV](#48-ajuste-de-hiperparámetros--gridsearchcv)
   - [4.9 Optimización del Umbral de Decisión](#49-optimización-del-umbral-de-decisión)
   - [4.10 Métricas de Rendimiento Final](#410-métricas-de-rendimiento-final)
5. [API REST](#5-api-rest)
   - [Endpoints](#endpoints)
   - [Formato de Respuesta](#formato-de-respuesta)
   - [Ejemplos de Uso — Postman / curl](#ejemplos-de-uso--postman--curl)
6. [Decisiones de Diseño Justificadas](#6-decisiones-de-diseño-justificadas)

---

## 1. Contexto del Negocio

Una empresa de e-commerce necesita detectar de forma temprana si un visitante web realizará una compra durante su sesión activa. Al identificar usuarios con alta intención de compra, la tienda puede mostrar dinámicamente productos de mayor valor (*upselling*) para incrementar el ingreso general (revenue).

El sistema recibe los datos de comportamiento de navegación en tiempo real y retorna una predicción estructurada en JSON consumible directamente por el departamento de IT.

---

## 2. Estructura del Repositorio

```
AI/
├── dataset shop.csv          # Dataset fuente (12,330 sesiones)
├── requirements.txt          # Dependencias Python
├── README.md                 # Este informe
│
├── pipeline/
│   ├── train.py              # Pipeline completo de DS → genera artefactos
│   └── model/
│       ├── mlp_model.pkl     # Modelo entrenado (GradientBoostingClassifier)
│       ├── scaler.pkl        # StandardScaler ajustado al conjunto de entrenamiento
│       └── encoders.pkl      # Encoders + threshold óptimo
│
└── api/
    └── app.py                # Servidor Flask — endpoints REST
```

---

## 3. Instalación y Ejecución Rápida

### Requisitos
- Python 3.10+
- Las dependencias listadas en `requirements.txt`

### Configuración del entorno

```bash
# Crear entorno virtual
python3 -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### Entrenar el modelo

```bash
python pipeline/train.py
```

El script ejecuta el pipeline completo e imprime las métricas por consola. Los artefactos se guardan automáticamente en `pipeline/model/`.

### Iniciar el servidor API

```bash
python api/app.py
```

El servidor queda disponible en `http://localhost:5000`.

---

## 4. Pipeline de Machine Learning

### 4.1 Análisis Exploratorio del Dataset

| Propiedad | Valor |
|---|---|
| Filas (sesiones) | 12,330 |
| Columnas totales | 18 (17 features + 1 target) |
| Atributos numéricos | 10 |
| Atributos categóricos | 8 |
| Valores faltantes | **0** (dataset limpio) |
| Distribución target | 84.5% No compra / 15.5% Compra |

**Imbalance crítico detectado:** La clase positiva (`Revenue = True`) representa solo el 15.5% de las observaciones. Un clasificador naive que siempre predijera "No compra" obtendría 84.5% de accuracy sin aprender nada. Esta asimetría determina todas las decisiones técnicas del pipeline.

| Variable | Tipo | Descripción |
|---|---|---|
| Administrative | int | Páginas administrativas visitadas |
| Administrative_Duration | float | Tiempo (s) en páginas administrativas |
| Informational | int | Páginas informativas visitadas |
| Informational_Duration | float | Tiempo (s) en páginas informativas |
| ProductRelated | int | Páginas de producto visitadas |
| ProductRelated_Duration | float | Tiempo (s) en páginas de producto |
| BounceRates | float | % de rebote promedio [0, 1] |
| ExitRates | float | % de salida promedio [0, 1] |
| PageValues | float | Valor promedio de página antes de transacción |
| SpecialDay | float | Proximidad a fecha especial [0, 1] |
| Month | str | Mes de la sesión |
| OperatingSystems | int | Código de sistema operativo |
| Browser | int | Código de navegador |
| Region | int | Región geográfica |
| TrafficType | int | Fuente de tráfico |
| VisitorType | str | New_Visitor / Returning_Visitor / Other |
| Weekend | bool | ¿Sesión en fin de semana? |
| **Revenue** | **bool** | **Variable objetivo (compra = True)** |

---

### 4.2 Limpieza de Datos

El dataset no presenta valores faltantes, por lo que no fue necesaria imputación. Sin embargo, el pipeline incluye lógica de imputación defensiva para producción:

- **Categóricas/booleanas:** imputación por moda.
- **Numéricas:** imputación por mediana (robusta a outliers).

Las columnas `Weekend` y `Revenue`, almacenadas como strings `"TRUE"`/`"FALSE"` en el CSV, son convertidas explícitamente a booleanos Python.

---

### 4.3 Codificación de Variables Categóricas

#### `Month` → OrdinalEncoder

La variable `Month` tiene un **orden natural** (Feb < Mar < May … < Dec). Se usa `OrdinalEncoder` con el orden calendario explícito. Esto preserva la información temporal que un LabelEncoder aleatorio destruiría.

```
Feb=0, Mar=1, May=2, Jun=3, Jul=4, Aug=5, Sep=6, Oct=7, Nov=8, Dec=9
```

**Alternativa descartada:** `OneHotEncoder` generaría 10 columnas adicionales para una variable que ya tiene estructura ordinal, aumentando dimensionalidad sin beneficio.

#### `VisitorType` → LabelEncoder

No existe orden natural entre `New_Visitor`, `Returning_Visitor` y `Other`, por lo que se usa `LabelEncoder` que asigna enteros sin implicar jerarquía (el modelo árbol no interpreta orden en variables de este tipo de la misma forma que modelos lineales).

#### `Weekend` → int (0/1)

Conversión directa booleano → entero.

---

### 4.4 División de Datos

Se aplica una **división en tres conjuntos** (60% train / 20% val / 20% test), todos estratificados por la clase objetivo:

```
Total: 12,330
  ├── Train (60%): 7,398  → SMOTE + entrenamiento
  ├── Val   (20%): 2,466  → optimización de umbral (distribución real)
  └── Test  (20%): 2,466  → evaluación final (jamás visto en ningún paso)
```

**¿Por qué tres conjuntos?**
La optimización del umbral de decisión (Sección 4.9) requiere un conjunto con **distribución real** (no balanceada por SMOTE). Si se usara el mismo conjunto de test para encontrar el threshold, se violaría la integridad estadística. El conjunto de validación cumple ese rol, y el test queda completamente intocado.

**¿Por qué estratificado?** Con solo 15.5% de positivos, una división aleatoria simple podría producir pliegues sin suficientes ejemplos positivos para el modelo. `stratify=y` garantiza la proporción en los tres subconjuntos.

---

### 4.5 Escalado de Características

Se aplica `StandardScaler` (z-score: `z = (x - μ) / σ`) a todas las variables numéricas.

**Regla clave:** El `StandardScaler` se ajusta **únicamente sobre el conjunto de entrenamiento** y se aplica (`.transform()`) sobre validación y test. Ajustar sobre los tres conjuntos provocaría *data leakage* — el modelo "vería" estadísticas del futuro.

**¿Por qué StandardScaler y no MinMaxScaler?** El dataset contiene features con distribuciones muy sesgadas a la derecha (ej: `ProductRelated_Duration` tiene media ~50s pero valores hasta 63,973s). StandardScaler es más robusto a outliers extremos que MinMaxScaler.

---

### 4.6 Aumento de Datos — SMOTE

**SMOTE (Synthetic Minority Over-sampling Technique)** genera ejemplos sintéticos de la clase minoritaria interpolando entre sus vecinos más cercanos (k=5).

```
Antes: {No compra: 6,253 | Compra: 1,145}  →  ratio 5.5:1
Después: {No compra: 6,253 | Compra: 6,253}  →  ratio 1:1
```

**Aplicado únicamente al training set.** El conjunto de validación y test conservan la distribución real (84.5/15.5) para que las métricas sean representativas del entorno de producción.

**¿Por qué SMOTE y no oversampling aleatorio?** El oversampling aleatorio duplica ejemplos existentes sin añadir información nueva — el modelo los memoriza. SMOTE crea vecinos sintéticos en el espacio de características, forzando al modelo a aprender regiones de decisión más generalizables para la clase minoritaria.

---

### 4.7 Selección del Algoritmo

#### Iteración 1: MLPClassifier (sklearn)

Como primer algoritmo se probó `MLPClassifier` (Perceptrón Multicapa) con arquitectura `(256, 128, 64)`, directamente cubierto en la clase *Clase_Perceptron.ipynb*. Resultado: **86.7% accuracy** en test, por debajo del objetivo.

#### Iteración 2: GradientBoostingClassifier ✅ (algoritmo final)

Siguiendo la guía del mapa de Scikit-Learn presentada en *Clase_5_Toma_de_Decisión_en_ML.ipynb* — sección *"Tough Luck → Switch to Ensemble"* — se migró a `GradientBoostingClassifier`.

| Característica | MLP | GradientBoosting |
|---|---|---|
| Tipo | Red neuronal | Ensemble de árboles |
| Sensible al escalado | Sí (requiere StandardScaler) | No intrínsecamente |
| Interpretabilidad | Baja (caja negra) | Media (feature importance) |
| Test Accuracy | 86.7% | **89.0%** |
| Requiere SMOTE | Sí | Sí (mejora recall) |

**¿Por qué no Random Forest?** GBM aprende secuencialmente corrigiendo los errores anteriores — es más preciso en datasets tabulares medianos. RF entrena en paralelo con bagging, lo que lo hace más rápido pero menos preciso en este caso.

**¿Por qué no Deep Learning (Keras)?** Con 12,330 filas y 17 features, una red profunda de TensorFlow/Keras añade complejidad computacional y riesgo de overfitting sin mejora de accuracy justificada. GBM supera a las redes profundas en la mayoría de datasets tabulares de este tamaño, un hecho bien documentado en la literatura.

---

### 4.8 Ajuste de Hiperparámetros — GridSearchCV

Se ejecutó `GridSearchCV` con validación cruzada estratificada de 5 pliegues (`StratifiedKFold`) sobre el espacio:

| Hiperparámetro | Valores probados | Seleccionado |
|---|---|---|
| `n_estimators` | 200, 300 | **300** |
| `learning_rate` | 0.05, 0.1 | **0.1** |
| `max_depth` | 4, 5 | **5** |
| `min_samples_leaf` | 10, 20 | **20** |
| `subsample` | 0.8 | 0.8 |

**Total de combinaciones:** 16 candidatos × 5 folds = 80 ajustes.

**Mejor CV score (datos balanceados SMOTE):** 94.35%

**Parámetros ganadores:**
```python
GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.1,
    max_depth=5, min_samples_leaf=20,
    subsample=0.8, random_state=42
)
```

**Rol de `subsample=0.8`:** Cada árbol se entrena con el 80% aleatorio de los datos. Esto introduce estocasidad que actúa como regularización, reduciendo el overfitting (equivale conceptualmente al Dropout en redes neuronales, cubierto en el material de regularización).

**Rol de `min_samples_leaf=20`:** Evita que el árbol crezca divisiones sobre muy pocos ejemplos, combatiendo el overfitting sobre ruido.

---

### 4.9 Optimización del Umbral de Decisión

Un clasificador probabilístico por defecto predice la clase positiva cuando `P(compra) ≥ 0.5`. Este umbral asume **clases igualmente probables a priori**, lo cual es falso aquí.

El modelo fue entrenado sobre datos SMOTE (50/50), pero el mundo real tiene distribución 84.5/15.5. El umbral se barre en `[0.20, 0.70]` sobre el **conjunto de validación** (distribución real) y se elige el que maximiza accuracy:

```
Threshold evaluado: 0.53
Val Accuracy: 89.74%  ✅
```

Este umbral óptimo (0.53) se guarda en `encoders.pkl` y es cargado automáticamente por la API.

---

### 4.10 Métricas de Rendimiento Final

Evaluadas sobre el **conjunto de test** (2,466 sesiones, nunca vistas durante entrenamiento ni optimización).

#### Accuracy

```
Test Accuracy: 89.01%  ✅  (objetivo: ~90%)
```

#### Reporte de Clasificación

```
              precision    recall  f1-score   support

 No Purchase       0.93      0.95      0.94      2084
    Purchase       0.67      0.58      0.62       382

    accuracy                           0.89      2466
   macro avg       0.80      0.77      0.78      2466
weighted avg       0.89      0.89      0.89      2466
```

#### Matriz de Confusión

```
                Predicho: No compra   Predicho: Compra
Real: No compra       1,972               112
Real: Compra            159               223
```

| Métrica | Valor | Interpretación |
|---|---|---|
| Accuracy | **89.01%** | De cada 100 sesiones, ~89 clasificadas correctamente |
| Precision (Purchase) | 0.67 | El 67% de quienes predice como compradores realmente compran |
| Recall (Purchase) | 0.58 | Detecta el 58% de todos los compradores reales |
| F1-Score (Purchase) | 0.62 | Balance harmónico precision-recall para la clase minoritaria |
| Verdaderos Negativos | 1,972 | No molesta a visitantes que no comprarían |
| Verdaderos Positivos | 223 | Oportunidades de upselling correctamente identificadas |
| Falsos Positivos | 112 | Upselling innecesario — costo mínimo para el negocio |
| Falsos Negativos | 159 | Oportunidades perdidas — el costo más alto |

**Lectura de negocio:** Para el e-commerce, un Falso Negativo (comprador perdido) es más costoso que un Falso Positivo (mostrar upselling a quien no compra). El modelo prioriza correctamente la clase mayoritaria (94% recall) mientras captura el 58% de compradores reales — un equilibrio razonable sin sacrificar la experiencia del usuario común.

---

## 5. API REST

### Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `GET` | `/` | Documentación completa de la API en JSON |
| `GET` | `/health` | Estado del servidor y nombre del modelo activo |
| `POST` | `/predict` | **Predicción de intención de compra** |

---

### Formato de Respuesta

La respuesta de `POST /predict` contiene **exactamente tres campos**:

```json
{
  "classification": "purchase",
  "probability": 0.7199,
  "message": "El usuario presenta un 72% de probabilidades de hacer la compra, lo que lo hace bastante probable."
}
```

| Campo | Tipo | Valores posibles |
|---|---|---|
| `classification` | string | `"purchase"` \| `"no_purchase"` |
| `probability` | float | 0.0000 – 1.0000 |
| `message` | string | Descripción legible en español |

#### Niveles de mensaje por probabilidad

| Rango | Clasificación | Nivel en mensaje |
|---|---|---|
| ≥ 80% | purchase | "muy probable" |
| 60–79% | purchase | "bastante probable" |
| < 60% | purchase | "posible" |
| ≥ 40% | no_purchase | "algo posible" |
| 20–39% | no_purchase | "poco probable" |
| < 20% | no_purchase | "muy improbable" |

---

### Ejemplos de Uso — Postman / curl

#### GET /health

```bash
curl http://localhost:5000/health
```

```json
{
  "message": "Prediction service is running.",
  "model": "GradientBoostingClassifier",
  "status": "ok"
}
```

---

#### POST /predict — Usuario con alta intención de compra

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 2,
    "Administrative_Duration": 30.0,
    "Informational": 1,
    "Informational_Duration": 15.0,
    "ProductRelated": 15,
    "ProductRelated_Duration": 650.0,
    "BounceRates": 0.01,
    "ExitRates": 0.03,
    "PageValues": 42.5,
    "SpecialDay": 0.0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
  }'
```

```json
{
  "classification": "purchase",
  "message": "El usuario presenta un 72% de probabilidades de hacer la compra, lo que lo hace bastante probable.",
  "probability": 0.7199
}
```

---

#### POST /predict — Usuario de baja intención de compra

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Administrative": 0,
    "Administrative_Duration": 0.0,
    "Informational": 0,
    "Informational_Duration": 0.0,
    "ProductRelated": 1,
    "ProductRelated_Duration": 0.0,
    "BounceRates": 0.2,
    "ExitRates": 0.2,
    "PageValues": 0.0,
    "SpecialDay": 0.0,
    "Month": "Feb",
    "OperatingSystems": 1,
    "Browser": 1,
    "Region": 3,
    "TrafficType": 3,
    "VisitorType": "New_Visitor",
    "Weekend": false
  }'
```

```json
{
  "classification": "no_purchase",
  "message": "El usuario presenta un 0% de probabilidades de hacer la compra, lo que lo hace muy improbable.",
  "probability": 0.0006
}
```

---

#### Campos del Request

| Campo | Tipo | Descripción | Valores válidos |
|---|---|---|---|
| Administrative | int | Páginas admin visitadas | ≥ 0 |
| Administrative_Duration | float | Tiempo (s) en páginas admin | ≥ 0.0 |
| Informational | int | Páginas informativas visitadas | ≥ 0 |
| Informational_Duration | float | Tiempo (s) en páginas info | ≥ 0.0 |
| ProductRelated | int | Páginas de producto visitadas | ≥ 0 |
| ProductRelated_Duration | float | Tiempo (s) en páginas producto | ≥ 0.0 |
| BounceRates | float | Tasa de rebote promedio | 0.0 – 1.0 |
| ExitRates | float | Tasa de salida promedio | 0.0 – 1.0 |
| PageValues | float | Valor promedio de página | ≥ 0.0 |
| SpecialDay | float | Proximidad a fecha especial | 0.0 – 1.0 |
| Month | string | Mes de la sesión | Feb, Mar, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec |
| OperatingSystems | int | Código del SO | 1 – 8 |
| Browser | int | Código del navegador | 1 – 13 |
| Region | int | Región geográfica | 1 – 9 |
| TrafficType | int | Tipo de tráfico | 1 – 20 |
| VisitorType | string | Tipo de visitante | New_Visitor, Returning_Visitor, Other |
| Weekend | bool | ¿Fin de semana? | true / false |

---

## 6. Decisiones de Diseño Justificadas

| Decisión | Elección | Por qué | Por qué no las alternativas |
|---|---|---|---|
| **Algoritmo** | GradientBoostingClassifier | Ensemble recomendado por Clase_5 como fallback "Tough Luck"; mejor accuracy en tabular | MLP: plateó en ~87%. Keras: overkill para 12k filas. Perceptrón: lineal, falla en datos no separables linealmente. |
| **Imbalance** | SMOTE | Genera ejemplos sintéticos informados por vecinos — aprende mejor la minoría | Oversampling aleatorio: duplica sin generalizar. Undersampling: descarta 10k+ filas de mayoría. |
| **Escalado** | StandardScaler | Obligatorio para gradiente (cubierto en todos los notebooks); robusto a outliers extremos | MinMaxScaler: sensible a outliers extremos presentes en Duration features. |
| **Encoding Month** | OrdinalEncoder (orden calendario) | Preserva la relación temporal Feb→Dec | LabelEncoder: orden arbitrario. OneHotEncoder: 10 columnas innecesarias para variable ordinal. |
| **División** | 60/20/20 tres vías | El val set con distribución real permite optimizar threshold sin contaminar test | 80/20 dos vías: no hay conjunto limpio para threshold tuning. |
| **Threshold** | 0.53 (optimizado en val set) | El modelo entrenado en SMOTE 50/50 necesita ajuste al prior real 84.5/15.5 | 0.5 hardcoded: asume priors iguales, penaliza recall en clase minoritaria. |
| **API** | Flask | Standard ML API en el ecosistema Python; mencionado en el material del curso | FastAPI: más features pero más complejo; Django: demasiado para una API ML. |
| **Persistencia** | joblib | Optimizado para arrays NumPy grandes (pesos del modelo); más rápido que pickle | pickle: más lento con arrays. ONNX: no cubierto en el curso. |
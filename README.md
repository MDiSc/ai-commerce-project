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
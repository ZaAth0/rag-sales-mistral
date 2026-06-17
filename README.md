# rag-sales-mistral
Sistema RAG con Mistral-7B-Instruct-v0.3 para análisis de ventas en lenguaje natural. Chat interactivo sobre dataset CSV usando búsqueda semántica y API de Mistral. Ejecutable en Google Colab.

## Respuestas proveídas
### 1
Tú: Ordenes que hay 
Según el contexto proporcionado, las órdenes registradas son:

1. **Orden #0.5** (Classic Cars):
   - Monto: $0.37
   - Estado: Shipped
   - País: Australia
   - Fecha: 2004-07-19
   - Tamaño: Medium
   - Cliente: Souveniers And Things Co.

2. **Orden #0.5** (Classic Cars):
   - Monto: $0.47
   - Estado: Shipped
   - País: Australia
   - Fecha: 2004-07-19
   - Tamaño: Medium
   - Cliente: Souveniers And Things Co.

3. **Orden #0.5** (Classic Cars):
   - Monto: $0.40
   - Estado: Shipped
   - País: Australia
   - Fecha: 2004-07-19
   - Tamaño: Medium
   - Cliente: Souveniers And Things Co.

4. **Orden #0.5** (Vintage Cars):
   - Monto: $0.21
   - Estado: Shipped
   - País: Australia
   - Fecha: 2004-07-19
   - Tamaño: Small
   - Cliente: Souveniers And Things Co.
### 2
 Tú: Dame todos los clientes
 Según el contexto proporcionado, el único cliente registrado es:

- **Souveniers And Things Co.**
  - Venta #0.5: $0.07 (Tamaño: Small)
  - Venta #0.5: $0.38 (Tamaño: Medium)
  - Venta #0.5: $0.38 (Tamaño: Medium)
  - Venta #0.5: $0.23 (Tamaño: Small)
### 3

Tú: quiero el monto mas grande
El monto más grande registrado en el contexto es **$0.47** (VENTA #0.5).

**Nota:** Todas las órdenes tienen el mismo número (#0.5), lo que sugiere que podrían ser transacciones parciales o registros duplicados del mismo evento. No hay suficiente información para determinar si son órdenes independientes o parte de un mismo proceso.


# AGENTES - PROYECTO 08062026
# Sistema de Agentes de IA para Análisis Predictivo de Reviews de Amazon

## Descripción General

Este proyecto implementa un sistema multiagente basado en Inteligencia Artificial para el análisis predictivo de reseñas de productos alimenticios de Amazon. El sistema integra técnicas de Procesamiento de Lenguaje Natural (NLP), análisis de sentimiento y Machine Learning para predecir la puntuación otorgada por los usuarios a partir del contenido de sus reseñas.

La solución está compuesta por tres agentes especializados que trabajan de forma coordinada para realizar la carga, transformación, análisis, entrenamiento y comunicación de resultados sobre el conjunto de datos Amazon Fine Food Reviews.

---

## Objetivos

- Procesar grandes volúmenes de reseñas de productos alimenticios.
- Extraer características relevantes a partir de texto y metadatos.
- Aplicar análisis de sentimiento mediante VADER.
- Entrenar y comparar múltiples modelos de Machine Learning.
- Evaluar el desempeño predictivo mediante métricas especializadas.
- Generar reportes y visualizaciones para facilitar la interpretación de resultados.

---

## Dataset Utilizado

| Característica | Valor |
|---------------|--------|
| Nombre        | Amazon Fine Food Reviews |
| Fuente        | Kaggle |
| Registros     | 568,454 |
| Var objetivo  | Score (1-5 estrellas) |

### Variables Originales

- Id
- ProductId
- UserId
- ProfileName
- HelpfulnessNumerator
- HelpfulnessDenominator
- Score
- Time
- Summary
- Text

### Distribución de Scores

| Score | Cantidad | Porcentaje |
|---------|-----------|------------|
   | 1    | 52,268 | 9.2% |
   | 2    | 29,769 | 5.2% |
   | 3    | 42,640 | 7.5% |
   | 4    | 80,655 | 14.2% |
   | 5    | 363,122| 63.9% |

---

## Arquitectura del Sistema

El sistema está compuesto por tres agentes especializados.

### Agente 1: Normalizador

Responsable de la preparación y transformación de los datos.

#### Funcionalidades

- Carga del dataset.
- Limpieza de registros.
- Extracción de características textuales.
- Transformación de variables temporales.
- Cálculo de métricas de utilidad.
- Análisis de sentimiento mediante VADER.
- Escalado de variables numéricas.

#### Variables Generadas

**Texto**

- Summary_length
- Summary_word_count
- Text_length
- Text_word_count

**Fecha**

- Año
- Mes
- DiaSemana

**Utilidad**

- HelpfulnessRatio

**Sentimiento**

- Summary_sentiment
- Text_sentiment

---

### Agente 2: Entrenador

Responsable del entrenamiento y evaluación de modelos predictivos.

#### Funcionalidades

- División de datos en entrenamiento, validación y prueba.
- Entrenamiento de modelos.
- Comparación de resultados.
- Evaluación mediante métricas avanzadas.
- Generación de insights analíticos.

#### División de Datos

| Conjunto | Porcentaje |
|-----------|------------|
| Entrenamiento | 60% |
| Validación | 20% |
| Prueba | 20% |

#### Modelos Evaluados

- Regresión Lineal
- Random Forest Regressor
- Gradient Boosting Regressor

---

### Agente 3: Comunicador

Responsable de la presentación de resultados.

#### Funcionalidades

- Generación de dashboard analítico.
- Exportación de métricas.
- Generación de reportes TXT.
- Generación de reportes PDF.
- Exportación de datasets procesados.

---

## Resultados

### Comparación de Modelos

| Modelo | R² Test | R² Validación | RMSE | MAE | Exactitud (±1) |
|----------|----------|--------------|---------|---------|--------------|
| Regresión Lineal | 0.3325 | - | 1.0661 | 0.7818 | 66.3% |
| Random Forest | 0.4460 | 0.4493 | 0.9712 | 0.6803 | 79.3% |
| Gradient Boosting | 0.4215 | - | 0.9925 | 0.7162 | 78.3% |

### Mejor Modelo: Random Forest

| Métrica | Valor |
|----------|--------|
| R² Test | 0.4460 |
| R² Validación | 0.4493 |
| RMSE | 0.9712 |
| MAE | 0.6803 |
| Exactitud Exacta | 51.3% |
| Exactitud (±1 punto) | 79.3% |

---

## Importancia de Características

| Característica | Importancia |
|----------------|-------------|
| Summary_sentiment | 66.5% |
| HelpfulnessRatio | 13.0% |
| HelpfulnessDenominator | 8.8% |
| Time | 5.1% |
| Otras variables | 6.6% |

---

## Hallazgos Principales

- El modelo Random Forest obtuvo el mejor desempeño general.
- El sentimiento del resumen es la característica más influyente.
- El sistema alcanza una capacidad predictiva moderada para un problema subjetivo.
- El error promedio es inferior a un punto sobre una escala de cinco niveles.
- El 79.3% de las predicciones presentan un error máximo de ±1 punto.
- Existe un sesgo importante hacia valoraciones positivas dentro del dataset.

---

## Tecnologías Utilizadas

- Python 3.14.3
- Pandas
- NumPy
- Scikit-learn
- NLTK
- VADER Sentiment
- Matplotlib
- Seaborn
- FPDF2
- KaggleHub

---

## Ejecución

El proyecto fue desarrollado para ejecutarse en Google Colab.

Orden recomendado de ejecución:

1. Instalación de dependencias.
2. Importación de librerías.
3. Definición del Agente Normalizador.
4. Ejecución del Agente Normalizador.
5. Diagnóstico de columnas.
6. Diagnóstico avanzado.
7. Definición del Agente Entrenador.
8. Ejecución del Agente Entrenador.
9. Definición del Agente Comunicador.
10. Generación de resultados finales.

---

---

## Archivos Generados

| Archivo | Descripción |
|----------|-------------|
| amazon_reviews_limpio.csv | Dataset procesado |
| reporte_analisis.txt | Reporte textual |
| reporte_analisis.pdf | Reporte profesional |
| metricas_modelo.csv | Métricas de evaluación |
| feature_importance.csv | Importancia de variables |
| dashboard.png | Dashboard analítico |

---

## Dashboard

El dashboard incluye:

1. Distribución de Scores.
2. Importancia de características.
3. Relación entre sentimiento y puntuación.
4. Matriz de confusión normalizada.



## Limitaciones

- Desbalance de clases hacia puntuaciones positivas.
- Análisis de sentimiento limitado por restricciones de rendimiento.
- No se consideran características propias de los productos.
- Dependencia exclusiva de información textual y metadatos disponibles.

---

## Enlace del COLAB
https://colab.research.google.com/drive/1Rx_3QrA-081IzilzK6IBAP_q5Lfll7qw?usp=sharing





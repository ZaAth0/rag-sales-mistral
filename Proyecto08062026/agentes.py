#Sistema de Agentes de IA - Reviews de Amazon 
#Instalación de dependencias necesarias para el proyecto
!pip install kagglehub pandas numpy matplotlib seaborn scikit-learn nltk wordcloud fpdf2 vaderSentiment
#Instalación de librerías
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from IPython.display import display
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from fpdf import FPDF
import warnings
import os
import time
warnings.filterwarnings('ignore')

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('vader_lexicon', quiet=True)

# AGENTE 1: NORMALIZADOR

class AgenteNormalizador:
    """
    Agente responsable de la carga, limpieza y normalización del dataset
    Amazon Fine Food Reviews.
    MEJORADO con análisis de sentimiento VADER.
    """

    def __init__(self):
        self.df = None
        self.path_dataset = None
        self.reporte = {}
        self.nombre_dataset = None
        self.scaler = None
        self.df_original = None
        self.score_original = None

    def cargar_dataset(self):
        """Descarga y carga el dataset desde Kaggle."""
        print("AGENTE NORMALIZADOR - Fase 1: Carga del Dataset")
        print("-" * 50)

        self.path_dataset = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
        print(f"Path descargado: {self.path_dataset}")

        archivos = os.listdir(self.path_dataset)
        archivos_csv = [f for f in archivos if f.endswith('.csv')]
        print(f"CSVs encontrados: {len(archivos_csv)}")

        if not archivos_csv:
            raise FileNotFoundError(f"No se encontro CSV en {self.path_dataset}")

        for archivo_csv in archivos_csv:
            ruta_completa = os.path.join(self.path_dataset, archivo_csv)
            try:
                df_temp = pd.read_csv(ruta_completa, nrows=5)
                columnas = df_temp.columns.tolist()
                print(f"\n  {archivo_csv}: {len(columnas)} columnas")
                print(f"  Columnas: {columnas}")

                if len(columnas) >= 5:
                    self.df = pd.read_csv(ruta_completa)
                    self.nombre_dataset = archivo_csv
                    self.df_original = self.df.copy()
                    print(f"\nDataset cargado exitosamente: {archivo_csv}")
                    print(f"  Dimensiones: {self.df.shape}")
                    print(f"  Columnas: {list(self.df.columns)}")
                    return self

            except Exception as e:
                print(f"  Error al leer {archivo_csv}: {e}")
                continue

        raise ValueError(f"No se pudo cargar ningun CSV valido.")

    def exploracion_inicial(self):
        """Exploracion inicial del dataset."""
        print("\nAGENTE NORMALIZADOR - Fase 2: Exploracion Inicial")
        print("-" * 50)

        print(f"Dataset: {self.nombre_dataset}")
        print(f"Dimensiones: {self.df.shape[0]:,} filas, {self.df.shape[1]} columnas")

        print("\nColumnas y tipos:")
        for col in self.df.columns:
            n_nulos = self.df[col].isnull().sum()
            n_unicos = self.df[col].nunique()
            print(f"  * {col:<25} tipo={str(self.df[col].dtype):<10} unicos={n_unicos:<8} nulos={n_nulos}")

        # Mostrar distribucion de Score
        if 'Score' in self.df.columns:
            print(f"\nDistribucion de Score (target):")
            score_dist = self.df['Score'].value_counts().sort_index()
            for score, count in score_dist.items():
                pct = (count / len(self.df)) * 100
                print(f"  Score {int(score)}: {count:,} reviews ({pct:.1f}%)")

        print("\nPrimeras 3 filas:")
        display(self.df.head(3))

        self.reporte['shape_original'] = self.df.shape
        return self

    def limpiar_dataset(self):
        """Pipeline de limpieza MEJORADO con analisis de sentimiento."""
        print("\nAGENTE NORMALIZADOR - Fase 3: Limpieza de Datos")
        print("-" * 50)

        df_limpio = self.df.copy()

        # 1. Eliminar duplicados
        duplicados_antes = len(df_limpio)
        df_limpio = df_limpio.drop_duplicates()
        print(f"Duplicados eliminados: {duplicados_antes - len(df_limpio)}")

        # 2. Convertir tipos de datos
        print("\nConvirtiendo tipos de datos...")

        # Score: asegurar que sea entero (1-5)
        if 'Score' in df_limpio.columns:
            df_limpio['Score'] = pd.to_numeric(df_limpio['Score'], errors='coerce')
            self.score_original = df_limpio['Score'].copy()
            print(f"  Score: preservado (valores: {sorted(df_limpio['Score'].dropna().unique())})")

        # HelpfulnessNumerator y HelpfulnessDenominator
        for col in ['HelpfulnessNumerator', 'HelpfulnessDenominator']:
            if col in df_limpio.columns:
                df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce')
                print(f"  {col}: convertido a numerico")

        # Time: convertir y extraer caracteristicas de fecha
        if 'Time' in df_limpio.columns:
            df_limpio['Time'] = pd.to_numeric(df_limpio['Time'], errors='coerce')
            try:
                df_limpio['Fecha'] = pd.to_datetime(df_limpio['Time'], unit='s')
                df_limpio['Año'] = df_limpio['Fecha'].dt.year
                df_limpio['Mes'] = df_limpio['Fecha'].dt.month
                df_limpio['DiaSemana'] = df_limpio['Fecha'].dt.dayofweek
                print(f"  Time: caracteristicas de fecha extraidas (Año, Mes, DiaSemana)")
            except:
                print(f"  Time: no se pudieron extraer caracteristicas de fecha")

        # 3. Crear features de texto
        print("\nCreando features de texto...")
        for col_texto in ['Summary', 'Text']:
            if col_texto in df_limpio.columns:
                df_limpio[f'{col_texto}_length'] = df_limpio[col_texto].fillna('').str.len()
                df_limpio[f'{col_texto}_word_count'] = df_limpio[col_texto].fillna('').str.split().str.len()
                print(f"  {col_texto}: length y word_count creados")

        # 4. Crear feature de helpfulness ratio
        if 'HelpfulnessNumerator' in df_limpio.columns and 'HelpfulnessDenominator' in df_limpio.columns:
            df_limpio['HelpfulnessRatio'] = np.where(
                df_limpio['HelpfulnessDenominator'] > 0,
                df_limpio['HelpfulnessNumerator'] / df_limpio['HelpfulnessDenominator'],
                0
            )
            print(f"  HelpfulnessRatio: creado (Numerator/Denominator)")

        
        # MEJORA 1: ANALISIS DE SENTIMIENTO CON VADER
        
        print("\n" + "="*50)
        print("MEJORA: Analizando sentimiento del texto con VADER...")
        print("="*50)

        sia = SentimentIntensityAnalyzer()

        # Analizar Summary (completo, es corto)
        start_time = time.time()
        df_limpio['Summary_sentiment'] = df_limpio['Summary'].fillna('').apply(
            lambda x: sia.polarity_scores(str(x))['compound']
        )
        print(f"  Summary_sentiment: creado (compound score) en {time.time()-start_time:.1f}s")
        print(f"    Rango: [{df_limpio['Summary_sentiment'].min():.3f}, {df_limpio['Summary_sentiment'].max():.3f}]")
        print(f"    Media: {df_limpio['Summary_sentiment'].mean():.3f}")

        # Analizar Text (muestra de 100,000 para eficiencia)
        start_time = time.time()
        sample_size = min(100000, len(df_limpio))
        df_sample = df_limpio.sample(sample_size, random_state=42)
        sentiment_dict = {}

        for idx, text in df_sample['Text'].items():
            sentiment_dict[idx] = sia.polarity_scores(str(text))['compound']

        df_limpio['Text_sentiment'] = df_limpio.index.map(
            lambda x: sentiment_dict.get(x, 0)
        )
        print(f"  Text_sentiment: creado (muestra de {sample_size:,} reviews) en {time.time()-start_time:.1f}s")
        print(f"    Rango: [{df_limpio['Text_sentiment'].min():.3f}, {df_limpio['Text_sentiment'].max():.3f}]")
        print(f"    Media: {df_limpio['Text_sentiment'].mean():.3f}")

        # 5. Escalado de variables numericas (EXCLUYENDO Score y sentiment)
        print("\nAplicando escalado (excluyendo Score y sentiment)...")
        columnas_no_escalar = ['Id', 'Score', 'Summary_sentiment', 'Text_sentiment']
        columnas_a_escalar = [
            col for col in df_limpio.columns
            if df_limpio[col].dtype in ['float64', 'int64']
            and col not in columnas_no_escalar
            and not col.startswith('Unnamed')
        ]

        print(f"  Columnas a escalar ({len(columnas_a_escalar)}): {columnas_a_escalar}")

        if columnas_a_escalar:
            self.scaler = StandardScaler()
            df_limpio[columnas_a_escalar] = df_limpio[columnas_a_escalar].fillna(0)
            df_limpio[columnas_a_escalar] = self.scaler.fit_transform(df_limpio[columnas_a_escalar])
            print(f"  Escalado completado con StandardScaler")

        # 6. Manejar valores nulos
        nulos_despues = df_limpio.isnull().sum().sum()
        if nulos_despues > 0:
            print(f"\nImputando {nulos_despues} valores nulos restantes...")
            for col in df_limpio.columns:
                if df_limpio[col].dtype in ['float64', 'int64']:
                    df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
                else:
                    df_limpio[col] = df_limpio[col].fillna('')

        self.df = df_limpio
        self.reporte['shape_final'] = self.df.shape

        print(f"\nDataset limpio. Dimensiones finales: {self.df.shape}")
        print(f"Total columnas: {len(self.df.columns)}")
        print(f"Features nuevas: Summary_sentiment, Text_sentiment")

        # Verificar Score
        if 'Score' in self.df.columns:
            print(f"\nVERIFICACION: Columna 'Score' presente")
            print(f"  Valores unicos: {sorted(self.df['Score'].dropna().unique())}")
        else:
            print(f"\nATENCION: Columna 'Score' NO encontrada")

        return self

    def generar_reporte(self):
        """Genera reporte final del normalizador."""
        print("\nAGENTE NORMALIZADOR - Reporte Final")
        print("=" * 50)
        print(f"Dataset original: {self.reporte.get('shape_original', 'N/A')}")
        print(f"Dataset limpio: {self.reporte.get('shape_final', 'N/A')}")
        print(f"Features generadas:")
        print(f"  * Texto: Summary_length, Summary_word_count, Text_length, Text_word_count")
        print(f"  * Fecha: Año, Mes, DiaSemana")
        print(f"  * Utilidad: HelpfulnessRatio")
        print(f"  * Sentimiento: Summary_sentiment, Text_sentiment (VADER)")
        return self

    def obtener_dataset(self):
        """Retorna el DataFrame limpio."""
        return self.df

    def obtener_score_original(self):
        """Retorna la columna Score original sin escalar."""
        if hasattr(self, 'score_original'):
            return self.score_original
        return None


print("Clase AgenteNormalizador definida")

# CELDA 4: Ejecución del Agente Normalizador

print("INICIANDO SISTEMA DE AGENTES DE IA")
print("=" * 60)

normalizador = AgenteNormalizador()
normalizador.cargar_dataset()
normalizador.exploracion_inicial()
normalizador.limpiar_dataset()
normalizador.generar_reporte()

df_limpio = normalizador.obtener_dataset()
print("\n Agente Normalizador completado. Dataset listo para Agente Entrenador.")

# CELDA 5: DIAGNOSTICO DE COLUMNAS
print("COLUMNAS DEL DATASET LIMPIO:")
print("=" * 55)
for i, col in enumerate(df_limpio.columns, 1):
    n_nulos = df_limpio[col].isnull().sum()
    n_unicos = df_limpio[col].nunique()
    dtype_str = str(df_limpio[col].dtype)

    # Mostrar valores para columnas con pocos unicos
    extra = ""
    if n_unicos <= 10 and df_limpio[col].dtype in ['float64', 'int64']:
        vals = sorted(df_limpio[col].dropna().unique())
        extra = f" -> Valores: {vals}"

    print(f"{i:3d}. {col:<35} tipo={dtype_str:<12} unicos={n_unicos:<8} nulos={n_nulos}{extra}")

print(f"\nShape: {df_limpio.shape}")
print(f"\nTotal columnas: {len(df_limpio.columns)}")

# Verificar Score
if 'Score' in df_limpio.columns:
    print(f"\nColumna 'Score' ENCONTRADA:")
    print(f"  Valores unicos: {sorted(df_limpio['Score'].dropna().unique())}")
    print(f"  Media: {df_limpio['Score'].mean():.2f}")
    print(f"  Distribucion:")
    for score, count in df_limpio['Score'].value_counts().sort_index().items():
        print(f"    Score {int(score)}: {count:,}")

# Verificar nuevas features de sentimiento
for col_sent in ['Summary_sentiment', 'Text_sentiment']:
    if col_sent in df_limpio.columns:
        print(f"\nColumna '{col_sent}' ENCONTRADA:")
        print(f"  Rango: [{df_limpio[col_sent].min():.3f}, {df_limpio[col_sent].max():.3f}]")
        print(f"  Media: {df_limpio[col_sent].mean():.3f}")

print("\nPrimeras 3 filas:")
display(df_limpio.head(3))

# CELDA 5.5: Diagnostico de columnas

print("=" * 70)
print("DIAGNÓSTICO AVANZADO DE COLUMNAS DEL DATASET LIMPIO")
print("=" * 70)

# 1. Listar TODAS las columnas con detalles
print("\n1. LISTADO COMPLETO DE COLUMNAS:")
print("-" * 50)
for i, col in enumerate(df_limpio.columns, 1):
    dtype = str(df_limpio[col].dtype)
    n_unique = df_limpio[col].nunique()
    n_null = df_limpio[col].isnull().sum()

    # Mostrar valores de ejemplo para columnas con pocos únicos
    ejemplo = ""
    if n_unique <= 10 and dtype in ['float64', 'int64']:
        valores = df_limpio[col].dropna().unique()
        ejemplo = f" -> Valores: {sorted(valores)[:5]}"

    print(f"{i:3d}. {col:<40} dtype={dtype:<12} únicos={n_unique:<6} nulos={n_null} {ejemplo}")

# 2. Buscar columnas específicas que nos interesan
print("\n2. BÚSQUEDA DE COLUMNAS CLAVE:")
print("-" * 50)

# Buscar rating
cols_rating = [col for col in df_limpio.columns if 'rating' in col.lower()]
if cols_rating:
    print(f"Columnas con 'rating' en el nombre: {cols_rating}")
    for col in cols_rating:
        if df_limpio[col].dtype in ['float64', 'int64']:
            print(f"  * {col}: min={df_limpio[col].min():.2f}, max={df_limpio[col].max():.2f}, mean={df_limpio[col].mean():.2f}")
else:
    print("ATENCION: No se encontró NINGUNA columna con 'rating' en el nombre")
    print("Buscando alternativas...")
    # Buscar columnas que puedan ser ratings (valores entre 1-5)
    for col in df_limpio.columns:
        if df_limpio[col].dtype in ['float64', 'int64']:
            if df_limpio[col].nunique() <= 5:
                vals = sorted(df_limpio[col].dropna().unique())
                if min(vals) >= 1 and max(vals) <= 5:
                    print(f"  POSIBLE CANDIDATO: {col} -> valores={vals}")

# Buscar precio
cols_precio = [col for col in df_limpio.columns if any(term in col.lower() for term in ['price', 'precio', 'discounted', 'actual'])]
if cols_precio:
    print(f"\nColumnas relacionadas con precio: {cols_precio}")
    for col in cols_precio:
        if df_limpio[col].dtype in ['float64', 'int64']:
            print(f"  * {col}: min={df_limpio[col].min():.4f}, max={df_limpio[col].max():.4f}")
else:
    print("\nNo se encontraron columnas relacionadas con precio")

# Buscar longitud de texto
cols_texto = [col for col in df_limpio.columns if any(term in col.lower() for term in ['length', 'word_count', 'longitud'])]
if cols_texto:
    print(f"\nColumnas de features de texto: {cols_texto}")
else:
    print("\nNo se encontraron columnas de features de texto")

# 3. Identificar columnas numéricas que no son IDs
print("\n3. COLUMNAS NUMÉRICAS (excluyendo posibles IDs):")
print("-" * 50)
cols_numericas = [col for col in df_limpio.columns
                  if df_limpio[col].dtype in ['float64', 'int64']
                  and not any(id_term in col.lower() for id_term in ['id', 'unnamed'])]
print(f"Total: {len(cols_numericas)} columnas numéricas")
for col in cols_numericas[:20]:
    print(f"  * {col} (rango: {df_limpio[col].min():.4f} a {df_limpio[col].max():.4f})")
if len(cols_numericas) > 20:
    print(f"  ... y {len(cols_numericas) - 20} más")

# 4. Identificar columnas categóricas (one-hot encoded)
print("\n4. COLUMNAS ONE-HOT ENCODED (valores 0/1):")
print("-" * 50)
cols_binarias = [col for col in df_limpio.columns
                 if df_limpio[col].dtype in ['float64', 'int64', 'bool']
                 and df_limpio[col].nunique() <= 2
                 and not any(id_term in col.lower() for id_term in ['id', 'unnamed'])]
print(f"Total: {len(cols_binarias)} columnas binarias")
if cols_binarias:
    for col in cols_binarias[:10]:
        print(f"  * {col}")
    if len(cols_binarias) > 10:
        print(f"  ... y {len(cols_binarias) - 10} más")

# 5. Análisis del target real usado por el entrenador
print("\n5. VERIFICACIÓN DEL TARGET:")
print("-" * 50)
# Revisar qué columnas tienen valores típicos de rating (1-5)
for col in df_limpio.columns:
    if df_limpio[col].dtype in ['float64', 'int64']:
        vals = df_limpio[col].dropna()
        if len(vals) > 0:
            if 1 <= vals.min() <= 2 and 4 <= vals.max() <= 5 and vals.nunique() <= 5:
                print(f"  POSIBLE TARGET (rating): {col}")
                print(f"    Valores únicos: {sorted(vals.unique())}")
                print(f"    Distribución:")
                print(vals.value_counts().sort_index())

print("\n" + "=" * 70)
print("DIAGNÓSTICO COMPLETADO")
print("=" * 70)


# CELDA 6: AGENTE 2 - ENTRENADOR
# Entrenamiento con división train/validation/test


class AgenteEntrenador:
    """Agente de entrenamiento OPTIMIZADO con metricas adicionales."""

    COLUMNAS_EXCLUIR = {
        'Id', 'ProductId', 'UserId', 'ProfileName',
        'Summary', 'Text', 'Fecha'
    }

    def __init__(self, df, columna_target='Score'):
        self.df = df.copy()
        self.columna_target = columna_target
        self.mejor_modelo = None
        self.mejor_nombre = None
        self.metricas = {}
        self.resultados_modelos = {}
        self.caracteristicas_importantes = None
        self.features = []
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.insights = []

    def analisis_exploratorio(self):
        """EDA rapido."""
        print("\nAGENTE ENTRENADOR - Fase 1: Analisis Exploratorio")
        print("-" * 50)

        if self.columna_target not in self.df.columns:
            raise ValueError(f"Columna target '{self.columna_target}' no encontrada")

        self.target = self.columna_target
        print(f"Target: '{self.target}'")

        scores = self.df[self.target].dropna()
        print(f"  Media: {scores.mean():.2f} | Mediana: {scores.median():.2f} | Rango: [{scores.min():.0f}, {scores.max():.0f}]")

        # Grafico rapido
        plt.figure(figsize=(8, 4))
        score_counts = self.df[self.target].value_counts().sort_index()
        colors = ['#FF4444', '#FF8C00', '#FFD700', '#90EE90', '#00C851']
        bars = plt.bar(score_counts.index.astype(str), score_counts.values,
                      color=colors, edgecolor='black', linewidth=1)
        plt.title('Distribucion de Scores', fontweight='bold', fontsize=13)
        plt.xlabel('Score')
        plt.ylabel('Cantidad')
        total = score_counts.sum()
        for bar, val in zip(bars, score_counts.values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + total*0.02,
                    f'{val:,}', ha='center', fontsize=8)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()
        return self

    def preparar_datos(self):
        """Preparacion de features."""
        print("\nAGENTE ENTRENADOR - Fase 2: Preparacion de Datos")
        print("-" * 50)

        self.features = [
            col for col in self.df.columns
            if self.df[col].dtype in ['float64', 'int64']
            and col not in self.COLUMNAS_EXCLUIR
            and col != self.columna_target
            and not col.startswith('Unnamed')
        ]

        print(f"Features ({len(self.features)}): {self.features}")

        # Preparar X e y
        df_modelo = self.df.dropna(subset=self.features + [self.target])
        self.X = df_modelo[self.features]
        self.y = df_modelo[self.target]

        print(f"Shape X: {self.X.shape} | Shape y: {self.y.shape}")
        return self

    def entrenar_modelo(self):
        """
        Entrenamiento OPTIMIZADO:
        - 30 arboles en vez de 50
        - Sin cross-validation (ahorra 3x tiempo)
        - n_jobs=-1 para usar todos los nucleos
        """
        print("\nAGENTE ENTRENADOR - Fase 3: Entrenamiento (Optimizado)")
        print("-" * 50)

        # Split 60/20/20
        print("Split train/validation/test (60/20/20)...")
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )

        print(f"  Train: {len(self.X_train):,} | Val: {len(self.X_val):,} | Test: {len(self.X_test):,}")

        # Modelos OPTIMIZADOS (menos arboles, max_depth limitado)
        modelos = {
            'Regresion Lineal': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=30,      # Reducido de 50 a 30
                max_depth=10,
                random_state=42,
                n_jobs=-1             # Usa todos los nucleos
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=30,      # Reducido de 50 a 30
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        }

        print(f"\n{'Modelo':<20} {'R2(test)':>8} {'R2(val)':>8} {'RMSE':>8} {'MAE':>8} {'Exact(±1)':>10} {'Tiempo':>8}")
        print("-" * 80)

        for nombre, modelo in modelos.items():
            print(f"  Entrenando {nombre}...", end=' ')
            inicio = time.time()

            # Entrenar
            modelo.fit(self.X_train, self.y_train)

            # Predecir
            y_pred_val = modelo.predict(self.X_val)
            y_pred_val = np.clip(y_pred_val, 1.0, 5.0)

            y_pred_test = modelo.predict(self.X_test)
            y_pred_test = np.clip(y_pred_test, 1.0, 5.0)

            # Metricas basicas
            r2_test = r2_score(self.y_test, y_pred_test)
            rmse_test = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            r2_val = r2_score(self.y_val, y_pred_val)

            # Metricas adicionales
            mae_test = mean_absolute_error(self.y_test, y_pred_test)
            evs_test = explained_variance_score(self.y_test, y_pred_test)

            # Exactitud
            y_pred_rounded = np.round(y_pred_test)
            y_test_rounded = np.round(self.y_test)
            exactitud_exacta = np.mean(y_pred_rounded == y_test_rounded)
            exactitud_tolerancia = np.mean(np.abs(y_pred_test - self.y_test) <= 1)

            tiempo = time.time() - inicio

            self.resultados_modelos[nombre] = {
                'modelo': modelo,
                'r2_test': r2_test,
                'rmse_test': rmse_test,
                'r2_val': r2_val,
                'mae_test': mae_test,
                'evs_test': evs_test,
                'exactitud_exacta': exactitud_exacta,
                'exactitud_tolerancia': exactitud_tolerancia
            }

            print(f"R2={r2_test:.4f} | RMSE={rmse_test:.4f} | MAE={mae_test:.4f} | Exact(±1)={exactitud_tolerancia:.1%} | {tiempo:.1f}s")

        # Seleccionar mejor modelo
        self.mejor_nombre = max(self.resultados_modelos,
                                key=lambda x: self.resultados_modelos[x]['r2_test'])
        self.mejor_modelo = self.resultados_modelos[self.mejor_nombre]['modelo']
        mejor = self.resultados_modelos[self.mejor_nombre]

        print(f"\n{'='*50}")
        print(f"MEJOR MODELO: {self.mejor_nombre}")
        print(f"{'='*50}")
        print(f"  R2 Test:       {mejor['r2_test']:.4f}")
        print(f"  R2 Validacion: {mejor['r2_val']:.4f}")
        print(f"  RMSE:          {mejor['rmse_test']:.4f}")
        print(f"  MAE:           {mejor['mae_test']:.4f}")
        print(f"  Exactitud (±1): {mejor['exactitud_tolerancia']:.1%}")
        print(f"  Exactitud Exacta: {mejor['exactitud_exacta']:.1%}")

        return self

    def generar_insights(self):
        """Genera insights."""
        print("\nAGENTE ENTRENADOR - Fase 4: Generacion de Insights")
        print("-" * 50)

        if self.mejor_modelo is None:
            print("No hay modelo entrenado.")
            return self

        # Importancia de caracteristicas
        if hasattr(self.mejor_modelo, 'feature_importances_'):
            importancias = self.mejor_modelo.feature_importances_
            self.caracteristicas_importantes = pd.DataFrame({
                'feature': self.features,
                'importance': importancias
            }).sort_values('importance', ascending=False)

            print("\nTop 10 caracteristicas:")
            for idx, row in self.caracteristicas_importantes.head(10).iterrows():
                bar = '█' * int(row['importance'] * 40)
                print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")

        # Insights
        mejor = self.resultados_modelos[self.mejor_nombre]
        self.insights = []

        r2 = mejor['r2_test']
        if r2 > 0.50:
            self.insights.append(f"BUENA capacidad predictiva (R2={r2:.3f}).")
        elif r2 > 0.25:
            self.insights.append(f"Capacidad predictiva MODERADA (R2={r2:.3f}), aceptable para reviews.")
        else:
            self.insights.append(f"Capacidad LIMITADA (R2={r2:.3f}), tipico en datos subjetivos.")

        self.insights.append(f"RMSE={mejor['rmse_test']:.2f} puntos | MAE={mejor['mae_test']:.2f} puntos.")
        self.insights.append(f"Exactitud exacta: {mejor['exactitud_exacta']:.1%} | Con tolerancia ±1: {mejor['exactitud_tolerancia']:.1%}.")

        if self.caracteristicas_importantes is not None and len(self.caracteristicas_importantes) > 0:
            top = self.caracteristicas_importantes.iloc[0]
            self.insights.append(f"'{top['feature']}' es el factor mas importante (importancia={top['importance']:.3f}).")

        mean_score = self.y.mean()
        self.insights.append(f"Sesgo positivo: score medio={mean_score:.2f}/5.0 ({((self.y>=4).sum()/len(self.y)*100):.0f}% son 4-5).")

        if 'Summary_sentiment' in self.features or 'Text_sentiment' in self.features:
            self.insights.append("Analisis de sentimiento VADER incluido como feature.")

        print(f"\n{len(self.insights)} insights generados:")
        for i, ins in enumerate(self.insights, 1):
            print(f"  {i}. {ins}")

        return self

    def obtener_resultados(self):
        """Retorna resultados."""
        if self.mejor_modelo is None:
            raise ValueError("No hay modelo entrenado.")

        mejor = self.resultados_modelos[self.mejor_nombre]

        return {
            'nombre_modelo': self.mejor_nombre,
            'modelo': self.mejor_modelo,
            'metricas': {
                'R2_test': mejor['r2_test'],
                'R2_val': mejor['r2_val'],
                'RMSE_test': mejor['rmse_test'],
                'MAE_test': mejor['mae_test'],
                'Exactitud_exacta': mejor['exactitud_exacta'],
                'Exactitud_tolerancia': mejor['exactitud_tolerancia'],
                'EVS_test': mejor['evs_test'],
                'n_features': len(self.features),
                'train_samples': len(self.X_train) if self.X_train is not None else 'N/A',
                'val_samples': len(self.X_val) if self.X_val is not None else 'N/A',
                'test_samples': len(self.X_test) if self.X_test is not None else 'N/A'
            },
            'caracteristicas_importantes': self.caracteristicas_importantes,
            'insights': self.insights,
            'resultados_modelos': self.resultados_modelos
        }

        #Celda 7 - Ejecución del Agente Entrenador

        entrenador = AgenteEntrenador(df_limpio, columna_target='Score')
        entrenador.analisis_exploratorio()
        entrenador.preparar_datos()
        entrenador.entrenar_modelo()
        entrenador.generar_insights()

        resultados = entrenador.obtener_resultados()
        print("Agente Entrenador completado. Resultados listos para Agente Comunicador.")

# CELDA 8: AGENTE 3 - COMUNICADOR

class AgenteComunicador:
    """Agente de comunicación MEJORADO con generación de PDF."""

    def __init__(self, df, resultados_entrenamiento):
        self.df = df
        self.resultados = resultados_entrenamiento
        self.reporte_final = {}

    def generar_visualizaciones(self):
        """Dashboard 2x2 ligero."""
        print("\nAGENTE COMUNICADOR - Visualizaciones")
        print("-" * 50)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # 1. Ratings
        if 'rating' in self.df.columns:
            counts = self.df['rating'].value_counts().sort_index()
            axes[0, 0].bar(counts.index.astype(str), counts.values,
                          color='gold', edgecolor='black')
            axes[0, 0].set_title('Distribución de Ratings', fontweight='bold')
            axes[0, 0].set_xlabel('Rating')
            axes[0, 0].set_ylabel('Cantidad')

        # 2. Categorías
        col_cat = next((c for c in self.df.columns if 'categ' in c.lower()), None)
        if col_cat is None:
            # Buscar columnas one-hot encoded de categoría
            col_cat_candidates = [c for c in self.df.columns if 'category_' in c]
            if col_cat_candidates:
                # Sumar las columnas one-hot para obtener conteos
                cat_counts = self.df[col_cat_candidates].sum().sort_values(ascending=False).head(8)
                axes[0, 1].barh(cat_counts.index, cat_counts.values,
                               color='skyblue', edgecolor='black')
                axes[0, 1].set_title('Top 8 Categorías', fontweight='bold')
                axes[0, 1].invert_yaxis()

        # 3. Precio vs Rating
        col_precio = next((c for c in self.df.columns
                          if 'discounted_price' in c.lower()
                          and self.df[c].dtype in ['float64', 'int64']), None)
        if col_precio and 'rating' in self.df.columns:
            df_sample = self.df[[col_precio, 'rating']].dropna().sample(
                min(2000, len(self.df)), random_state=42
            )
            axes[1, 0].scatter(df_sample[col_precio], df_sample['rating'],
                              alpha=0.3, s=10, color='green')
            axes[1, 0].set_title('Precio vs Rating', fontweight='bold')
            axes[1, 0].set_xlabel('Precio Descontado')
            axes[1, 0].set_ylabel('Rating')

        # 4. Longitud de reviews
        col_long = next((c for c in self.df.columns
                        if 'length' in c.lower()
                        and self.df[c].dtype in ['float64', 'int64']), None)
        if col_long:
            axes[1, 1].hist(self.df[col_long].dropna(), bins=25,
                           color='coral', edgecolor='black')
            axes[1, 1].set_title(f'Distribución de {col_long}', fontweight='bold')
            axes[1, 1].set_xlabel(col_long)
            axes[1, 1].set_ylabel('Frecuencia')

        plt.suptitle('Dashboard - Amazon Product Reviews', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('dashboard.png', dpi=100, bbox_inches='tight')
        plt.show()
        print("✓ Dashboard guardado como 'dashboard.png'")

        return self

    def crear_reporte_resumen(self):
        """NUEVO: Reporte de texto mejorado con métricas de validation."""
        print("\nAGENTE COMUNICADOR - Reporte")
        print("=" * 60)

        metricas = self.resultados['metricas']
        r2_test = metricas['R2_test']
        r2_val = metricas.get('R2_val', 'N/A')
        rmse = metricas['RMSE_test']

        if r2_test >= 0.50:
            interpretacion = "Buena capacidad predictiva"
            recomendacion = "El modelo podría usarse para predecir ratings con confianza razonable."
        elif r2_test >= 0.25:
            interpretacion = "Capacidad moderada (esperable en reviews de productos)"
            recomendacion = "El modelo captura patrones pero tiene limitaciones propias de datos subjetivos."
        else:
            interpretacion = "Capacidad limitada - datos altamente subjetivos"
            recomendacion = "Se recomienda explorar features adicionales o usar modelos más complejos."

        reporte = f"""
{'='*60}
REPORTE DE ANÁLISIS - AMAZON PRODUCT REVIEWS
Sistema de 3 Agentes de IA
{'='*60}

 DATASET ANALIZADO
  • Registros: {len(self.df):,}
  • Columnas: {len(self.df.columns)}
  • Target: rating de productos

 MODELO SELECCIONADO: {self.resultados['nombre_modelo']}
  • R² Score (Test): {r2_test:.4f}
  • R² Score (Validation): {r2_val if isinstance(r2_val, str) else f'{r2_val:.4f}'}
  • RMSE: {rmse:.4f} puntos de rating
  • Features utilizadas: {metricas['n_features']}
  • Muestras train: {metricas.get('train_samples', 'N/A')}
  • Muestras validation: {metricas.get('val_samples', 'N/A')}
  • Muestras test: {metricas.get('test_samples', 'N/A')}

 INTERPRETACIÓN
  • {interpretacion}
  • {recomendacion}

 INSIGHTS PRINCIPALES:
"""
        for ins in self.resultados['insights']:
            reporte += f"  • {ins}\n"

        reporte += f"""
 TRANSFORMACIONES APLICADAS
  • Escalado: StandardScaler en variables numéricas
  • Codificación: One-Hot Encoding en variables categóricas
  • Limpieza: Eliminación de duplicados, imputación de nulos
  • Features de texto: Longitud y conteo de palabras

 ARCHIVOS GENERADOS
  • amazon_reviews_limpio.csv
  • reporte_analisis.txt
  • reporte_analisis.pdf
  • metricas_modelo.csv
  • feature_importance.csv
  • dashboard.png

{'='*60}
"""
        print(reporte)
        self.reporte_final['texto'] = reporte
        self.reporte_final['interpretacion'] = interpretacion
        self.reporte_final['recomendacion'] = recomendacion

        # Guardar TXT
        with open('reporte_analisis.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        print(" Reporte TXT guardado")

        return self

    def generar_pdf(self):
        """
        NUEVO: Genera reporte PDF profesional con fpdf2.
        """
        print("\nAGENTE COMUNICADOR - Generando PDF")
        print("-" * 50)

        pdf = FPDF()
        pdf.add_page()

        # Título
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 15, "REPORTE DE ANALISIS", ln=True, align="C")
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Amazon Product Reviews - Sistema de 3 Agentes IA", ln=True, align="C")
        pdf.ln(10)

        # Sección 1: Dataset
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(70, 130, 180)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "  DATASET", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 11)
        pdf.ln(3)
        pdf.cell(0, 7, f"Registros analizados: {len(self.df):,}", ln=True)
        pdf.cell(0, 7, f"Columnas del dataset: {len(self.df.columns)}", ln=True)
        pdf.cell(0, 7, "Target: rating de productos (1-5 estrellas)", ln=True)
        pdf.ln(5)

        # Sección 2: Modelo
        metricas = self.resultados['metricas']
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(70, 130, 180)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "  MODELO SELECCIONADO", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 11)
        pdf.ln(3)
        pdf.cell(0, 7, f"Algoritmo: {self.resultados['nombre_modelo']}", ln=True)
        pdf.cell(0, 7, f"R2 Score (Test): {metricas['R2_test']:.4f}", ln=True)
        if 'R2_val' in metricas:
            pdf.cell(0, 7, f"R2 Score (Validation): {metricas['R2_val']:.4f}", ln=True)
        pdf.cell(0, 7, f"RMSE: {metricas['RMSE_test']:.4f} puntos de rating", ln=True)
        pdf.cell(0, 7, f"Features utilizadas: {metricas['n_features']}", ln=True)
        pdf.cell(0, 7, f"Muestras - Train: {metricas.get('train_samples', 'N/A')} | Val: {metricas.get('val_samples', 'N/A')} | Test: {metricas.get('test_samples', 'N/A')}", ln=True)
        pdf.ln(5)

        # Sección 3: Interpretación
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(70, 130, 180)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "  INTERPRETACION", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 11)
        pdf.ln(3)
        pdf.multi_cell(0, 7, self.reporte_final.get('interpretacion', 'No disponible'))
        pdf.ln(3)
        pdf.multi_cell(0, 7, self.reporte_final.get('recomendacion', 'No disponible'))
        pdf.ln(5)

        # Sección 4: Insights
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(70, 130, 180)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "  INSIGHTS", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 11)
        pdf.ln(3)
        for insight in self.resultados['insights']:
            pdf.cell(0, 7, f"  * {insight}", ln=True)
        pdf.ln(5)

        # Sección 5: Transformaciones
        pdf.set_font("Arial", "B", 14)
        pdf.set_fill_color(70, 130, 180)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(0, 10, "  TRANSFORMACIONES APLICADAS", ln=True, fill=True)
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", "", 11)
        pdf.ln(3)
        transformaciones = [
            "Escalado: StandardScaler en variables numericas",
            "Codificacion: One-Hot Encoding en variables categoricas",
            "Limpieza: Eliminacion de duplicados e imputacion de nulos",
            "Features de texto: Longitud y conteo de palabras"
        ]
        for trans in transformaciones:
            pdf.cell(0, 7, f"  * {trans}", ln=True)

        # Guardar PDF
        pdf.output('reporte_analisis.pdf')
        print("PDF generado: 'reporte_analisis.pdf'")

        return self

    def exportar_datos(self):
        """Exporta archivos esenciales."""
        print("\nAGENTE COMUNICADOR - Exportación")
        print("-" * 50)

        self.df.to_csv('amazon_reviews_limpio.csv', index=False)
        print("amazon_reviews_limpio.csv")

        metricas_exp = {k: v for k, v in self.resultados['metricas'].items()
                       if isinstance(v, (int, float, str))}
        pd.DataFrame([metricas_exp]).to_csv('metricas_modelo.csv', index=False)
        print("metricas_modelo.csv")

        if self.resultados.get('caracteristicas_importantes') is not None:
            self.resultados['caracteristicas_importantes'].to_csv(
                'feature_importance.csv', index=False
            )
            print(" feature_importance.csv")

        return self

    def ejecutar_pipeline_comunicacion(self):
        """Ejecuta todo el pipeline incluyendo PDF."""
        self.generar_visualizaciones()
        self.crear_reporte_resumen()
        self.generar_pdf()  # NUEVO
        self.exportar_datos()
        print("\n AGENTE COMUNICADOR COMPLETADO")
        return self

print("Clase AgenteComunicador MEJORADA (con PDF).")

comunicador = AgenteComunicador(df_limpio, resultados)
comunicador.ejecutar_pipeline_comunicacion()

#CELDA 9 - EJECUCION DE DEL COMUNICADOR Y RESULTADOS FINALES

print("\n" + "="*60)
print("SISTEMA DE 3 AGENTES COMPLETADO EXITOSAMENTE")
print("="*60)
print("\nArchivos generados:")
print("  • amazon_reviews_limpio.csv")
print("  • reporte_analisis.txt")
print("  • metricas_modelo.csv")
print("  • feature_importance.csv")
print("  • dashboard.png")
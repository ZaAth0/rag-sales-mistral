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
    """Agente de comunicacion MEJORADO con matriz de confusion y PDF profesional."""

    def __init__(self, df, resultados_entrenamiento, entrenador_obj=None, columna_target='Score'):
        self.df = df
        self.resultados = resultados_entrenamiento
        self.entrenador = entrenador_obj
        self.columna_target = columna_target
        self.reporte_final = {}
        self.cols_numericas = [col for col in self.df.columns
                               if self.df[col].dtype in ['float64', 'int64']]

    def generar_visualizaciones(self):
        """Dashboard 2x2 MEJORADO con matriz de confusion."""
        print("\nAGENTE COMUNICADOR - Visualizaciones")
        print("-" * 50)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        plots_generados = 0

        
        # GRAFICO 1: Distribucion de Scores
        
        print("Grafico 1: Distribucion de Scores...")
        if self.columna_target in self.df.columns:
            scores = self.df[self.columna_target].value_counts().sort_index()
            colors = ['#FF4444', '#FF8C00', '#FFD700', '#90EE90', '#00C851']
            bars = axes[0, 0].bar(scores.index.astype(str), scores.values,
                                 color=colors[:len(scores)], edgecolor='black', linewidth=1.5)
            axes[0, 0].set_title('Distribucion de Scores (1-5 estrellas)',
                                fontweight='bold', fontsize=13)
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_ylabel('Cantidad de Reviews')

            total = scores.sum()
            for bar, val in zip(bars, scores.values):
                axes[0, 0].text(bar.get_x() + bar.get_width()/2,
                               bar.get_height() + total*0.02,
                               f'{val:,}\n({val/total*100:.1f}%)',
                               ha='center', fontsize=9, fontweight='bold')
            axes[0, 0].grid(axis='y', alpha=0.3)
            plots_generados += 1
            print(f"  * OK")
        else:
            axes[0, 0].text(0.5, 0.5, f'Columna "{self.columna_target}" no encontrada',
                           ha='center', va='center', transform=axes[0, 0].transAxes, fontsize=12)
            print(f"  * ATENCION: Score no encontrado")

        
        # GRAFICO 2: Importancia de Caracteristicas
        
        print("Grafico 2: Importancia de Caracteristicas...")
        if (self.resultados.get('caracteristicas_importantes') is not None and
            len(self.resultados['caracteristicas_importantes']) > 0):

            top = self.resultados['caracteristicas_importantes'].head(10)
            colors_blue = plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))
            axes[0, 1].barh(top['feature'], top['importance'],
                           color=colors_blue, edgecolor='black', linewidth=1)
            axes[0, 1].set_title('Top 10 Caracteristicas Importantes',
                                fontweight='bold', fontsize=13)
            axes[0, 1].set_xlabel('Importancia')
            axes[0, 1].invert_yaxis()
            axes[0, 1].grid(axis='x', alpha=0.3)

            # Agregar valores
            for i, (idx, row) in enumerate(top.iterrows()):
                axes[0, 1].text(row['importance'] + 0.01, i,
                               f"{row['importance']:.3f}", va='center', fontsize=9)
            plots_generados += 1
            print(f"  * {len(top)} caracteristicas")
        else:
            axes[0, 1].text(0.5, 0.5, 'No hay datos de importancia',
                           ha='center', va='center', transform=axes[0, 1].transAxes, fontsize=12)

        
        # GRAFICO 3: Scatter - Variable principal vs Score
        
        print("Grafico 3: Variable principal vs Score...")
        var_x = None
        if (self.resultados.get('caracteristicas_importantes') is not None and
            len(self.resultados['caracteristicas_importantes']) > 0):
            var_x = self.resultados['caracteristicas_importantes'].iloc[0]['feature']

        if var_x is None:
            for col in self.cols_numericas:
                if col != self.columna_target and 'Id' not in col:
                    var_x = col
                    break

        if var_x and var_x in self.df.columns and self.columna_target in self.df.columns:
            df_plot = self.df[[var_x, self.columna_target]].dropna()
            if len(df_plot) > 3000:
                df_plot = df_plot.sample(3000, random_state=42)

            axes[1, 0].scatter(df_plot[var_x], df_plot[self.columna_target],
                              alpha=0.3, s=20, c='steelblue', edgecolors='none')
            axes[1, 0].set_title(f'{var_x} vs {self.columna_target}',
                                fontweight='bold', fontsize=13)
            axes[1, 0].set_xlabel(var_x)
            axes[1, 0].set_ylabel(self.columna_target)
            axes[1, 0].grid(alpha=0.3)
            plots_generados += 1
            print(f"  * {var_x}")
        else:
            axes[1, 0].text(0.5, 0.5, 'No se encontraron variables',
                           ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=12)

        
        # MEJORA 3: GRAFICO 4 - MATRIZ DE CONFUSION
        
        print("Grafico 4: Matriz de Confusion...")
        if self.entrenador is not None and hasattr(self.entrenador, 'mejor_modelo') and self.entrenador.mejor_modelo is not None:
            try:
                # Predecir en test
                y_pred = self.entrenador.mejor_modelo.predict(self.entrenador.X_test)
                y_pred_rounded = np.clip(np.round(y_pred), 1, 5).astype(int)
                y_test_rounded = np.round(self.entrenador.y_test).astype(int)

                # Matriz de confusion
                cm = confusion_matrix(y_test_rounded, y_pred_rounded, labels=[1, 2, 3, 4, 5])

                # Normalizar por fila
                cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                cm_normalized = np.nan_to_num(cm_normalized)

                # Graficar
                im = axes[1, 1].imshow(cm_normalized, cmap='Blues', aspect='auto', vmin=0, vmax=1)
                axes[1, 1].set_title('Matriz de Confusion Normalizada\n(Score Real vs Predicho)',
                                    fontweight='bold', fontsize=13)
                axes[1, 1].set_xlabel('Score Predicho')
                axes[1, 1].set_ylabel('Score Real')
                axes[1, 1].set_xticks(range(5))
                axes[1, 1].set_xticklabels(['1', '2', '3', '4', '5'])
                axes[1, 1].set_yticks(range(5))
                axes[1, 1].set_yticklabels(['1', '2', '3', '4', '5'])

                # Agregar valores en celdas
                for i in range(5):
                    for j in range(5):
                        if cm[i, j] > 0:
                            text = axes[1, 1].text(j, i,
                                                 f'{cm_normalized[i, j]:.1%}\n({cm[i, j]:,})',
                                                 ha="center", va="center",
                                                 color="white" if cm_normalized[i, j] > 0.5 else "black",
                                                 fontsize=9, fontweight='bold')

                plt.colorbar(im, ax=axes[1, 1], label='Proporcion')
                plots_generados += 1
                print(f"  * Matriz generada correctamente")
            except Exception as e:
                axes[1, 1].text(0.5, 0.5, f'Error al generar matriz:\n{e}',
                               ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=10)
                print(f"  * Error: {e}")
        else:
            axes[1, 1].text(0.5, 0.5, 'Entrenador no disponible\npara matriz de confusion',
                           ha='center', va='center', transform=axes[1, 1].transAxes, fontsize=12)

        # Finalizar
        plt.suptitle('Dashboard - Amazon Fine Food Reviews\nSistema de 3 Agentes de IA',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig('dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
        plt.show()

        print(f"\n{'='*50}")
        print(f"Dashboard: {plots_generados}/4 graficos generados")
        print("Guardado como: dashboard.png")
        print(f"{'='*50}")

        return self

    def crear_reporte_resumen(self):
        """Reporte MEJORADO con metricas adicionales."""
        print("\nAGENTE COMUNICADOR - Generando Reporte")
        print("=" * 60)

        metricas = self.resultados['metricas']
        r2_test = metricas['R2_test']
        r2_val = metricas.get('R2_val', 'N/A')
        rmse = metricas['RMSE_test']
        mae = metricas.get('MAE_test', 'N/A')
        exactitud = metricas.get('Exactitud_tolerancia', 'N/A')
        exactitud_exacta = metricas.get('Exactitud_exacta', 'N/A')

        # Interpretacion
        if r2_test >= 0.50:
            interpretacion = "BUENA capacidad predictiva"
            recomendacion = "El modelo predice scores con buena precision."
        elif r2_test >= 0.25:
            interpretacion = "Capacidad predictiva MODERADA"
            recomendacion = "Rendimiento aceptable y esperable en datos subjetivos de reviews de alimentos."
        else:
            interpretacion = "Capacidad predictiva LIMITADA"
            recomendacion = "Las reviews de alimentos son inherentemente dificiles de predecir por su subjetividad."

        reporte = f"""
{'='*70}
REPORTE DE ANALISIS - AMAZON FINE FOOD REVIEWS
Sistema de 3 Agentes de IA con Analisis de Sentimiento VADER
{'='*70}

1. DATASET ANALIZADO
   * Nombre: Amazon Fine Food Reviews
   * Registros: {len(self.df):,}
   * Columnas totales: {len(self.df.columns)}
   * Target: Score (1-5 estrellas)
   * Periodo: Reviews de alimentos en Amazon

2. DISTRIBUCION DE SCORES
   * Score 1: {(self.df['Score']==1).sum():,} ({(self.df['Score']==1).sum()/len(self.df)*100:.1f}%)
   * Score 2: {(self.df['Score']==2).sum():,} ({(self.df['Score']==2).sum()/len(self.df)*100:.1f}%)
   * Score 3: {(self.df['Score']==3).sum():,} ({(self.df['Score']==3).sum()/len(self.df)*100:.1f}%)
   * Score 4: {(self.df['Score']==4).sum():,} ({(self.df['Score']==4).sum()/len(self.df)*100:.1f}%)
   * Score 5: {(self.df['Score']==5).sum():,} ({(self.df['Score']==5).sum()/len(self.df)*100:.1f}%)
   * Score medio: {self.df['Score'].mean():.2f}

3. MODELO SELECCIONADO: {self.resultados['nombre_modelo']}
   * R2 Score (Test):        {r2_test:.4f}
   * R2 Score (Validacion):  {r2_val if isinstance(r2_val, str) else f'{r2_val:.4f}'}
   * RMSE (Error cuadratico): {rmse:.4f} puntos
   * MAE (Error absoluto):    {mae if isinstance(mae, str) else f'{mae:.4f}'} puntos
   * Exactitud exacta:        {exactitud_exacta if isinstance(exactitud_exacta, str) else f'{exactitud_exacta:.1%}'}
   * Exactitud con tolerancia ±1: {exactitud if isinstance(exactitud, str) else f'{exactitud:.1%}'}
   * Features utilizadas: {metricas['n_features']}
   * Muestras Train:      {metricas.get('train_samples', 'N/A'):,}
   * Muestras Validacion: {metricas.get('val_samples', 'N/A'):,}
   * Muestras Test:       {metricas.get('test_samples', 'N/A'):,}

4. INTERPRETACION DE RESULTADOS
   * {interpretacion}
   * {recomendacion}
   * La exactitud con tolerancia de ±1 punto es del {exactitud if isinstance(exactitud, str) else f'{exactitud:.1%}'},
     lo que significa que el modelo acierta o se equivoca por solo 1 punto en la mayoria de casos.

5. INSIGHTS PRINCIPALES
"""
        for ins in self.resultados['insights']:
            reporte += f"   * {ins}\n"

        reporte += f"""
6. FEATURES Y TRANSFORMACIONES APLICADAS
   * Texto: Summary_length, Summary_word_count, Text_length, Text_word_count
   * Fecha: Año, Mes, DiaSemana (extraidos del timestamp)
   * Utilidad: HelpfulnessRatio (Numerator/Denominator)
   * Sentimiento VADER: Summary_sentiment, Text_sentiment (compound score)
   * Escalado: StandardScaler en variables numericas (excluyendo Score y sentiment)

7. RENDIMIENTO COMPARATIVO DE MODELOS
"""
        for nombre, metrics in self.resultados.get('resultados_modelos', {}).items():
            reporte += f"   * {nombre:<20} R2={metrics['r2_test']:.4f} | RMSE={metrics['rmse_test']:.4f} | MAE={metrics.get('mae_test', 'N/A'):.4f}\n"

        reporte += f"""
8. ARCHIVOS GENERADOS
   * amazon_reviews_limpio.csv     - Dataset procesado
   * reporte_analisis.txt          - Este reporte
   * reporte_analisis.pdf          - Reporte en formato PDF
   * metricas_modelo.csv           - Metricas de rendimiento
   * feature_importance.csv        - Importancia de caracteristicas
   * dashboard.png                 - Dashboard con 4 graficos

{'='*70}
FECHA DE GENERACION: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}
"""
        print(reporte)
        self.reporte_final['texto'] = reporte
        self.reporte_final['interpretacion'] = interpretacion
        self.reporte_final['recomendacion'] = recomendacion

        # Guardar TXT
        with open('reporte_analisis.txt', 'w', encoding='utf-8') as f:
            f.write(reporte)
        print("\nReporte TXT guardado: reporte_analisis.txt")

        return self

    def generar_pdf(self):
        """Genera PDF profesional."""
        print("\nAGENTE COMUNICADOR - Generando PDF...")

        try:
            pdf = FPDF()
            pdf.add_page()

            # Titulo principal
            pdf.set_font("Arial", "B", 18)
            pdf.set_text_color(0, 51, 102)
            pdf.cell(0, 12, "REPORTE DE ANALISIS", ln=True, align="C")
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 8, "Amazon Fine Food Reviews - Sistema de 3 Agentes IA", ln=True, align="C")
            pdf.set_font("Arial", "I", 10)
            pdf.set_text_color(100, 100, 100)
            pdf.cell(0, 6, f"Generado: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align="C")
            pdf.ln(8)

            # Contenido
            pdf.set_text_color(0, 0, 0)
            pdf.set_font("Arial", "", 9)

            lineas = self.reporte_final.get('texto', '').split('\n')
            for linea in lineas:
                linea_limpia = linea.encode('latin-1', 'replace').decode('latin-1')
                if linea_limpia.strip():
                    if linea_limpia.strip().startswith(('1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.')):
                        pdf.set_font("Arial", "B", 10)
                    else:
                        pdf.set_font("Arial", "", 9)
                    pdf.cell(0, 4.5, linea_limpia, ln=True)

            pdf.output('reporte_analisis.pdf')
            print("PDF generado: reporte_analisis.pdf")
        except Exception as e:
            print(f"Error al generar PDF: {e}")
            print("El reporte TXT esta disponible como alternativa")

        return self

    def exportar_datos(self):
        """Exporta archivos CSV."""
        print("\nAGENTE COMUNICADOR - Exportando datos...")
        print("-" * 50)

        self.df.to_csv('amazon_reviews_limpio.csv', index=False)
        print("  * amazon_reviews_limpio.csv")

        metricas_exp = {}
        for k, v in self.resultados['metricas'].items():
            if isinstance(v, (int, float, str, np.integer, np.floating)):
                metricas_exp[k] = float(v) if isinstance(v, (np.integer, np.floating)) else v
        pd.DataFrame([metricas_exp]).to_csv('metricas_modelo.csv', index=False)
        print("  * metricas_modelo.csv")

        if self.resultados.get('caracteristicas_importantes') is not None:
            self.resultados['caracteristicas_importantes'].to_csv(
                'feature_importance.csv', index=False
            )
            print("  * feature_importance.csv")

        print("Exportacion completada")
        return self

    def ejecutar_pipeline_comunicacion(self):
        """Ejecuta todo el pipeline de comunicacion."""
        self.generar_visualizaciones()
        self.crear_reporte_resumen()
        self.generar_pdf()
        self.exportar_datos()

        print("\n" + "="*60)
        print("AGENTE COMUNICADOR COMPLETADO EXITOSAMENTE")
        print("="*60)
        return self


#CELDA 9 - EJECUCION DE DEL COMUNICADOR Y RESULTADOS FINALES

comunicador = AgenteComunicador(
    df_limpio,
    resultados,
    entrenador_obj=entrenador,
    columna_target='Score'
)
comunicador.ejecutar_pipeline_comunicacion()

print("\n" + "="*70)
print("SISTEMA DE 3 AGENTES DE IA COMPLETADO EXITOSAMENTE")
print("="*70)
print("\nResumen de archivos generados:")
print("  1. amazon_reviews_limpio.csv     - Dataset procesado (568,454 registros)")
print("  2. reporte_analisis.txt          - Reporte completo en texto")
print("  3. reporte_analisis.pdf          - Reporte profesional en PDF")
print("  4. metricas_modelo.csv           - Metricas de rendimiento")
print("  5. feature_importance.csv        - Importancia de caracteristicas")
print("  6. dashboard.png                 - Dashboard con 4 graficos")
print("\nMejoras implementadas:")
print("  * Analisis de sentimiento VADER (Summary + Text)")
print("  * Metricas adicionales (MAE, Exactitud, EVS)")
print("  * Matriz de confusion en dashboard")
print("  * Reporte PDF profesional")
print("="*70)
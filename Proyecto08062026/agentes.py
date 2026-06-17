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
    Agente responsable de la carga, limpieza y normalización del dataset.
    Incluye validación de columnas esperadas y diagnóstico de tipos.
    """

    # Columnas que DEBERÍA tener el dataset según requisitos
    COLUMNAS_ESPERADAS = [
        'product_id', 'product_name', 'category',
        'discounted_price', 'actual_price', 'discount_percentage',
        'rating', 'rating_count', 'about_product',
        'user_id', 'user_name', 'review_id',
        'review_title', 'review_content', 'img_link', 'product_link'
    ]

    def __init__(self):
        self.df = None
        self.path_dataset = None
        self.reporte = {}
        self.nombre_dataset = None
        self.columnas_faltantes = []
        self.columnas_extra = []
        # NUEVO: Guardar objetos de transformación
        self.scaler = None
        self.label_encoders = {}
        self.df_original = None  # Para comparación

    def cargar_dataset(self):
        """Descarga y carga el dataset desde Kaggle usando kagglehub."""
        print("AGENTE NORMALIZADOR - Fase 1: Carga del Dataset")
        print("-" * 50)

        self.path_dataset = kagglehub.dataset_download("arhamrumi/amazon-product-reviews")
        print(f"Path descargado: {self.path_dataset}")

        archivos = os.listdir(self.path_dataset)
        archivos_csv = [f for f in archivos if f.endswith('.csv')]
        print(f"CSVs encontrados: {len(archivos_csv)}")

        if not archivos_csv:
            raise FileNotFoundError(
                f"No se encontró ningún archivo CSV en {self.path_dataset}"
            )

        # Probar cada CSV encontrado
        for archivo_csv in archivos_csv:
            ruta_completa = os.path.join(self.path_dataset, archivo_csv)
            try:
                df_temp = pd.read_csv(ruta_completa, nrows=5)
                columnas = df_temp.columns.tolist()
                print(f"\n  {archivo_csv}: {columnas}")

                if len(columnas) >= 5:
                    self.df = pd.read_csv(ruta_completa)
                    self.nombre_dataset = archivo_csv
                    self.df_original = self.df.copy()  # NUEVO: guardar copia original
                    print(f"\n✓ Dataset cargado: {archivo_csv}")
                    print(f"  Dimensiones: {self.df.shape}")
                    return self

            except Exception as e:
                print(f"  Error al leer {archivo_csv}: {e}")
                continue

        raise ValueError(
            f"No se pudo cargar ningún CSV válido. "
            f"Archivos: {archivos_csv}"
        )

    def validar_columnas(self):
        """
        MEJORA: Valida que el dataset tenga las columnas esperadas.
        Clasifica variables por tipo (numéricas, categóricas, texto).
        """
        print("\nAGENTE NORMALIZADOR - Fase 1.5: Validación de Columnas")
        print("-" * 50)

        columnas_actuales = set(self.df.columns)
        columnas_esperadas = set(self.COLUMNAS_ESPERADAS)

        # Columnas que faltan
        self.columnas_faltantes = list(columnas_esperadas - columnas_actuales)
        # Columnas extra que no estaban en la lista esperada
        self.columnas_extra = list(columnas_actuales - columnas_esperadas)

        if self.columnas_faltantes:
            print(f"  Columnas esperadas NO encontradas ({len(self.columnas_faltantes)}):")
            for col in self.columnas_faltantes:
                print(f"   - {col}")
        else:
            print("✓ Todas las columnas esperadas están presentes.")

        if self.columnas_extra:
            print(f"\n Columnas adicionales encontradas ({len(self.columnas_extra)}):")
            for col in self.columnas_extra:
                print(f"   + {col}")

        # Clasificar variables por tipo
        self.variables_numericas = []
        self.variables_categoricas = []
        self.variables_texto = []

        for col in self.df.columns:
            if self.df[col].dtype in ['float64', 'int64']:
                self.variables_numericas.append(col)
            elif self.df[col].dtype  'object':
                # Heurística: si tiene pocos valores únicos, es categórica
                if self.df[col].nunique() < 50:
                    self.variables_categoricas.append(col)
                else:
                    self.variables_texto.append(col)

        print(f"\n Clasificación de variables:")
        print(f"   Numéricas  : {len(self.variables_numericas)} → {self.variables_numericas[:5]}...")
        print(f"   Categóricas: {len(self.variables_categoricas)} → {self.variables_categoricas[:5]}...")
        print(f"   Texto      : {len(self.variables_texto)} → {self.variables_texto[:5]}...")

        # Verificar que hay al menos 1000 filas
        if len(self.df) < 1000:
            print(f"\n ADVERTENCIA: El dataset tiene solo {len(self.df)} filas (mínimo esperado: 1000)")
        else:
            print(f"\n Tamaño del dataset: {len(self.df):,} filas (cumple mínimo de 1000)")

        self.reporte['columnas_faltantes'] = self.columnas_faltantes
        self.reporte['columnas_extra'] = self.columnas_extra
        self.reporte['tipos_variables'] = {
            'numericas': len(self.variables_numericas),
            'categoricas': len(self.variables_categoricas),
            'texto': len(self.variables_texto)
        }

        return self

    def exploracion_inicial(self):
        """Realiza exploración inicial identificando tipos, nulos y estadísticas."""
        print("\nAGENTE NORMALIZADOR - Fase 2: Exploración Inicial")
        print("-" * 50)

        print(f"Dataset: {self.nombre_dataset}")
        print(f"Dimensiones: {self.df.shape[0]} filas, {self.df.shape[1]} columnas")

        print("\nTipos de datos por columna:")
        print(self.df.dtypes.to_string())

        print("\nPrimeras filas del dataset:")
        display(self.df.head())

        print("\nEstadísticas descriptivas (variables numéricas):")
        display(self.df.describe())

        print("\nAnálisis de valores nulos:")
        nulos = self.df.isnull().sum()
        nulos_pct = (self.df.isnull().sum() / len(self.df)) * 100

        df_nulos = pd.DataFrame({
            'Columna': nulos.index,
            'Valores_Nulos': nulos.values,
            'Porcentaje': nulos_pct.values
        })
        df_nulos = df_nulos[df_nulos['Valores_Nulos'] > 0].sort_values('Porcentaje', ascending=False)

        if len(df_nulos) > 0:
            display(df_nulos)
        else:
            print("No se encontraron valores nulos en el dataset.")

        self.reporte['shape_original'] = self.df.shape
        self.reporte['nulos'] = df_nulos

        return self

    def limpiar_dataset(self):
        """Ejecuta el pipeline completo de limpieza de datos."""
        print("\nAGENTE NORMALIZADOR - Fase 3: Limpieza de Datos")
        print("-" * 50)

        df_limpio = self.df.copy()

        # 1. Eliminar duplicados
        duplicados_antes = len(df_limpio)
        df_limpio = df_limpio.drop_duplicates()
        duplicados_despues = len(df_limpio)
        print(f"Duplicados eliminados: {duplicados_antes - duplicados_despues}")

        
        # Limpieza de columnas del dataset
        
        print("\nProcesando columnas del dataset de Amazon Product Reviews...")
        
        # 2. Limpiar precios (discounted_price, actual_price)
        for col_precio in ['discounted_price', 'actual_price']:
            if col_precio in df_limpio.columns:
                # Convertir strings con símbolos a numérico
                df_limpio[col_precio] = df_limpio[col_precio].astype(str)
                df_limpio[col_precio] = df_limpio[col_precio].str.replace('₹', '', regex=False)
                df_limpio[col_precio] = df_limpio[col_precio].str.replace('$', '', regex=False)
                df_limpio[col_precio] = df_limpio[col_precio].str.replace(',', '', regex=False)
                df_limpio[col_precio] = pd.to_numeric(df_limpio[col_precio], errors='coerce')
                print(f"  {col_precio}: convertido a numérico")
        
        # 3. Limpiar porcentaje de descuento
        if 'discount_percentage' in df_limpio.columns:
            df_limpio['discount_percentage'] = df_limpio['discount_percentage'].astype(str)
            df_limpio['discount_percentage'] = df_limpio['discount_percentage'].str.replace('%', '', regex=False)
            df_limpio['discount_percentage'] = pd.to_numeric(df_limpio['discount_percentage'], errors='coerce')
            print(f"   discount_percentage: convertido a numérico")
        
        # 4. Limpiar rating y rating_count
        if 'rating' in df_limpio.columns:
            df_limpio['rating'] = pd.to_numeric(df_limpio['rating'], errors='coerce')
            print(f"   rating: convertido a numérico")
        
        if 'rating_count' in df_limpio.columns:
            df_limpio['rating_count'] = df_limpio['rating_count'].astype(str)
            df_limpio['rating_count'] = df_limpio['rating_count'].str.replace(',', '', regex=False)
            df_limpio['rating_count'] = pd.to_numeric(df_limpio['rating_count'], errors='coerce')
            print(f"   rating_count: convertido a numérico")

        # 5. Limpiar texto (review_content, review_title, about_product)
        print("\nLimpiando columnas de texto...")
        for col_texto in ['review_content', 'review_title', 'about_product']:
            if col_texto in df_limpio.columns:
                df_limpio[f'{col_texto}_clean'] = (
                    df_limpio[col_texto]
                    .fillna('')
                    .apply(self._limpiar_texto)
                )
                # Crear features de texto
                df_limpio[f'{col_texto}_length'] = df_limpio[f'{col_texto}_clean'].str.len()
                df_limpio[f'{col_texto}_word_count'] = df_limpio[f'{col_texto}_clean'].str.split().str.len()
                print(f"   {col_texto}: texto limpiado y features creadas")

        # Escalado de variables numéricas
        
        print("\n" + "="*50)
        print("NUEVO: Aplicando ESCALADO a variables numéricas...")
        print("="*50)
        
        # Identificar columnas numéricas (excluyendo IDs y target)
        columnas_no_escalar = ['product_id', 'user_id', 'review_id', 'rating']
        columnas_a_escalar = [
            col for col in df_limpio.columns 
            if df_limpio[col].dtype in ['float64', 'int64'] 
            and col not in columnas_no_escalar
            and not col.startswith('Unnamed')
        ]
        
        print(f"Columnas a escalar ({len(columnas_a_escalar)}):")
        for col in columnas_a_escalar[:10]:  # Mostrar primeras 10
            print(f"  • {col}")
        
        # Aplicar StandardScaler
        self.scaler = StandardScaler()
        df_limpio[columnas_a_escalar] = df_limpio[columnas_a_escalar].fillna(0)
        df_limpio[columnas_a_escalar] = self.scaler.fit_transform(df_limpio[columnas_a_escalar])
        print(f"\n✓ Escalado completado con StandardScaler")
        print(f"  Media ≈ 0, Desviación estándar ≈ 1 para columnas numéricas")

         
        # Codificación de variables categóricas
        
        print("\n" + "="*50)
        print("NUEVO: Aplicando CODIFICACIÓN a variables categóricas...")
        print("="*50)
        
        # Identificar columnas categóricas
        columnas_a_codificar = [
            col for col in df_limpio.columns 
            if df_limpio[col].dtype  'object' 
            and df_limpio[col].nunique() < 50
            and col not in ['review_content', 'review_title', 'about_product', 
                           'product_name', 'user_name', 'img_link', 'product_link']
        ]
        
        if columnas_a_codificar:
            print(f"Columnas a codificar ({len(columnas_a_codificar)}):")
            for col in columnas_a_codificar:
                print(f"  • {col} ({df_limpio[col].nunique()} categorías)")
            
            # Aplicar One-Hot Encoding
            df_limpio = pd.get_dummies(df_limpio, columns=columnas_a_codificar, drop_first=True)
            print(f"\n✓ Codificación One-Hot completada")
            print(f"  Nuevas dimensiones: {df_limpio.shape}")
        else:
            print("  No se encontraron columnas categóricas para codificar")

        # 6. Manejar valores nulos restantes
        nulos_despues = df_limpio.isnull().sum().sum()
        if nulos_despues > 0:
            print(f"\nImputando {nulos_despues} valores nulos restantes...")
            for col in df_limpio.columns:
                if df_limpio[col].dtype in ['float64', 'int64']:
                    df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
                else:
                    df_limpio[col] = df_limpio[col].fillna('No disponible')
            print(f"  ✓ Valores nulos imputados")

        self.df = df_limpio
        self.reporte['duplicados_eliminados'] = duplicados_antes - duplicados_despues
        self.reporte['shape_final'] = self.df.shape
        self.reporte['columnas_escaladas'] = len(columnas_a_escalar)
        self.reporte['columnas_codificadas'] = len(columnas_a_codificar) if columnas_a_codificar else 0

        print(f"\n✓ Dataset limpio. Dimensiones finales: {self.df.shape}")
        return self

    @staticmethod
    def _limpiar_texto(texto):
        """Método estático para limpieza de texto."""
        if pd.isna(texto) or texto  '':
            return ''
        texto = str(texto).lower()
        texto = re.sub(r'[^a-zA-Z\s]', '', texto)
        texto = re.sub(r'\s+', ' ', texto).strip()
        return texto

    def generar_reporte(self):
        """Genera reporte final del proceso de normalización."""
        print("\nAGENTE NORMALIZADOR - Reporte Final")
        print("=" * 50)
        print(f"Dataset original : {self.reporte['shape_original']}")
        print(f"Dataset limpio   : {self.reporte['shape_final']}")
        print(f"Duplicados eliminados: {self.reporte['duplicados_eliminados']}")
        
        # NUEVO: Reporte de transformaciones
        print(f"\nTransformaciones aplicadas:")
        print(f"   • Escalado (StandardScaler): {self.reporte.get('columnas_escaladas', 'N/A')} columnas")
        print(f"   • Codificación (One-Hot): {self.reporte.get('columnas_codificadas', 'N/A')} columnas")
        print(f"   • Features de texto creadas: longitud y conteo de palabras")

        if self.columnas_faltantes:
            print(f"  Columnas faltantes: {len(self.columnas_faltantes)}")
        else:
            print(" Columnas esperadas: todas presentes")

        print(f"\nTipos de variables:")
        print(f"   • Numéricas  : {self.reporte['tipos_variables']['numericas']}")
        print(f"   • Categóricas: {self.reporte['tipos_variables']['categoricas']}")
        print(f"   • Texto      : {self.reporte['tipos_variables']['texto']}")

        return self

    def obtener_dataset(self):
        """Retorna el DataFrame limpio."""
        return self.df

    def obtener_dataset_original(self):
        """NUEVO: Retorna el DataFrame original para comparación."""
        return self.df_original


print("Clase AgenteNormalizador definida")

# Ejecución del Agente Normalizador

print("INICIANDO SISTEMA DE AGENTES DE IA")
print("=" * 60)

normalizador = AgenteNormalizador()
normalizador.cargar_dataset()
normalizador.validar_columnas() 
normalizador.exploracion_inicial()
normalizador.limpiar_dataset()
normalizador.generar_reporte()

df_limpio = normalizador.obtener_dataset()
print("\n Agente Normalizador completado. Dataset listo para Agente Entrenador.")

# CELDA 5: DIAGNOSTICO DE COLUMNAS
# (Ejecutar para verificar que el dataset es el correcto)

print("COLUMNAS DEL DATASET LIMPIO:")
print("=" * 55)
for i, col in enumerate(df_limpio.columns, 1):
    n_nulos = df_limpio[col].isnull().sum()
    print(
        f"{i:3d}. {col:<35} "
        f"tipo={str(df_limpio[col].dtype):<10} "
        f"únicos={df_limpio[col].nunique():<6} "
        f"nulos={n_nulos}"
    )

print(f"\nShape: {df_limpio.shape}")
print("\nPrimeras 3 filas:")
display(df_limpio.head(3))

# Verificacion critica: confirmar que rating tiene valores validos
if 'rating' in df_limpio.columns:
    print(f"\n Columna 'rating' encontrada.")
    print(f"  Rango   : [{df_limpio['rating'].min()}, {df_limpio['rating'].max()}]")
    print(f"  Media   : {df_limpio['rating'].mean():.2f}")
    print(f"  Nulos   : {df_limpio['rating'].isnull().sum()}")
else:
    print("\n ERROR: Columna 'rating' NO encontrada. Revisar carga del dataset.")

# ============================================
# CELDA 6: AGENTE 2 - ENTRENADOR
# Entrenamiento con división train/validation/test
# ============================================

class AgenteEntrenador:
    """Agente de entrenamiento con división train/validation/test."""

    COLUMNAS_EXCLUIR = {
        'product_id', 'user_id', 'review_id', 'id', 'Id',
        'ProductId', 'UserId', 'img_link', 'product_link',
        'user_name', 'product_name', 'review_title', 'review_content',
        'about_product', 'review_content_clean', 'review_title_clean',
        'about_product_clean', 'category'
    }

    TARGET_CANDIDATES = ['rating', 'Score']

    def __init__(self, df):
        self.df = df.copy()
        self.mejor_modelo = None
        self.mejor_nombre = None
        self.metricas = {}
        self.resultados_modelos = {}
        self.caracteristicas_importantes = None
        self.features = []
        self.target = None
        self.X = None
        self.y = None
        # NUEVO: Guardar splits
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.insights = []

    def _detectar_target(self, df):
        for candidato in self.TARGET_CANDIDATES:
            if candidato in df.columns:
                return candidato
        raise ValueError(f"No se encontró target. Buscadas: {self.TARGET_CANDIDATES}")
    
    def analisis_exploratorio(self):
        """EDA rápido sin gráficos pesados."""
        print("\nAGENTE ENTRENADOR - Fase 1: Análisis Exploratorio")
        print("-" * 50)

        self.target = self._detectar_target(self.df)
        print(f"Target detectado: '{self.target}'")
        print(f"  Media: {self.df[self.target].mean():.2f}")
        print(f"  Mediana: {self.df[self.target].median():.2f}")
        print(f"  Rango: [{self.df[self.target].min()}, {self.df[self.target].max()}]")

        # Gráfico simple
        plt.figure(figsize=(8, 4))
        self.df[self.target].value_counts().sort_index().plot(
            kind='bar',
            color='steelblue',
            edgecolor='black'
        )
        plt.title(f'Distribución de {self.target}')
        plt.tight_layout()
        plt.show()

        return self

    def preparar_datos(self):
        """Preparación optimizada - SOLO features numéricas esenciales."""
        print("\nAGENTE ENTRENADOR - Fase 2: Preparación de Datos")
        print("-" * 50)

        df_modelo = self.df.copy()
        self.target = self._detectar_target(df_modelo)

        if self.target == 'Score' and 'rating' not in df_modelo.columns:
            df_modelo['rating'] = df_modelo['Score']
            self.target = 'rating'

        # Seleccionar SOLO features numéricas
        self.features = [
            col for col in df_modelo.columns
            if df_modelo[col].dtype in ['float64', 'int64']
            and col not in self.COLUMNAS_EXCLUIR
            and col != self.target
            and col != 'Score'
            and not col.startswith('Unnamed')
        ]

        # LIMITAR a 20 features máximo
        if len(self.features) > 20:
            correlaciones = df_modelo[self.features + [self.target]].corr()[self.target].abs()
            correlaciones = correlaciones.drop(self.target).sort_values(ascending=False)
            self.features = correlaciones.head(20).index.tolist()
            print("  Features limitadas a las 20 más relevantes")

        print(f"Features seleccionadas: {len(self.features)}")

        # Preparar X e y
        cols_relevantes = self.features + [self.target]
        df_modelo = df_modelo.dropna(subset=cols_relevantes)

        self.X = df_modelo[self.features]
        self.y = df_modelo[self.target]

        print(f"Shape X: {self.X.shape}")
        print(f"Muestras totales: {len(self.X)}")

        return self

    def entrenar_modelo(self):
        """
        NUEVO: Entrenamiento con división train/validation/test
        """
        print("\nAGENTE ENTRENADOR - Fase 3: Entrenamiento")
        print("-" * 50)

        # ============================================
        # NUEVO: Split en train/validation/test (60/20/20)
        # ============================================
        print("NUEVO: Dividiendo en train/validation/test (60/20/20)...")

        # Primer split: separar test set
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Segundo split: separar validation del resto
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 del total
        )

        print(f"Split completado:")
        print(f"  • Train     : {len(self.X_train)} muestras")
        print(f"  • Validation: {len(self.X_val)} muestras")
        print(f"  • Test      : {len(self.X_test)} muestras")

        # MODELOS LIGEROS
        modelos = {
            'Regresión Lineal': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=30,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                random_state=42
            )
        }

        import time

        for nombre, modelo in modelos.items():
            print(f" {nombre}...", end=' ')
            inicio = time.time()

            modelo.fit(self.X_train, self.y_train)

            # NUEVO: Evaluar en validation set también
            y_pred_val = modelo.predict(self.X_val)
            y_pred_val = np.clip(y_pred_val, 1.0, 5.0)

            y_pred_test = modelo.predict(self.X_test)
            y_pred_test = np.clip(y_pred_test, 1.0, 5.0)

            # Métricas en test
            r2 = r2_score(self.y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))

            # NUEVO: Métricas en validation
            r2_val = r2_score(self.y_val, y_pred_val)
            rmse_val = np.sqrt(mean_squared_error(self.y_val, y_pred_val))

            # CV con solo 2 folds
            cv_scores = cross_val_score(modelo, self.X, self.y, cv=2, scoring='r2')

            tiempo = time.time() - inicio

            self.resultados_modelos[nombre] = {
                'modelo': modelo,
                'r2_test': r2,
                'rmse_test': rmse,
                'r2_val': r2_val,        # NUEVO
                'rmse_val': rmse_val,    # NUEVO
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }

            print(f" R²(test)={r2:.4f} | R²(val)={r2_val:.4f} | RMSE={rmse:.4f} | {tiempo:.1f}s")

        # Seleccionar mejor
        self.mejor_nombre = max(self.resultados_modelos,
                                key=lambda x: self.resultados_modelos[x]['r2_test'])
        self.mejor_modelo = self.resultados_modelos[self.mejor_nombre]['modelo']

        print(f"\nMEJOR MODELO: {self.mejor_nombre}")
        print(f"  R² Score (Test): {self.resultados_modelos[self.mejor_nombre]['r2_test']:.4f}")

        #Celda 7 - Ejecución del Agente Entrenador

        entrenador = AgenteEntrenador(df_limpio)
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
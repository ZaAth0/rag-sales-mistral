#Sistema de Agentes de IA - Reviews de Amazon 
#Instalación de dependencias necesarias para el proyecto
!pip install kagglehub pandas numpy matplotlib seaborn scikit-learn nltk wordcloud

#Instalación de librerías
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

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
            print(f"⚠️  Columnas esperadas NO encontradas ({len(self.columnas_faltantes)}):")
            for col in self.columnas_faltantes:
                print(f"   - {col}")
        else:
            print("✓ Todas las columnas esperadas están presentes.")

        if self.columnas_extra:
            print(f"\n📋 Columnas adicionales encontradas ({len(self.columnas_extra)}):")
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


print("Clase AgenteNormalizador definida (CORREGIDA con Escalado y Codificación).")

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
print("\n✓ Agente Normalizador completado. Dataset listo para Agente Entrenador.")



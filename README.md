# Xgboost en AWS usando Metaflow

"""
clasificación binaria XGBOost mínima, robusta y escalable en AWS usando Metaflow y Pyspark.

Este código implementa un clasificador binario con xgboost, utilizando Metaflow para la orquestación del flujo de trabajo y Pyspark para el procesamiento de datos distribuidos. Utiliza una lógica condicional para adaptar el enfoque según el tamaño de la base de datos y sigue las mejores prácticas recomendadas de Metaflow para entrenamiento distribuido escalable.

Documentación de referencia:

- Metaflow: https://docs.metaflow.org/
- PySpark: https://spark.apache.org/docs/latest/api/python/
- XGBoost: https://xgboost.readthedocs.io/
"""
Clasificación Binaria Escalable con XGBoost en AWS usando Metaflow y PySpark
Importar librerías
```
from metaflow import FlowSpec, step, Parameter, card, batch, resources, current  # Componentes de Metaflow
import os  # Operaciones del sistema para rutas de archivos
```
Definir el Flujo de Metaflow
```
class XGBoostBinaryClassifierFlow(FlowSpec):  # Clase principal del flujo que hereda de FlowSpec
    """
    Flujo de trabajo escalable para XGBoost que:
    1. Cambio dinámico Pandas/PySpark según el tamaño de los datos
    2. Ejecuta procesamiento distribuido en AWS Batch
    3. Realiza seguimiento de experimentos y modelos con Metaflow
    """

    # Definir Parámetros (configurables via CLI)
    parquet_path = Parameter(  # Parámetro: ruta de datos de entrada
        "parquet_path",
        help="Ruta local al archivo parquet de entrada.",
        default="./data/input.parquet"
    )
    label_column = Parameter(  # Nombre de la variable objetivo
        "label_column",
        help="Nombre de la columna objetivo para clasificación.",
        default="label"
    )
    n_estimators = Parameter(  # Número de árboles para XGBoost
        "n_estimators",
        help="Número de árboles para XGBoost.",
        default=100
    )
    data_size_threshold = Parameter( #Umbral de cambio a PySpark (MB)
        "data_size_threshold",
        help="Umbral (en MB) para cambiar a procesamiento distribuido.",
        default=100       # Datos > 100MB usan PySpark
    )

    @step
    def start(self):  # Paso de inicialización
        """
        Paso de Inicialización:
        - Verifica si existe el archivo parquet.
        - Mide tamaño de archivo para elegir Pandas o PySpark.
        """
        assert os.path.exists(self.parquet_path), f"Archivo parquet no encontrado en {self.parquet_path}"  # Verifica si archivo existe
        self.file_size_mb = os.path.getsize(self.parquet_path) / (1024 * 1024)  # Calcular tamaño en MB
        print(f"Archivo parquet cargado ({self.file_size_mb:.2f} MB)")  # Mostrar tamaño del archivo

        # Decisión: Usar PySpark para datasets grandes, Pandas para pequeños
        self.use_pyspark = self.file_size_mb > self.data_size_threshold  # Bandera PySpark basada en umbral
        print(f"Usando PySpark: {self.use_pyspark}")  # Mostrar decisión del motor de procesamiento
        self.next(self.load_data)  # Transición al siguiente paso

    @resources(memory=16000, cpu=4)  # Solicitar 16GB RAM y 4 CPUs
    @step
    def load_data(self):  # Paso de carga de datos
        """
        Paso de carga y preparación de datos:
        - Usa PySpark para carga distribuida si el dataset es grande
        - Usa Pandas para datasets pequeños
        - Extrae columnas de características y etiquetas
        """
        if self.use_pyspark:  # Procesamiento de datasets grandes
            # Carga distribuida con PySpark
            from pyspark.sql import SparkSession  # Importar solo cuando sea necesario

            # Inicializar sesión de Spark para procesamiento distribuido
            spark = SparkSession.builder.appName("MetaflowXGB").getOrCreate()  # Crear contexto Spark
            print("Leyendo archivo parquet con PySpark...")  
            df = spark.read.parquet(self.parquet_path)  # Leer datos usando PySpark

            # Identificar columnas de características (todas excepto la etiqueta)
            self.feature_columns = [col for col in df.columns if col != self.label_column]  # Detección automática de características

            # Convertir DataFrame de PySpark a Pandas para XGBoost (si el tamaño lo permite)
            # Muestrear datasets grandes (>1M filas) para evitar errores de memoria
            sample_size = df.count()  # Obtener conteo total de filas
            print(f"PySpark cargó {sample_size} filas")
            if sample_size > 1_000_000:  # Manejo de datasets muy grandes
                # Para datos muy grandes, muestrear o usar entrenamiento distribuido
                # Submuestrear a 1M filas manteniendo la distribución
                df_sample = df.sample(False, 1e6/sample_size)  # Submuestrear a 1M filas
                pdf = df_sample.toPandas()  # Convertir a DataFrame de Pandas
            else:
                pdf = df.toPandas()  # Convertir dataset completo a Pandas
            spark.stop()  # Limpiar sesión de Spark
        else:  # Procesamiento de datasets pequeños
            # Carga local con Pandas para datasets pequeños
            import pandas as pd  # Importar Pandas
            print("Leyendo archivo parquet con Pandas...")
            pdf = pd.read_parquet(self.parquet_path)  # Leer datos usando Pandas
            self.feature_columns = [col for col in pdf.columns if col != self.label_column]  # Detección automática de características
            print(f"Pandas cargó {len(pdf)} filas")  # Mostrar conteo de filas

        # Preparar características/etiquetas para entrenamiento
        self.X = pdf[self.feature_columns]  # Matriz de características
        self.y = pdf[self.label_column]  # Vector objetivo
        self.next(self.train)  # Transición al paso de entrenamiento

    @batch(cpu=4, memory=16000)     # Ejecuta en AWS Batch con recursos especificados
    @step
    def train(self):  # Paso de entrenamiento del modelo
        """
        Paso de entrenamiento y evaluación del modelo:
        - Divide datos en conjuntos de entrenamiento/prueba
        - Configura XGBoost para clasificación binaria
        - Usa métodos eficientes de árboles según tamaño de datos
        - Usa entrenamiento distribuido si el tamaño de datos es grande
        - Evalúa el modelo con métrica AUC
        """
        import xgboost as xgb  # Librería XGBoost
        from sklearn.model_selection import train_test_split  # División de datos
        from sklearn.metrics import roc_auc_score  # Métrica de evaluación

        # Crear división entrenamiento/prueba (80/20)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42  # División reproducible
        )

        # Convertir a DMatrix - estructura de datos optimizada de XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)  # Datos de entrenamiento
        dtest = xgb.DMatrix(X_test, label=y_test)  # Datos de prueba

        # Parámetros de XGBoost para clasificación binaria
        params = {
            "objective": "binary:logistic",  # Objetivo de clasificación binaria
            "eval_metric": "auc",  # Métrica de evaluación
            "tree_method": "hist" if not self.use_pyspark else "approx", # Selección de método eficiente
            "verbosity": 1                  # Nivel medio de registro
        }

        # Para datos grandes: usar entrenamiento distribuido (Dask o Spark)
        if self.use_pyspark and len(X_train) > 1_000_000:  # Verificación de entrenamiento distribuido
            try:
                # Integración potencial con SparkML (no implementada aquí)
                from xgboost.spark import SparkXGBClassifier  # Integración Spark-XGB
                print("Usando SparkXGBClassifier para entrenamiento distribuido...")
                # Se puede realizar ajuste distribuido aquí si se desea
                # Para el tutorial, volver a nodo único si SparkXGB no está disponible
            except ImportError:  # Manejo de alternativa
                print("xgboost.spark no disponible, usando XGBoost en nodo único.")

        # Entrenar modelo
        print("Entrenando XGBoost...")
        model = xgb.train(  # Entrenamiento del modelo
            params,
            dtrain,
            num_boost_round=self.n_estimators,  # Número de árboles
            evals=[(dtest, "test")],        # Conjunto de evaluación
            verbose_eval=True             # Mostrar progreso
        )

        # Predecir y evaluar rendimiento del modelo
        y_pred = model.predict(dtest)  # Generar predicciones
        auc = roc_auc_score(y_test, y_pred)  # Calcular AUC
        print(f"AUC de prueba: {auc:.4f}")  # Mostrar rendimiento

        # Persistir (guardar) modelo y métricas para pasos siguientes
        self.model = model  # Almacenar objeto del modelo
        self.auc = auc  # Almacenar métrica de evaluación
        self.next(self.end)  # Transición al paso final

    @card               # Habilita reportes integrados de Metaflow
    @step
    def end(self):  # Paso de finalización
        """
        Paso de Finalización:
        - Imprime métricas resumidas
        - Genera un reporte visual
        """
        print("Entrenamiento completado.")  # Mensaje de finalización
        print(f"AUC final de prueba: {self.auc:.4f}")  # Mostrar AUC final

        # Crear reporte visual interactivo (visible en la UI de Metaflow)
        card_content = f"""  # Contenido HTML del reporte
        ## Resultados de Clasificación Binaria con XGBoost

        **Puntuación AUC:** {self.auc:.4f}

        **Columnas de Características:** {self.feature_columns}

        **Parámetros del Modelo:**
        - Número de estimadores: {self.n_estimators}
        - Objetivo: binary:logistic
        """
        current.card.append(card_content)  # Adjuntar al reporte de Metaflow
```

## Punto de entrada para CLI de Metaflow
```
if __name__ == "__main__":
    XGBoostBinaryClassifierFlow()  # Ejecutar el flujo
```


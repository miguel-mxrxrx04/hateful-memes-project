Creación de dataset.py



Fases del desarrollo:

Fase 1: Baselines Unimodales. Establecer el suelo de rendimiento. No puedes evaluar la fusión si no sabes cómo rinden las modalidades por separado.	

1. Entrenar clasificador solo-texto usando los tensores offline de DistilBERT pre-extraídos.

2. Entrenar clasificador solo-imagen (ej. ResNet50 preentrenado) congelando las capas iniciales.
	Sección de Resultados: Tabla comparativa base. Demuestra la dificultad intrínseca del dataset cuando falta contexto.


Fase 2: Fusión Tardía. Construir la red multimodal base.	

1. Unir ambos branches concatenando los embeddings finales.

2. Entrenar la red conjunta monitorizando el AUROC.

3. Guardar los pesos del modelo (.pt).
	Metodología: Diagrama de la arquitectura (texto offline + visión online). Justificación técnica del diseño para no exceder los 8GB VRAM.


Fase 3: Auditoría Cuantitativa. Demostrar el colapso unimodal. Aquí está el núcleo científico.	

1. Zero-ing: Evaluar el modelo multimodal apagando el texto (tensor de ceros) y midiendo la caída de métricas.

2. Repetir apagando la imagen.
	Discusión Clave: Gráfico de barras mostrando la asimetría en la caída de rendimiento. Evidencia cuantitativa del sesgo hacia el texto.


Fase 4: Explicabilidad Visual. Proporcionar la prueba cualitativa del colapso.	

1. Implementar GradCAM sobre la última capa convolucional de ResNet50.

2. Generar heatmaps para casos donde el modelo acierta y donde falla (falsos positivos por sesgo de texto).
	Análisis Cualitativo: Rejilla de imágenes mostrando cómo el modelo ignora los estímulos visuales clave del meme cuando el texto es altamente polarizante.


Fase 5: Redacción. Empaquetar los hallazgos en formato de artículo científico (IEEE/ACM).	

1. Consolidar el repositorio en GitHub (limpio, modular).

2. Extraer gráficos de los notebooks.

3. Redactar conclusiones basadas puramente en los datos empíricos.
	Entregable Final: Un paper denso, reproducible, con una narrativa lógica clara y defendible ante tribunal.
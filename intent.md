Pregunta de investigación:

¿Las arquitecturas de fusión tardía en tareas de clasificación multimodal (Hateful Memes) desarrollan una verdadera sinergia inter-modal, o sufren "colapso unimodal" hacia la semántica del texto que resulta cuantificable mediante ablación y evidenciable visualmente mediante GradCAM?

Quiero ver qué pesos se han activado mas para saber donde falla la multimodalidad de la red con fusión tardia, comparando rendimientos con zero ing para la red de texto y la de imagen. Despues, buscar mejorar los resultados con ponderaciones obligatorias (mejoras sencillas para conseguir multimodalidad)

Consideraciones sobre el proyecto:

- No queremos batir el benchmark del dataset con arquitecturas masivas, sino entender cómo falla la fusión.

- Alnpreprocesar el texto con DistilBERT de forma offline, eliminamos la computación del loop de train. Me permite iterar rápidamente en la rama visual y mecanismo de fusión

Aislamiento de la variabilidad: Al pre-procesar el texto con DistilBERT de forma offline, eliminas toda esa computación del loop de entrenamiento. Esto te permite iterar rápidamente en la rama visual y en el mecanismo de fusión, que es donde verdaderamente reside el valor de tu investigación.

Justificación de GradCAM: No lo incluyes "porque queda bonito", sino como herramienta de auditoría forense. Si los números (Zero-ing en la Fase 3) dicen que la imagen aporta poco, GradCAM debe demostrar (en la Fase 4) que los mapas de activación visual son difusos, ruidosos o se centran en el texto del meme en lugar de en el contenido de la imagen.
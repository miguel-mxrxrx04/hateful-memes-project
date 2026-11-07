# Multimodal Hate Speech Detection & Unimodal Collapse Audit

Proyecto académico para la evaluación y auditoría empírica del sesgo modal (*Text Dominance / Modality Collapse*) en arquitecturas de fusión tardía utilizando el dataset **Hateful Memes** de Meta AI.

## Estructura del Proyecto

```text
├── data/
│   ├── raw/          # Imágenes (.png) y metadatos (.jsonl)
│   ├── features/     # Embeddings pre-extraídos de texto (.pt)
│   └── processed/    # Datos procesados intermedios
├── models/
│   └── weights/      # Checkpoints de modelos entrenados
├── notebooks/        # Prototipado y análisis exploratorio
├── paper/            # Documento y figuras en formato IEEE
└── src/
    ├── data/         # Dataset y extracción de características
    ├── models/       # Arquitecturas (unimodales y fusión)
    ├── training/     # Bucles de entrenamiento y evaluación
    └── utils/        # Métricas y explicabilidad (GradCAM)
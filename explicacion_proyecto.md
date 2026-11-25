# Explicación del Proyecto y Bitácora Técnica: Moderación Multimodal Auditada

## 1. Visión General y Tesis de Ingeniería
Este proyecto aborda el problema del **aprendizaje de atajos unimodales (*Multimodal Shortcut Learning*)** en la detección de discurso de odio sobre memes (*Hateful Memes*). En lugar de competir mediante modelos fundacionales masivos inviables en producción por coste y latencia, se diseña una **arquitectura ligera, explicable y auditada** capaz de:

1. **Detectar cuándo la rama textual intenta monopolizar la decisión** (*Modality Collapse*) ante pares contrafactuales (*benign confounders*).
2. **Mitigar la dominancia léxica** mediante fusión bilineal, atención cruzada espacial con inyección de coordenadas (`CoordConv2d`), regularización por *Modality Dropout* y supervisión auxiliar multi-tarea.
3. **Abstenerse selectivamente (*Human-in-the-Loop*)** mediante un enrutador diagnóstico que combina incertidumbre epistémica (MC-Dropout), sensibilidad visual contrafactual ($\Delta_{\text{vis}}$) y mapas de activación espacial (Grad-CAM), exponiendo toda la trazabilidad a través de una API REST.

---

## 2. Arquitectura del Pipeline (End-to-End)

### Fase A: Ingeniería de Datos y Validación Contrafactual (`src/data/`)
* **Dataset:** Facebook Hateful Memes (8.500 muestras en `train`, 500 en `dev`, 1.000 en `test`).
* **Aislamiento de *Benign Confounders*:** Extracción programática en `dev` de pares mínimos que comparten texto idéntico o imagen base con etiquetas opuestas (`0` frente a `1`). Este subconjunto actúa como prueba de estrés para medir la tasa real de falsos positivos (FPR) por atajos léxicos.
* **Extracción Offline (`fp16`):**
  * **Texto:** Embeddings de 768 dimensiones extraídos con `DistilBERT` (`{id}.pt`).
  * **Visión:** Mapas convolucionales espaciales `[2048, 7, 7]` extraídos de `layer4` de `ResNet-50` sin *Global Average Pooling* (`{id}_img.pt`), preservando la topología espacial para atención cruzada y Grad-CAM.

### Fase B: Escalera Experimental de Fusión (`src/models/fusion_zoo.py`)
Todas las variantes comparten proyección latente ($d = 256$), inyección espacial `CoordConv2d` $(x, y) \in [-1, 1]$ y cabezales auxiliares unimodales ($z_T, z_V$) entrenados bajo pérdida multi-tarea ($\mathcal{L}_{\text{joint}} + 0.2(\mathcal{L}_T + \mathcal{L}_V)$):

1. **`text_only` (Baseline Unimodal Texto):** DistilBERT + MLP. Fija el techo léxico.
2. **`image_only` (Baseline Unimodal Visión):** ResNet-50 + CoordConv + MLP.
3. **`concat` (Fusión Tardía Ingenua):** Concatenación $[\mathbf{v} \,\|\, \mathbf{t}]$. Demuestra empíricamente el colapso hacia el texto y el disparo de falsos positivos en *confounders*.
4. **`hadamard` (Interacción Bilineal Factorizada):** Vector $[\mathbf{v} \,\|\, \mathbf{t} \,\|\, \mathbf{v} \odot \mathbf{t}]$ inspirado en *Hate-CLIPper*. Actúa como puerta lógica AND diferenciable.
5. **`cross_attn` (Atención Cruzada Espacial + Modality Dropout):** El texto interroga los 49 tokens espaciales ($7 \times 7$) de la imagen mientras *Modality Dropout* ($p=0.25$) enmascara estocásticamente el texto en entrenamiento para forzar dependencia visual.

*Todos los checkpoints óptimos según AUROC en validación se serializan en `models/weights/{arch}_best.pt`.*

### Fase C: Auditoría Espacial y Router Selectivo HITL (`src/utils/`)
* **Grad-CAM Determinista (`gradcam_engine.py`):** Implementado mediante *hooks* nativos de PyTorch en bloques `try...finally` para evitar fugas de memoria VRAM y cuantificar la masa de activación espacial.
* **Enrutador Diagnóstico (`selective_router.py`):** Evalúa en inferencia:
  1. **Incertidumbre Epistémica ($\sigma^2_{\text{MC}}$):** Varianza tras $K=5$ pasadas con `Dropout` activo.
  2. **Test Contrafactual Visual ($\Delta_{\text{vis}}$):** Caída de probabilidad al sustituir la imagen por un tensor nulo ($|p(\mathbf{v}, \mathbf{t}) - p(\mathbf{0}, \mathbf{t})|$).
  3. **Gobernanza:** Si $\sigma^2_{\text{MC}} > \tau_{\text{var}}$ o si predice odio con $\Delta_{\text{vis}} < \tau_{\text{vis}}$ (`UNIMODAL_TEXT_SHORTCUT_DETECTED`), deriva a revisión humana (`FLAG_FOR_HUMAN_REVIEW`).

### Fase D: Despliegue Trazable y Pruebas (`src/api/app.py` & `tests/`)
* **Microservicio FastAPI + Pydantic:** Endpoint `POST /moderate` con validación estricta de tensores de entrada, medición de latencia (`ms`) y pico de memoria (`MB`), persistiendo cada decisión y sus señales diagnósticas en `logs/audit_trail.jsonl`.
* **Suite `pytest` (`tests/test_safety_gate.py`):** Verifica ausencia de fugas de VRAM en Grad-CAM, bloqueo de atajos unimodales ante imágenes nulas y cumplimiento del contrato JSON de la API.

---

## 3. Bitácora de Ejecución y Resultados

### Sprint 0: Estructura, Extracción Offline y Smoke Test
* **Entorno:** Ubuntu | Portátil (4 GB VRAM, desarrollo/testing) y Torre (8 GB VRAM, entrenamiento).
* **Estado:**
  * [x] Jerarquía de carpetas y `.gitignore` blindados.
  * [x] Módulos `fusion_zoo.py`, `selective_router.py`, `gradcam_engine.py`, `run_experiments.py` y `app.py` implementados.
  * [ ] Extracción completa de tensores `fp16` en torre (`data/features/`).
  * [ ] Ejecución de ablación completa (`results/metrics/ablation_summary.json`).

### Registro de Métricas Comparativas (Pendiente de volcado desde Torre)
| Arquitectura | AUROC Global (Dev) | FPR Benign Confounders ($\downarrow$) | Unimodal Shortcut Gap ($\uparrow$) |
| :--- | :---: | :---: | :---: |
| `text_only` (DistilBERT) | -- | -- | 0.0000 |
| `image_only` (ResNet-50) | -- | -- | -- |
| `concat` (Baseline Multimodal) | -- | -- | -- |
| `hadamard` (Bilinear AND) | -- | -- | -- |
| `cross_attn` (+ CoordConv + ModDrop) | -- | -- | -- |

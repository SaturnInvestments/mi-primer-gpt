# 🤖 Mi Primer GPT — Entrena tu propia IA con los clásicos de la literatura

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaturnInvestments/mi-primer-gpt/blob/main/notebooks/Mi_primer_GPT.ipynb)
[![Licencia MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)

> *"No necesitas un superordenador para entender cómo funciona ChatGPT. Solo necesitas curiosidad, un texto clásico y Google Colab."*

Este repositorio te guiará paso a paso en la construcción de tu propio **modelo de lenguaje autorregresivo desde cero**. No es una simulación ni el consumo de una API comercial: es **la misma arquitectura Transformer (*Decoder-Only*)** que sustenta a GPT-3, GPT-4 y Claude, pero reducida a escala didáctica para que puedas entrenarla gratis en tu navegador sin instalar nada en tu computadora.

Este proyecto forma parte del laboratorio práctico complementario a la **Sección 8: Modelos de Lenguaje Masivos — El Salto Exponencial** del libro *Historia de la Inteligencia Artificial y el Dataverso*.

---

## 🎯 ¿Qué vas a construir?

Vas a construir un modelo de lenguaje que aprende a **predecir el siguiente carácter (*Next-Token Prediction*)** basándose en el contexto previo. Al finalizar el entrenamiento:
1. Verás cómo la pérdida de error (*loss*) desciende en una gráfica interactiva.
2. Experimentarás con el parámetro de **Temperatura** para modular la creatividad de la red.
3. Podrás **chatear en tiempo real con la IA en la consola**, viendo cómo adopta la sintaxis y el vocabulario de Cervantes o Dumas.

---

## 📚 Catálogo de Corpus Literarios

Todos los textos incluidos pertenecen al **dominio público**:

| Corpus | Autor | Tamaño | Características y Estilo |
| :--- | :--- | :--- | :--- |
| **Don Quijote de la Mancha** | Miguel de Cervantes Saavedra | ~2.1 MB | Español del Siglo de Oro, vocabulario arcaico, diálogos caballerescos. |
| **El Conde de Montecristo** | Alexandre Dumas | ~1.5 MB | Aventura decimonónica, intriga, drama y diálogos dinámicos. |
| **Los Tres Mosqueteros** | Alexandre Dumas | ~1.3 MB | Acción, novela de capa y espada, diálogos de camaradería y honor. |
| **Texto de Prueba Rápida** | Sintético | ~1.5 KB | Muestra ultracorta para verificar el funcionamiento en menos de 1 minuto. |
| **Tu Propio Texto** | Tú | Libre | Sube cualquier archivo `.txt` (cuentos, ensayos, poemas) desde el notebook. |

> 💡 **Experimentación multicorpus:** En el notebook puedes activar varias casillas a la vez para combinar obras (por ejemplo, toda la saga de Alexandre Dumas), sumando millones de caracteres para enriquecer el vocabulario del modelo.

---

## 🚀 ¿Cómo funciona el Pipeline?

```
Texto en Español (Corpus)
          │
          ▼
Tokenización por Caracteres  ──> [ stoi / itos ]  ──> Tensores PyTorch
          │
          ▼
Arquitectura Transformer Decoder:
  • Token & Positional Embeddings
  • N Bloques Transformer (Multi-Head Causal Self-Attention + FeedForward GELU + LayerNorm)
  • Cabeza lineal de proyección al vocabulario
          │
          ▼
Entrenamiento con AdamW (Minimización de Cross-Entropy Loss)
          │
          ▼
Muestreo Autorregresivo con Temperatura  ──>  Chat Interactivo con la IA
```

---

## ⏱️ Tiempos Estimados de Entrenamiento

| Modo / Configuración | Tiempo en GPU Colab (T4 Gratuita) | Tiempo en CPU | Resultado Esperado |
| :--- | :--- | :--- | :--- |
| **Prueba rápida (`prueba.txt`)** | ~30 segundos | ~2 minutos | El modelo aprende a repetir patrones básicos. |
| **Modelo Estándar (3.000 iters, 4 capas)** | ~6 a 9 minutos | ~25 a 35 minutos | Palabras coherentes en español y estructura gramatical reconocible. |
| **Modelo Profundo (5.000 iters, 6 capas)** | ~15 a 20 minutos | ~1.5 horas | Cadencia estilística muy marcada del autor elegido. |

---

## 📂 Estructura del Repositorio

```text
mi-primer-gpt/
├── README.md                      # Presentación general y guía (este archivo)
├── requirements.txt               # Dependencias para ejecución local
├── LICENSE                        # Licencia MIT (código abierto)
├── notebooks/
│   └── Mi_primer_GPT.ipynb        # Cuaderno interactivo con formularios de Colab
├── corpus/
│   ├── README.md                  # Descripción de las obras y licencias
│   ├── prueba.txt                 # Texto de prueba rápida
│   ├── quijote.txt                # Don Quijote de la Mancha completo y limpio
│   ├── montecristo.txt            # El Conde de Montecristo limpio
│   ├── tres_mosqueteros.txt       # Los Tres Mosqueteros limpio
│   └── preparar_corpus.py         # Script para descargar y limpiar automáticamente obras
└── scripts/
    └── generar_notebook.py        # Generador del notebook interactivo
```

---

## 🏁 ¿Cómo empezar?

### Opción 1: En Google Colab (Recomendada, no requiere instalación)

1. Haz clic en el botón oficial:  
   [![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SaturnInvestments/mi-primer-gpt/blob/main/notebooks/Mi_primer_GPT.ipynb)
2. En el menú superior de Colab, ve a: **Entorno de ejecución $\rightarrow$ Cambiar tipo de entorno $\rightarrow$ T4 GPU**.
3. Ejecuta las celdas en orden secuencial con `Shift + Enter`.
4. ¡Elige tus obras favoritas, ajusta los controles interactivos y conversa con tu modelo!

### Opción 2: Ejecución Local

Si prefieres ejecutarlo en tu propia máquina con Python y Jupyter:

```bash
# 1. Clonar el repositorio
git clone https://github.com/SaturnInvestments/mi-primer-gpt.git
cd mi-primer-gpt

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Descargar y preparar los corpus (si deseas actualizar o añadir más)
python corpus/preparar_corpus.py

# 4. Iniciar Jupyter Notebook
jupyter notebook notebooks/Mi_primer_GPT.ipynb
```

---

## 👨‍🏫 Para Docentes y Estudiantes

* **Para Estudiantes:** Cada celda de código está profusamente comentada en español y precedida por explicaciones teóricas conceptuales, sin obviar el rigor matemático pero accesible para quien aprende por primera vez.
* **Para Docentes:** El notebook sirve como laboratorio ideal para sesiones de 1 a 2 horas sobre NLP, Transformers y redes neuronales profundas. Se pueden proponer ejercicios como:
  * Comparar la perplejidad y pérdida al entrenar con Cervantes frente a Dumas.
  * Experimentar con la longitud de contexto (`block_size`) y observar el impacto en memoria.
  * Analizar la entropía de salida variando la temperatura entre 0.2 y 1.5.

---

## 📖 Lecturas Recomendadas

* **Capítulo 3 / Sección 8** de *Historia de la Inteligencia Artificial y el Dataverso*.
* Vaswani et al. (2017): [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762).
* Andrej Karpathy: [nanoGPT](https://github.com/karpathy/nanoGPT).

---

## 📄 Licencia

Este proyecto está bajo la Licencia **MIT**. Puedes usarlo, modificarlo y distribuirlo libremente con fines educativos y de investigación.

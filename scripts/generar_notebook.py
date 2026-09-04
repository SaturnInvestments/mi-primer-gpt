"""Script generador del Notebook interactivo 'Mi_primer_GPT.ipynb'.

Crea un archivo Jupyter Notebook (.ipynb) compatible con Google Colab y JupyterLab,
con formularios interactivos (# @param), explicaciones pedagógicas detalladas y
código optimizado en PyTorch.
"""

import json
import os
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

NOTEBOOK_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")
NOTEBOOK_PATH = os.path.join(NOTEBOOK_DIR, "Mi_primer_GPT.ipynb")

def build_notebook():
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # -------------------------------------------------------------
    # PORTADA Y PRESENTACIÓN
    # -------------------------------------------------------------
    add_md("""
# 🤖 Mi Primer GPT — Entrena tu propia IA desde cero con los Clásicos Literarios

> *"No es magia. Es álgebra lineal, atención y el Siglo de Oro español."*

Bienvenido a **Mi Primer GPT**. En este laboratorio práctico vas a construir y entrenar un **modelo de lenguaje autorregresivo** (exactamente la misma arquitectura *Transformer Decoder-only* que impulsa a GPT-3, GPT-4 y Claude), pero optimizado y reducido a escala didáctica para que puedas entrenarlo en minutos en tu navegador usando la **GPU gratuita de Google Colab**.

Este proyecto acompaña a la **Sección 8: Modelos de Lenguaje Masivos — El Salto Exponencial** del libro *Historia de la Inteligencia Artificial y el Dataverso*.

---

### 🎯 Objetivos de Aprendizaje
1. **Tokenización:** Comprender cómo el texto se transforma en representaciones numéricas procesables por una red neuronal.
2. **Mecanismo de Auto-Atención Causal (*Causal Self-Attention*):** Aprender cómo cada posición calcula afinidades con el contexto pasado sin mirar al futuro.
3. **Bloques Transformer:** Construir capas residuales con normalización (*LayerNorm*) y redes *Feed-Forward*.
4. **Entrenamiento Autorregresivo:** Minimizar la pérdida de entropía cruzada prediciendo el siguiente carácter (*Next-Token Prediction*).
5. **Inferencia y Creatividad:** Generar texto muestreando probabilidades moduladas por **Temperatura**.
6. **Chat Interactivo:** Conversar en tiempo real con una IA que adopta el estilo de Cervantes o Dumas.
""")

    # -------------------------------------------------------------
    # CELDA 1: CONFIGURACIÓN INICIAL Y GPU
    # -------------------------------------------------------------
    add_md("""
---
## ⚙️ 1. Configuración de Hardware y Entorno

Ejecuta esta celda para verificar si tienes una **GPU aceleradora** asignada en Google Colab y configurar el entorno.
""")

    add_code("""# @title ⚙️ 1. Configuración de Hardware y Repositorio { run: "auto" }
import os
import sys
import requests
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configurar dispositivo (GPU CUDA o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 60)
print(f"🖥️ Dispositivo en uso: {device.type.upper()}")
if device.type == "cuda":
    print(f"🚀 GPU Asignada: {torch.cuda.get_device_name(0)}")
    print(f"⚡ Memoria VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ Estás usando CPU. El entrenamiento será más lento.")
    print("💡 Consejo: En Colab, ve a: Entorno de ejecución -> Cambiar tipo de entorno -> T4 GPU")
print("=" * 60)

# Semillas para garantizar reproducibilidad
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Si estamos en Google Colab y no existe la carpeta corpus, clonar el repositorio
REPO_URL = "https://github.com/SaturnInvestments/mi-primer-gpt.git"
if not os.path.exists("corpus"):
    print("📦 Entorno detectado sin carpeta 'corpus'. Descargando catálogo...")
    # Intentar clonar si git está disponible
    exit_code = os.system(f"git clone {REPO_URL} repo_temp")
    if exit_code == 0 and os.path.exists("repo_temp/corpus"):
        os.system("cp -r repo_temp/corpus .")
        os.system("rm -rf repo_temp")
        print("✅ Carpeta 'corpus' lista desde el repositorio.")
    else:
        os.makedirs("corpus", exist_ok=True)
        print("📁 Carpeta 'corpus' inicializada localmente.")
""")

    # -------------------------------------------------------------
    # CELDA 2: SELECTOR INTERACTIVO DE CORPUS
    # -------------------------------------------------------------
    add_md("""
---
## 📚 2. Selección y Carga del Corpus Literario

Puedes seleccionar **uno o varios libros** para combinarlos. Si combinas varios títulos del mismo autor (por ejemplo, *El Conde de Montecristo* y *Los Tres Mosqueteros*), el modelo tendrá más datos para aprender el estilo y enriquecer su vocabulario.

También puedes activar la casilla para **subir tu propio archivo `.txt`**.
""")

    add_code("""# @title 📚 2. Selector de Obras Literarias { run: "auto" }
# @markdown Selecciona las obras que deseas incluir en el entrenamiento:

incluir_quijote = True # @param {type:"boolean"}
incluir_montecristo = False # @param {type:"boolean"}
incluir_tres_mosqueteros = False # @param {type:"boolean"}
incluir_prueba_rapida = False # @param {type:"boolean"}
subir_archivo_personalizado = False # @param {type:"boolean"}

# URLs de respaldo en GitHub (para descarga directa si no están locales)
RAW_URLS = {
    "quijote": "https://raw.githubusercontent.com/SaturnInvestments/mi-primer-gpt/main/corpus/quijote.txt",
    "montecristo": "https://raw.githubusercontent.com/SaturnInvestments/mi-primer-gpt/main/corpus/montecristo.txt",
    "tres_mosqueteros": "https://raw.githubusercontent.com/SaturnInvestments/mi-primer-gpt/main/corpus/tres_mosqueteros.txt",
    "prueba": "https://raw.githubusercontent.com/SaturnInvestments/mi-primer-gpt/main/corpus/prueba.txt"
}

def obtener_texto(nombre_archivo, url_respaldo):
    ruta_local = os.path.join("corpus", nombre_archivo)
    if os.path.exists(ruta_local):
        with open(ruta_local, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    # Si no existe localmente, descargarlo de GitHub o Project Gutenberg
    print(f"📥 Descargando {nombre_archivo}...")
    try:
        r = requests.get(url_respaldo, timeout=20)
        if r.status_code == 200:
            os.makedirs("corpus", exist_ok=True)
            with open(ruta_local, "w", encoding="utf-8") as f:
                f.write(r.text)
            return r.text
    except Exception as e:
        print(f"⚠️ No se pudo descargar {nombre_archivo}: {e}")
    return ""

textos_combinados = []
obras_seleccionadas = []

if incluir_quijote:
    t = obtener_texto("quijote.txt", RAW_URLS["quijote"])
    if t:
        textos_combinados.append(t)
        obras_seleccionadas.append("Don Quijote de la Mancha")

if incluir_montecristo:
    t = obtener_texto("montecristo.txt", RAW_URLS["montecristo"])
    if t:
        textos_combinados.append(t)
        obras_seleccionadas.append("El Conde de Montecristo")

if incluir_tres_mosqueteros:
    t = obtener_texto("tres_mosqueteros.txt", RAW_URLS["tres_mosqueteros"])
    if t:
        textos_combinados.append(t)
        obras_seleccionadas.append("Los Tres Mosqueteros")

if incluir_prueba_rapida:
    t = obtener_texto("prueba.txt", RAW_URLS["prueba"])
    if not t:
        # Texto de respaldo embebido
        t = "En un lugar de la Mancha, de cuyo nombre no quiero acordarme, no ha mucho tiempo que vivía un hidalgo de los de lanza en astillero, adarga antigua, rocín flaco y galgo corredor. " * 30
    textos_combinados.append(t)
    obras_seleccionadas.append("Texto de prueba rápida")

if subir_archivo_personalizado:
    try:
        from google.colab import files
        print("📁 Sube tu archivo de texto (.txt):")
        uploaded = files.upload()
        for fn in uploaded.keys():
            t_user = uploaded[fn].decode("utf-8", errors="replace")
            textos_combinados.append(t_user)
            obras_seleccionadas.append(f"Archivo personalizado: {fn}")
    except ImportError:
        print("ℹ️ Subida interactiva solo disponible en Google Colab. Para uso local coloca tu archivo en 'corpus/'.")

# Unir todos los textos
texto_completo = "\\n\\n".join(textos_combinados)

# Limpieza básica: caracteres imprimibles y saltos de línea
texto_completo = ''.join(c for c in texto_completo if c.isprintable() or c == '\\n')

if len(texto_completo) == 0:
    raise ValueError("❌ No has seleccionado ningún corpus. Por favor activa al menos una opción.")

print("\\n" + "=" * 60)
print(f"📖 Obras cargadas ({len(obras_seleccionadas)}): {', '.join(obras_seleccionadas)}")
print(f"📏 Longitud total del corpus: {len(texto_completo):,} caracteres")
print(f"📝 Muestra inicial:")
print("-" * 60)
print(texto_completo[:300].strip())
print("...")
print("=" * 60)
""")

    # -------------------------------------------------------------
    # CELDA 3: TOKENIZACIÓN A NIVEL DE CARÁCTER
    # -------------------------------------------------------------
    add_md("""
---
## 🔤 3. Tokenización a Nivel de Carácter

Las redes neuronales no entienden letras ni palabras directamente: solo comprenden **números**.

A diferencia de los modelos comerciales como GPT-4 (que agrupan subpalabras mediante algoritmos BPE), en este laboratorio utilizaremos **tokenización por caracteres**:
1. Construimos el **vocabulario** con todos los caracteres únicos presentes en las obras elegidas.
2. Creamos dos diccionarios:
   - `stoi`: String to Integer (convierte carácter $\\rightarrow$ número).
   - `itos`: Integer to String (convierte número $\\rightarrow$ carácter).
3. Transformamos todo el texto en un gran tensor unidimensional de números enteros (`torch.long`).
""")

    add_code("""# @title 🔤 3. Construcción del Vocabulario y Tokenizador
# Vocabulario de caracteres únicos ordenados
chars = sorted(list(set(texto_completo)))
vocab_size = len(chars)

print(f"🔤 Tamaño del vocabulario: {vocab_size} caracteres únicos")
print(f"🔡 Caracteres presentes en el vocabulario:\\n{' '.join(chars)}")

# Mapeos bidireccionales
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str) -> list:
    \"\"\"Convierte una cadena de texto en una lista de IDs numéricos.\"\"\"
    return [stoi[c] for c in s if c in stoi]

def decode(indices: list) -> str:
    \"\"\"Convierte una lista de IDs numéricos de vuelta a texto.\"\"\"
    return ''.join([itos[i] for i in indices])

# Demostración del tokenizador
frase_demo = "En un lugar de la Mancha"
tokens_demo = encode(frase_demo)
reconstruida = decode(tokens_demo)
print("\\n🧪 Demostración del tokenizador:")
print(f"   Original:      '{frase_demo}'")
print(f"   Tokenizada:    {tokens_demo[:15]}...")
print(f"   Reconstruida:  '{reconstruida}'")

# Convertir todo el texto en un tensor de PyTorch
data = torch.tensor(encode(texto_completo), dtype=torch.long)
print(f"\\n📊 Tensor de datos creado: {data.shape[0]:,} tokens listos para entrenamiento.")
""")

    # -------------------------------------------------------------
    # CELDA 4: DIVISIÓN TRAIN / VALIDATION
    # -------------------------------------------------------------
    add_md("""
---
## 📊 4. Partición de Datos: Entrenamiento vs. Validación

Para comprobar si nuestro modelo está **aprendiendo a generalizar el estilo** o simplemente **memorizando** el texto (*overfitting*), dividimos los datos:
* **90% para Entrenamiento (`train_data`):** El modelo ve estos datos y ajusta sus pesos con retropropagación.
* **10% para Validación (`val_data`):** Datos que el modelo jamás ve durante el entrenamiento. Calculamos el error en este conjunto para verificar su capacidad real de predicción.
""")

    add_code("""# @title 📊 4. Partición Train / Val
n_split = int(0.9 * len(data))
train_data = data[:n_split]
val_data = data[n_split:]

print(f"📚 Datos de Entrenamiento (90%): {len(train_data):,} tokens")
print(f"📖 Datos de Validación   (10%): {len(val_data):,} tokens")
""")

    # -------------------------------------------------------------
    # CELDA 5: HIPERPARÁMETROS DEL MODELO
    # -------------------------------------------------------------
    add_md("""
---
## 📐 5. Arquitectura e Hiperparámetros

Configuramos las dimensiones del Transformer. Puedes experimentar modificando estos valores desde el formulario:
* `block_size` (Contexto): Cuántos caracteres pasados mira el modelo para predecir el siguiente.
* `n_embd`: Dimensión del vector latente que representa cada carácter y posición.
* `n_head`: Número de cabezas de atención en paralelo (permite atender simultáneamente a sintaxis, concordancia, rima, etc.).
* `n_layer`: Número de bloques Transformer apilados en profundidad.
* `dropout`: Porcentaje de neuronas desactivadas aleatoriamente para regularización.
""")

    add_code("""# @title 📐 5. Configurar Hiperparámetros { run: "auto" }
block_size = 128      # @param {type:"slider", min:32, max:256, step:32}
n_embd = 128          # @param {type:"slider", min:64, max:256, step:32}
n_head = 4            # @param [2, 4, 8] {type:"raw"}
n_layer = 4           # @param {type:"slider", min:2, max:8, step:1}
dropout = 0.1         # @param {type:"slider", min:0.0, max:0.3, step:0.05}

assert n_embd % n_head == 0, "n_embd debe ser divisible por n_head"

print("📐 Hiperparámetros configurados:")
print(f"   - Contexto temporal (block_size): {block_size} caracteres")
print(f"   - Dimensión de embeddings (n_embd): {n_embd}")
print(f"   - Cabezas de atención (n_head): {n_head} ({n_embd // n_head} dims/cabeza)")
print(f"   - Capas Transformer (n_layer): {n_layer}")
print(f"   - Tasa de Dropout: {dropout}")
""")

    # -------------------------------------------------------------
    # CELDA 6: IMPLEMENTACIÓN DEL TRANSFORMER EN PYTORCH
    # -------------------------------------------------------------
    add_md("""
---
## 🧠 6. El Modelo Transformer (Decoder-Only)

Aquí implementamos la arquitectura matemática inspirada en *"Attention Is All You Need"* (Vaswani et al., 2017) y *nanoGPT* de Andrej Karpathy:

```
[Entrada de Tokens] ---> [Token + Position Embeddings]
                                │
                        ┌───────▼────────┐
                        │   LayerNorm    │
                        │  Multi-Head    │  <--- Máscara Causal (Tril)
                        │ Self-Attention │
                        │  + Residual    │
                        ├────────────────┤
                        │   LayerNorm    │  x n_layer
                        │  FeedForward   │
                        │  + Residual    │
                        └───────┬────────┘
                                │
                        [LayerNorm Final]
                                │
                          [Head Lineal] ---> [Logits del siguiente carácter]
```
""")

    add_code("""# @title 🧠 6. Definición de la Red Neuronal Transformer

class CausalSelfAttention(nn.Module):
    \"\"\"Atención Multi-Cabeza Causal: Solo atiende a tokens anteriores en el tiempo.\"\"\"
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Proyecciones lineales para Query, Key y Value
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Proyección de salida
        self.c_proj = nn.Linear(n_embd, n_embd)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Máscara causal: matriz triangular inferior para impedir ver el futuro
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)
        )

    def forward(self, x):
        B, T, C = x.shape  # Batch, Secuencia (Tiempo), Canales (Embedding)
        
        # Calcular Q, K, V en una sola multiplicación matricial
        qkv = self.c_attn(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Separar en múltiples cabezas: (B, n_head, T, head_dim)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Producto punto escalado de atención: Q @ K^T / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / np.sqrt(self.head_dim))
        
        # Aplicar máscara causal (-infinito a las posiciones futuras)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Ponderar Values con las afinidades de atención
        y = att @ v  # (B, n_head, T, head_dim)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # Reensamblar cabezas
        
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    \"\"\"Red densa de 2 capas con activación no lineal GELU.\"\"\"
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    \"\"\"Bloque Transformer: LayerNorm -> Self-Attention + Residual -> LayerNorm -> MLP + Residual.\"\"\"
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))  # Conexión residual 1
        x = x + self.mlp(self.ln2(x))   # Conexión residual 2
        return x


class MiniGPT(nn.Module):
    \"\"\"Modelo de Lenguaje Autorregresivo Completo.\"\"\"
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout):
        super().__init__()
        self.block_size = block_size
        
        # Tabla de embeddings para tokens y posiciones
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        
        # Bloques Transformer apilados
        self.blocks = nn.Sequential(*[
            TransformerBlock(n_embd, n_head, block_size, dropout)
            for _ in range(n_layer)
        ])
        
        # Capa final de normalización y proyección a vocabulario
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.block_size, f"Secuencia {T} excede el contexto máximo {self.block_size}"
        
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.pos_embedding(torch.arange(T, device=idx.device))  # (T, n_embd)
        x = self.drop(tok_emb + pos_emb)
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        \"\"\"Genera nuevos caracteres autorregresivamente uno por uno.\"\"\"
        for _ in range(max_new_tokens):
            # Recortar al contexto máximo permitido
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Obtener predicciones
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)
            
            # Convertir logits en distribución de probabilidades
            probs = F.softmax(logits, dim=-1)
            
            # Muestrear el siguiente token
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx

# Instanciar modelo
model = MiniGPT(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    dropout=dropout
).to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"🧠 Modelo MiniGPT creado con éxito.")
print(f"📊 Total de parámetros entrenables: {num_params:,}")
print(f"🔍 Comparación: GPT-3 tiene 175,000,000,000 parámetros ({num_params/175e9:.6%} de su escala).")
""")

    # -------------------------------------------------------------
    # CELDA 7: BATCHES Y ENTRENAMIENTO
    # -------------------------------------------------------------
    add_md("""
---
## 🎯 7. Entrenamiento del Modelo

El modelo aprenderá de forma **autónoma**:
1. Extrae un bloque de texto aleatorio de `block_size` caracteres.
2. Intenta predecir el carácter siguiente en cada posición.
3. Se compara con la verdad terreno mediante la función de pérdida de **Entropía Cruzada**.
4. El optimizador **AdamW** ajusta los pesos de la red para reducir el error.
""")

    add_code("""# @title 🎯 7. Ejecutar Entrenamiento { run: "auto" }
# @markdown Configura las iteraciones y el tamaño de lote:

max_iters = 3000       # @param {type:"slider", min:500, max:8000, step:500}
batch_size = 64        # @param [32, 64, 128] {type:"raw"}
learning_rate = 3e-4   # @param [1e-4, 3e-4, 5e-4, 1e-3] {type:"raw"}
eval_interval = 250    # @param {type:"integer"}
eval_iters = 50        # @param {type:"integer"}

def get_batch(split):
    dataset = train_data if split == 'train' else val_data
    ix = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i:i+block_size] for i in ix])
    y = torch.stack([dataset[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
iter_checkpoints = []

print("=" * 65)
print(f"🚀 Iniciando entrenamiento por {max_iters} iteraciones...")
print("=" * 65)

for iter_num in range(max_iters):
    # Evaluar periódicamente
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        train_losses.append(losses['train'])
        val_losses.append(losses['val'])
        iter_checkpoints.append(iter_num)
        print(f"Iteración {iter_num:5d} / {max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f}")
        
    # Paso de optimización
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("=" * 65)
print("✅ ¡Entrenamiento finalizado con éxito!")
print("=" * 65)
""")

    # -------------------------------------------------------------
    # CELDA 8: VISUALIZACIÓN DE CURVA DE PÉRDIDA
    # -------------------------------------------------------------
    add_md("""
---
## 📈 8. Curva de Aprendizaje

Esta gráfica ilustra cómo disminuye el error del modelo a lo largo del tiempo.
* Si ambas curvas descienden conjuntamente, el modelo está **generalizando**.
* Si la curva de validación se estanca o empieza a subir mientras la de entrenamiento baja, hay **sobreajuste** (*overfitting*).
""")

    add_code("""# @title 📈 8. Graficar Curva de Pérdida
plt.figure(figsize=(10, 5), dpi=120)
plt.plot(iter_checkpoints, train_losses, label='Train Loss (Pérdida Entrenamiento)', color='#1f77b4', lw=2)
plt.plot(iter_checkpoints, val_losses, label='Val Loss (Pérdida Validación)', color='#ff7f0e', lw=2, linestyle='--')
plt.xlabel('Iteración de Entrenamiento', fontsize=12)
plt.ylabel('Pérdida (Cross-Entropy Loss)', fontsize=12)
plt.title('Curva de Aprendizaje de MiniGPT', fontsize=14, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.show()
""")

    # -------------------------------------------------------------
    # CELDA 9: GENERACIÓN Y TEMPERATURA
    # -------------------------------------------------------------
    add_md("""
---
## ✍️ 9. Generación de Texto y Control de Temperatura

La **Temperatura** modula qué tan "arriesgado" o "creativo" es el modelo al muestrear el siguiente carácter:
* **Temperatura baja ($T < 0.7$):** El modelo selecciona casi siempre las letras más probables. Es coherente pero puede volverse repetitivo o monótono.
* **Temperatura alta ($T > 1.2$):** Aumenta la entropía y la variedad léxica, pero puede inventar palabras inexistentes o cometer errores ortográficos.
* **Temperatura equilibrada ($0.8 - 1.0$):** Balance ideal entre cohesión y creatividad.
""")

    add_code("""# @title ✍️ 9. Generador Interactivo con Temperatura { run: "auto" }
prompt_inicial = "En un lugar de la Mancha" # @param {type:"string"}
caracteres_a_generar = 400 # @param {type:"slider", min:100, max:1000, step:50}
temperatura = 0.85 # @param {type:"slider", min:0.2, max:1.6, step:0.05}

def generar_texto(prompt: str, max_tokens: int, temp: float) -> str:
    model.eval()
    context_tokens = encode(prompt)
    if len(context_tokens) == 0:
        context_tokens = [0]
    idx = torch.tensor([context_tokens], dtype=torch.long, device=device)
    out_idx = model.generate(idx, max_new_tokens=max_tokens, temperature=temp)
    return decode(out_idx[0].tolist())

resultado = generar_texto(prompt_inicial, caracteres_a_generar, temperatura)

print("=" * 65)
print(f"📝 Prompt inicial: \"{prompt_inicial}\"")
print(f"🎛️ Temperatura:    {temperatura}")
print("-" * 65)
print("🤖 Generación del MiniGPT:")
print("-" * 65)
print(resultado)
print("=" * 65)
""")

    # -------------------------------------------------------------
    # CELDA 10: CHAT INTERACTIVO
    # -------------------------------------------------------------
    add_md("""
---
## 💬 10. Chat Interactivo en Consola

¡Es hora de conversar con tu modelo! Escribe una frase o una pregunta y la IA continuará la conversación respondiendo en el estilo literario aprendido.

> 💡 **Nota pedagógica sobre la memoria:**  
> Como nuestro `block_size` es de 128 o 256 caracteres, la "memoria" activa del modelo corresponde a esa cantidad de caracteres pasados. ¡Exactamente igual a cómo los modelos comerciales tienen ventanas de contexto (8k, 32k, 128k)!
""")

    add_code("""# @title 💬 10. Iniciar Chat Interactivo
def chat_con_ia():
    print("=" * 65)
    print("💬 ¡Bienvenido al Chat con tu IA Literaria!")
    print(f"   Estilo aprendido: {', '.join(obras_seleccionadas)}")
    print("   Escribe 'salir' o presiona Enter vacío para terminar.")
    print("=" * 65 + "\\n")
    
    historial = ""
    while True:
        try:
            user_msg = input("👤 Tú: ").strip()
        except EOFError:
            break
        if user_msg.lower() in ["salir", "exit", "quit", ""]:
            print("\\n👋 ¡Hasta la próxima aventura literaria!")
            break
            
        historial += f"\\nUsuario: {user_msg}\\nIA: "
        # Mantener historial dentro de los últimos caracteres para no sobrepasar el contexto
        prompt_actual = historial[-block_size:]
        
        idx = torch.tensor([encode(prompt_actual)], dtype=torch.long, device=device)
        out_idx = model.generate(idx, max_new_tokens=150, temperature=0.85)
        texto_generado = decode(out_idx[0].tolist())
        
        # Extraer únicamente la respuesta generada
        respuesta = texto_generado[len(prompt_actual):].split("\\n")[0].strip()
        if not respuesta:
            respuesta = texto_generado[len(prompt_actual):][:80].strip()
            
        print(f"🤖 IA: {respuesta}\\n")
        historial += respuesta

# Para iniciar la conversación, desmarca el comentario en la siguiente línea:
chat_con_ia()
""")

    # -------------------------------------------------------------
    # CELDA 11: GUARDAR Y DESCARGAR MODELO
    # -------------------------------------------------------------
    add_md("""
---
## 💾 11. Guardar y Descargar los Pesos del Modelo

No necesitas volver a entrenar tu modelo desde cero cada vez. Puedes guardar los pesos aprendidos (`.pt`) y descargarlos en tu computadora o sincronizarlos con Google Drive.
""")

    add_code("""# @title 💾 11. Guardar y Descargar Checkpoint { run: "auto" }
nombre_archivo_modelo = "mi_primer_gpt_pesos.pt" # @param {type:"string"}
descargar_a_disco = False # @param {type:"boolean"}

# Guardar state_dict de PyTorch
torch.save({
    'model_state_dict': model.state_dict(),
    'chars': chars,
    'stoi': stoi,
    'itos': itos,
    'hyperparameters': {
        'block_size': block_size,
        'n_embd': n_embd,
        'n_head': n_head,
        'n_layer': n_layer,
        'vocab_size': vocab_size
    }
}, nombre_archivo_modelo)

peso_mb = os.path.getsize(nombre_archivo_modelo) / (1024 * 1024)
print(f"💾 Checkpoint guardado exitosamente: '{nombre_archivo_modelo}' ({peso_mb:.2f} MB)")

if descargar_a_disco:
    try:
        from google.colab import files
        print("⬇️ Iniciando descarga en tu navegador...")
        files.download(nombre_archivo_modelo)
    except ImportError:
        print(f"ℹ️ Archivo disponible localmente en: {os.path.abspath(nombre_archivo_modelo)}")
""")

    # -------------------------------------------------------------
    # RESUMEN Y CIERRE PEDAGÓGICO
    # -------------------------------------------------------------
    add_md("""
---
## 🎓 Resumen: ¿Qué has aprendido?

1. **La anatomía de un LLM:** Has programado manualmente cada matriz de atención ($Q, K, V$), la máscara causal triangular, las capas de proyección densas y la normalización.
2. **El poder de la predicción autoregresiva:** Una red neuronal no necesita entender la literatura para hablar como Cervantes; simplemente minimiza la entropía cruzada prediciendo la siguiente letra miles de veces.
3. **El efecto de la escala:** Viste cómo una red con ~800.000 parámetros logra estructurar palabras y frases. Ahora puedes dimensionar lo que ocurre cuando se escala a 175.000 millones de parámetros como en GPT-3 o billones como en los modelos frontera actuales.

📖 **Regresar a la lectura:** *Historia de la Inteligencia Artificial y el Dataverso* — Sección 8.
""")

    notebook_data = {
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }

    os.makedirs(NOTEBOOK_DIR, exist_ok=True)
    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(notebook_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Notebook generado con éxito: {NOTEBOOK_PATH}")
    print(f"   Total de celdas: {len(cells)}")

if __name__ == "__main__":
    build_notebook()

# 📚 Corpus Literarios para Mi Primer GPT

Este directorio contiene y gestiona los textos clásicos de la literatura en español utilizados para entrenar los modelos de lenguaje.

Todos los textos seleccionados pertenecen al **dominio público**.

---

## 📖 Catálogo de Obras

| Archivo | Obra | Autor | Año | Tamaño estimado | Estilo y Descripción |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `prueba.txt` | Muestra de prueba | — | — | ~1.5 KB (~1.500 caracteres) | Texto sintético rápido para verificar que el pipeline funcione sin esperar el entrenamiento largo. |
| `quijote.txt` | *Don Quijote de la Mancha* (Completo) | Miguel de Cervantes Saavedra | 1605 / 1615 | ~2.1 MB (~2.1M caracteres) | Español del Siglo de Oro. Sintaxis clásica, diálogos caballerescos y reflexiones filosóficas. |
| `montecristo.txt` | *El Conde de Montecristo* | Alexandre Dumas | 1844 | ~1.5 MB (~1.5M caracteres) | Novela de aventuras y drama del siglo XIX. Diálogos ágiles, intriga y suspense. |
| `tres_mosqueteros.txt` | *Los Tres Mosqueteros* | Alexandre Dumas | 1844 | ~1.3 MB (~1.3M caracteres) | Clásico de capa y espada. Interacciones de camaradería, acción y honor. |

---

## ⚙️ Descarga y Preparación Automática

Para garantizar que los modelos aprendan exclusivamente el estilo del autor (y no licencias legales ni encabezados en inglés), incluimos un script de descarga y limpieza automática:

```bash
python corpus/preparar_corpus.py
```

Este script:
1. Descarga el texto original de fuentes de dominio público (como Project Gutenberg).
2. Detecta y remueve automáticamente los encabezados legales en inglés (`*** START OF THE PROJECT GUTENBERG EBOOK... ***`) y notas de digitalización.
3. Normaliza los saltos de línea y codificación a **UTF-8 limpio**.
4. Almacena el resultado listo para entrenar en la carpeta `corpus/`.

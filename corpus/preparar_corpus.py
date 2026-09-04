"""Script para descargar y limpiar corpus literarios en dominio público.

Descarga obras clásicas en español de Miguel de Cervantes y Alexandre Dumas,
elimina prefacios, notas editoriales y licencias, y guarda el texto en UTF-8 puro
listo para entrenar el modelo Transformer.

Utiliza únicamente la biblioteca estándar de Python (urllib.request, re, os)
para no requerir dependencias externas al ejecutarse.
"""

import os
import re
import sys
import urllib.request

# Asegurar codificación UTF-8 en terminales Windows
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

CORPUS_DIR = os.path.dirname(os.path.abspath(__file__))

# Catálogo de fuentes públicas en ESPAÑOL
FUENTES = {
    "quijote": {
        "archivo": "quijote.txt",
        "nombre": "Don Quijote de la Mancha (Miguel de Cervantes - Español)",
        "url": "https://www.gutenberg.org/cache/epub/2000/pg2000.txt",
        "marca_inicio": r"\*\*\* START OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        "marca_fin": r"\*\*\* END OF TH(E|IS) PROJECT GUTENBERG EBOOK.*?\*\*\*",
    },
    "montecristo": {
        "archivo": "montecristo.txt",
        "nombre": "El Conde de Montecristo (Alexandre Dumas - Español)",
        "url": "https://archive.org/download/el-conde-de-monte-cristo-de-alexandre-dumas-pdf/EL%20CONDE%20DE%20MONTE%20CRISTO%20DE%20ALEXANDRE%20DUMAS%20PDF_djvu.txt",
        "marca_inicio": r"PRIMERA PARTE",
        "marca_fin": None,
    },
    "tres_mosqueteros": {
        "archivo": "tres_mosqueteros.txt",
        "nombre": "Los Tres Mosqueteros (Alexandre Dumas - Español)",
        "url": "https://archive.org/download/AlejandroDumasLosTresMosqueteros/Alejandro%20Dumas%20-%20Los%20tres%20mosqueteros_djvu.txt",
        "marca_inicio": r"I\.\s+Los tres presentes del se[ñn]or D'Artagnan padre",
        "marca_fin": None,
    },
}


def limpiar_texto(raw_text: str, marca_inicio: str = None, marca_fin: str = None) -> str:
    """Elimina metadatos, licencias y normaliza espacios en blanco."""
    texto = raw_text

    # Si hay marcas de inicio, extraer solo el interior
    if marca_inicio:
        match_inicio = re.search(marca_inicio, texto, re.IGNORECASE)
        if match_inicio:
            texto = texto[match_inicio.start():]

    if marca_fin:
        match_fin = re.search(marca_fin, texto, re.IGNORECASE)
        if match_fin:
            texto = texto[:match_fin.start()]

    # Eliminar posibles retornos de carro Windows/Mac y caracteres nulos
    texto = texto.replace('\r\n', '\n').replace('\r', '\n')
    
    # Mantener caracteres imprimibles y saltos de línea
    texto = ''.join(c for c in texto if c.isprintable() or c == '\n')

    # Reducir secuencias excesivas de saltos de línea a un máximo de dos
    texto = re.sub(r'\n{4,}', '\n\n\n', texto)

    return texto.strip()


def descargar_y_preparar(clave: str):
    """Descarga, limpia y guarda una obra en el directorio corpus."""
    if clave not in FUENTES:
        print(f"❌ Clave no reconocida: '{clave}'. Opciones válidas: {list(FUENTES.keys())}")
        return

    info = FUENTES[clave]
    destino = os.path.join(CORPUS_DIR, info["archivo"])

    print(f"\n📥 Descargando: {info['nombre']}...")
    try:
        req = urllib.request.Request(
            info["url"],
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=60) as response:
            raw_bytes = response.read()

        # Decodificar detectando UTF-8 o latin-1 como fallback
        try:
            raw_text = raw_bytes.decode('utf-8')
        except UnicodeDecodeError:
            raw_text = raw_bytes.decode('latin-1')

        # Corregir si el archivo tiene caracteres doblemente codificados (mojibake)
        if "Ã­" in raw_text or "Ã±" in raw_text or "Ã¡" in raw_text:
            try:
                raw_text = raw_text.encode('latin-1').decode('utf-8')
            except Exception:
                pass

        print(f"🧹 Limpiando metadatos y normalizando...")
        texto_limpio = limpiar_texto(
            raw_text,
            marca_inicio=info.get("marca_inicio"),
            marca_fin=info.get("marca_fin")
        )

        with open(destino, 'w', encoding='utf-8') as f:
            f.write(texto_limpio)

        print(f"✅ Guardado con éxito en español en: {destino}")
        tam_mb = os.path.getsize(destino) / (1024 * 1024)
        print(f"   📏 Longitud: {len(texto_limpio):,} caracteres ({tam_mb:.2f} MB)")

    except Exception as e:
        print(f"❌ Error al procesar {clave}: {e}")


def main():
    print("=" * 60)
    print("🚀 Preparador de Corpus Literarios en Español para Mi Primer GPT")
    print("=" * 60)

    args = sys.argv[1:]
    objetivos = args if args else list(FUENTES.keys())

    for obj in objetivos:
        descargar_y_preparar(obj)

    print("\n🎉 Proceso finalizado. Corpus en español listos en la carpeta 'corpus/'.\n")


if __name__ == "__main__":
    main()


from pathlib import Path
import spacy

ARCHIVO_ENTRADA = Path("orgullo_y_prejuicio.txt")
ARCHIVO_SALIDA = Path("texto_limpio.txt")


def limpiar_texto():
    if not ARCHIVO_ENTRADA.exists():
        print("ERROR: No se encontro orgullo_y_prejuicio.txt")
        print("Coloca el libro en la carpeta principal del proyecto.")
        return

    nlp = spacy.load("es_core_news_sm")

    texto = ARCHIVO_ENTRADA.read_text(encoding="utf-8-sig")

    if not texto.strip():
        print("ERROR: El archivo del libro esta vacio.")
        return

    documento = nlp(texto)
    palabras_limpias = []

    for token in documento:
        if token.is_alpha and not token.is_stop:
            lema = token.lemma_.lower().strip()

            if lema:
                palabras_limpias.append(lema)

    resultado = " ".join(palabras_limpias)
    ARCHIVO_SALIDA.write_text(resultado, encoding="utf-8")

    print("Limpieza terminada correctamente.")
    print(f"Tokens analizados: {len(documento)}")
    print(f"Palabras conservadas: {len(palabras_limpias)}")
    print("\nPrimeras 30 palabras:")
    print(" ".join(palabras_limpias[:30]))
    print(f"\nArchivo generado: {ARCHIVO_SALIDA}")


if __name__ == "__main__":
    limpiar_texto()

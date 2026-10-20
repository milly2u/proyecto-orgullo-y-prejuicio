
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

# 1. Cargar el texto procesado
RUTA_TEXTO = Path("texto_limpio.txt")

if not RUTA_TEXTO.exists():
    raise FileNotFoundError(
        "No se encontró texto_limpio.txt. "
        "Ejecuta primero limpieza_texto.py."
    )

texto = RUTA_TEXTO.read_text(encoding="utf-8").lower()

# 2. Preparar las oraciones para Word2Vec
# Separamos por líneas y conservamos palabras con contenido.
oraciones = []

for linea in texto.splitlines():
    palabras = [
        palabra.strip(".,;:¡!¿?()[]{}\"'—-")
        for palabra in linea.split()
    ]
    palabras = [p for p in palabras if p.isalpha()]

    if len(palabras) >= 2:
        oraciones.append(palabras)

if not oraciones:
    raise ValueError(
        "No hay suficientes palabras en texto_limpio.txt. "
        "Revisa el archivo generado por la limpieza."
    )

print(f"Oraciones utilizadas: {len(oraciones)}")

# 3. Entrenar Word2Vec
modelo = Word2Vec(
    sentences=oraciones,
    vector_size=10,
    window=5,
    min_count=1,
    workers=1,
    seed=42,
    epochs=100
)

vocabulario = list(modelo.wv.index_to_key)

print(f"Palabras en el vocabulario: {len(vocabulario)}")
print("Ejemplo de palabras:", vocabulario[:15])

# 4. Representación gráfica 3D mediante PCA
# Seleccionamos hasta 40 palabras para que las etiquetas sean legibles.
palabras_grafica = vocabulario[:40]
vectores = np.array([modelo.wv[p] for p in palabras_grafica])

if len(palabras_grafica) >= 3:
    pca = PCA(n_components=3)
    vectores_3d = pca.fit_transform(vectores)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        vectores_3d[:, 0],
        vectores_3d[:, 1],
        vectores_3d[:, 2],
        s=45,
        alpha=0.8
    )

    for i, palabra in enumerate(palabras_grafica):
        ax.text(
            vectores_3d[i, 0],
            vectores_3d[i, 1],
            vectores_3d[i, 2],
            palabra,
            fontsize=8
        )

    ax.set_title(
        "Orgullo y prejuicio: espacios vectoriales Word2Vec"
    )
    ax.set_xlabel("Dimensión 1")
    ax.set_ylabel("Dimensión 2")
    ax.set_zlabel("Dimensión 3")
    plt.tight_layout()
    plt.savefig("grafica_pca_3d.png", dpi=150)
    plt.close()
    print("Imagen generada: grafica_pca_3d.png")

# 5. Heatmap de similitud semántica
# Se comparan palabras frecuentes del vocabulario.
palabras_heatmap = vocabulario[:15]

if len(palabras_heatmap) >= 2:
    vectores_heatmap = np.array(
        [modelo.wv[p] for p in palabras_heatmap]
    )
    similitudes = cosine_similarity(vectores_heatmap)

    fig, ax = plt.subplots(figsize=(12, 10))
    imagen = ax.imshow(
        similitudes,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    ax.set_xticks(range(len(palabras_heatmap)))
    ax.set_xticklabels(
        palabras_heatmap, rotation=60, ha="right", fontsize=8
    )
    ax.set_yticks(range(len(palabras_heatmap)))
    ax.set_yticklabels(palabras_heatmap, fontsize=8)

    ax.set_title(
        "Similitud coseno entre palabras de Orgullo y prejuicio"
    )
    fig.colorbar(imagen, ax=ax, label="Similitud coseno")
    plt.tight_layout()
    plt.savefig("analogia_heatmap.png", dpi=150)
    plt.close()
    print("Imagen generada: analogia_heatmap.png")

# 6. Consultar palabras similares
for palabra in ["amor", "familia", "matrimonio"]:
    if palabra in modelo.wv:
        print(f"\nPalabras similares a '{palabra}':")
        for similar, puntuacion in modelo.wv.most_similar(
            palabra, topn=3
        ):
            print(f"  {similar}: {puntuacion:.3f}")
    else:
        print(
            f"\n'{palabra}' no aparece en el vocabulario "
            "del texto procesado."
        )

# 7. Guardar el modelo entrenado
modelo.save("modelo_word2vec.model")
print("\nProceso de vectorización terminado.")

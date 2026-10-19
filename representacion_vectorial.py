
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 1. Cargar el texto que ya fue limpiado y lematizado
RUTA_TEXTO = Path("texto_limpio.txt")

if not RUTA_TEXTO.exists():
    raise FileNotFoundError(
        "No se encontró texto_limpio.txt. "
        "Primero ejecuta limpieza_texto.py."
    )

texto = RUTA_TEXTO.read_text(encoding="utf-8").strip()

if not texto:
    raise ValueError("El archivo texto_limpio.txt está vacío.")

# 2. Dividir el libro en fragmentos para compararlos
palabras = texto.split()
tamano_fragmento = 500

corpus_lematizado = [
    " ".join(palabras[i:i + tamano_fragmento])
    for i in range(0, len(palabras), tamano_fragmento)
]

if len(corpus_lematizado) < 3:
    raise ValueError(
        "Se necesitan al menos 3 fragmentos para aplicar PCA en 3D. "
        "Comprueba que texto_limpio.txt contenga el libro completo."
    )

print(f"Fragmentos procesados: {len(corpus_lematizado)}")

# 3. Representación Bag of Words
bow_vectorizer = CountVectorizer()
X_bow = bow_vectorizer.fit_transform(corpus_lematizado)
vocab_bow = bow_vectorizer.get_feature_names_out()

print("\n--- BAG OF WORDS ---")
print("Dimensiones de la matriz:", X_bow.shape)
print("Primeras 15 palabras:", vocab_bow[:15])
print("Conteos del primer fragmento:")
print(X_bow[0, :15].toarray())

# 4. Representación TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_tfidf = tfidf_vectorizer.fit_transform(corpus_lematizado)
vocab_tfidf = tfidf_vectorizer.get_feature_names_out()

print("\n--- TF-IDF ---")
print("Dimensiones de la matriz:", X_tfidf.shape)
print("Primeras 15 palabras:", vocab_tfidf[:15])

# Mostrar palabras con mayor peso en el primer fragmento
pesos = X_tfidf[0].toarray().ravel()
indices = np.argsort(pesos)[::-1][:10]

print("10 palabras con mayor peso en el primer fragmento:")
for indice in indices:
    if pesos[indice] > 0:
        print(f"{vocab_tfidf[indice]}: {pesos[indice]:.4f}")

# 5. Reducir la matriz TF-IDF a 3 dimensiones con PCA
matriz_tfidf = X_tfidf.toarray()
pca = PCA(n_components=3)
coordenadas = pca.fit_transform(matriz_tfidf)

print("\n--- PCA ---")
print("Varianza explicada por componente:",
      pca.explained_variance_ratio_)
print("Varianza explicada total:",
      round(pca.explained_variance_ratio_.sum(), 4))

# 6. Graficar los fragmentos del libro en 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

fragmentos = np.arange(1, len(corpus_lematizado) + 1)

ax.scatter(
    coordenadas[:, 0],
    coordenadas[:, 1],
    coordenadas[:, 2],
    c=fragmentos,
    cmap="viridis",
    s=45
)

ax.set_title("Orgullo y prejuicio: representación TF-IDF con PCA")
ax.set_xlabel("Componente principal 1")
ax.set_ylabel("Componente principal 2")
ax.set_zlabel("Componente principal 3")

plt.tight_layout()
plt.savefig("grafica_pca_3d.png", dpi=300)
plt.show()

print("\nGráfica guardada en grafica_pca_3d.png")

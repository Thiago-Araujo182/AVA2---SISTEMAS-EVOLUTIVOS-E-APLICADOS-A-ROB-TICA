import numpy as np
import cv2
from deepface import DeepFace
from google.colab.patches import cv2_imshow
from google.colab import files

# Upload das suas fotos
print("Faça upload das suas fotos com expressões diferentes:")
uploaded = files.upload()

# Analisar cada foto
for filename in uploaded.keys():
    print(f"\n{'='*60}")
    print(f"Analisando: {filename}")
    print(f"{'='*60}")

    img = cv2.imread(filename)

    # Análise facial (idade, gênero, emoção, etnia)
    result = DeepFace.analyze(
        img_path=filename,
        actions=['age', 'gender', 'emotion', 'race'],
        enforce_detection=False
    )

    # DeepFace.analyze retorna uma lista (uma entrada por rosto detectado)
    face = result[0]

    print(f"Idade estimada: {face['age']} anos")
    print(f"Gênero: {face['dominant_gender']} ({face['gender'][face['dominant_gender']]:.1f}%)")
    print(f"Emoção dominante: {face['dominant_emotion']} ({face['emotion'][face['dominant_emotion']]:.1f}%)")
    print(f"Etnia dominante: {face['dominant_race']} ({face['race'][face['dominant_race']]:.1f}%)")

    print("\nTodas as emoções detectadas:")
    for emocao, valor in sorted(face['emotion'].items(), key=lambda x: -x[1]):
        print(f"  {emocao}: {valor:.2f}%")

    cv2_imshow(img)

import cv2
from fer import FER
import numpy as np
import random
import webbrowser
from collections import deque

# === Initialisation du détecteur FER avec MTCNN ===
detector = FER(mtcnn=True)

# === Messages selon l’humeur ===
positive_messages = [
    "Super ! Continue à sourire 😄",
    "Tu rayonnes aujourd’hui 🌟",
    "Félicitations ! Garde cette bonne énergie 💪",
    "Défi du jour : fais sourire quelqu’un autour de toi 😁"
]

neutral_messages = [
    "On dirait que tu es calme... un petit sourire ? 😊",
    "Relax, tout va bien ✨",
    "Petit conseil : pense à un bon souvenir 😌",
    "Et si tu mettais ta musique préférée ? 🎶"
]

negative_quotes = [
    "Ne te décourage pas, les nuages passent toujours ☀️",
    "Chaque jour est une nouvelle chance 💫",
    "Respire, souris, recommence 🌿",
    "Tu es plus fort(e) que tu ne le penses 💪"
]

relaxing_songs = [
    "https://www.youtube.com/watch?v=2OEL4P1Rz04",  # Chill music
    "https://www.youtube.com/watch?v=1ZYbU82GVz4",  # Relaxing piano
    "https://www.youtube.com/watch?v=DWcJFNfaw9c"   # Calm background
]

# === Moyenne glissante pour lisser les émotions ===
history = deque(maxlen=10)

# === Réaction selon l’émotion dominante ===
def react_to_emotion(emotion):
    if emotion == "happy":
        print("🎉", random.choice(positive_messages))
    elif emotion == "sad":
        print("💬", random.choice(negative_quotes))
        print("🎵 Je te propose d’écouter ceci pour te détendre :")
        webbrowser.open(random.choice(relaxing_songs))
    elif emotion == "neutral":
        print("🙂", random.choice(neutral_messages))

# === Capture webcam ===
cap = cv2.VideoCapture(0)
print("🎥 Assistant émotionnel en cours... (Appuie sur 'q' pour quitter)")

last_emotion = None

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Erreur : impossible d'accéder à la caméra.")
        break

    results = detector.detect_emotions(frame)

    if results:
        face = results[0]
        (x, y, w, h) = face["box"]
        emotions = face["emotions"]

        # Ajouter à l’historique pour lisser
        history.append(emotions)
        avg_emotions = {k: np.mean([h[k] for h in history]) for k in emotions}

        # Trouver l’émotion dominante
        dominant = max(avg_emotions, key=avg_emotions.get)

        # Réagir uniquement si elle change
        if dominant != last_emotion:
            react_to_emotion(dominant)
            last_emotion = dominant

        # Couleur selon émotion
        if dominant == "happy":
            color = (0, 255, 0)
        elif dominant == "sad":
            color = (0, 0, 255)
        else:
            color = (200, 200, 200)

        # === Afficher uniquement l’émotion dominante dans le cadre ===
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.rectangle(frame, (x, y - 36), (x + w, y), color, -1)
        cv2.putText(frame, f"{dominant.upper()}", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    cv2.imshow("Assistant Émotionnel (Appuie sur 'q' pour quitter)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# === Libération des ressources ===
cap.release()
cv2.destroyAllWindows()

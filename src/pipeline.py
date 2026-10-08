"""Pipeline par image : détection du visage (Haar), recadrage/alignement, 48x48,
puis prédiction par les 3 modèles avec chronométrage séparé."""
import time
from dataclasses import dataclass, field

import cv2
import joblib
import numpy as np
import torch

from src.cnn import load_model, predict_proba
from src.config import IMG_SIZE, LABELS, MODELS_DIR
from src.features import hog_features, lbp_features

MODEL_NAMES = {
    "lbp_knn": "LBP + KNN",
    "hog_svm": "HOG + SVM linéaire",
    "mini_xception": "mini-Xception (CNN)",
}


@dataclass
class Prediction:
    model: str
    probs: np.ndarray            # (7,), dans l'ordre de LABELS
    latency_ms: float            # features + inférence

    @property
    def label(self):
        return LABELS[int(self.probs.argmax())]

    @property
    def confidence(self):
        return float(self.probs.max())


@dataclass
class FaceResult:
    faces: list                  # toutes les boîtes (x, y, w, h), la plus grande en premier
    face48: np.ndarray | None    # visage retenu, uint8 48x48
    detect_ms: float
    predictions: list = field(default_factory=list)


class FaceDetector:
    def __init__(self):
        root = cv2.data.haarcascades
        self.face = cv2.CascadeClassifier(root + "haarcascade_frontalface_default.xml")
        self.eyes = cv2.CascadeClassifier(root + "haarcascade_eye.xml")

    def detect(self, gray):
        """Boîtes (x, y, w, h) triées par surface décroissante."""
        boxes = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        return sorted((tuple(int(v) for v in b) for b in boxes), key=lambda b: b[2] * b[3], reverse=True)

    def eye_angle(self, face_gray):
        """Angle (degrés) de la ligne des yeux, ou None si les deux yeux ne sont pas trouvés."""
        h, w = face_gray.shape
        eyes = self.eyes.detectMultiScale(face_gray[: h // 2], scaleFactor=1.1, minNeighbors=6,
                                          minSize=(w // 10, w // 10))
        if len(eyes) < 2:
            return None
        eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        (x1, y1), (x2, y2) = sorted((x + ew / 2, y + eh / 2) for x, y, ew, eh in eyes)
        if x2 - x1 < w * 0.2:  # deux détections du même œil
            return None
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        return angle if abs(angle) < 25 else None

    def crop(self, gray, box, margin=0.1, align=True):
        """Recadre le visage (avec une marge), l'aligne sur les yeux si possible, redimensionne en 48x48."""
        x, y, w, h = box
        if align:
            angle = self.eye_angle(gray[y:y + h, x:x + w])
            if angle:
                center = (x + w / 2, y + h / 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                gray = cv2.warpAffine(gray, M, gray.shape[::-1], flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        m = int(margin * w)
        x0, y0 = max(x - m, 0), max(y - m, 0)
        x1, y1 = min(x + w + m, gray.shape[1]), min(y + h + m, gray.shape[0])
        return cv2.resize(gray[y0:y1, x0:x1], (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)


class EmotionPipeline:
    def __init__(self, device="cpu"):
        self.detector = FaceDetector()
        knn = joblib.load(MODELS_DIR / "lbp_knn.joblib")
        svm = joblib.load(MODELS_DIR / "hog_svm.joblib")
        self.knn, self.lbp_params = knn["model"], knn["lbp_params"]
        self.knn.n_jobs = 1  # une seule requête à la fois : le parallélisme ne fait qu'ajouter du surcoût
        self.svm, self.hog_params = svm["model"], svm["hog_params"]
        self.device = torch.device(device)
        self.cnn, self.tta = load_model(MODELS_DIR / "mini_xception.pt", self.device)
        self.warmup()

    def warmup(self, n=3):
        """Premiers appels plus lents (allocation, JIT, caches) : exclus des mesures."""
        dummy = np.random.randint(0, 255, (IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        for _ in range(n):
            self.predict_face(dummy)

    def _knn(self, face):
        return self.knn.predict_proba(lbp_features(face, **self.lbp_params)[None])[0]

    def _svm(self, face):
        return self.svm.predict_proba(hog_features(face, **self.hog_params)[None])[0]

    def _cnn(self, face):
        x = torch.from_numpy(face).float().div_(255)[None, None].to(self.device)
        return predict_proba(self.cnn, x, self.tta)[0].cpu().numpy()

    def predict_face(self, face48):
        """Prédiction des 3 modèles sur un visage 48x48 uint8, chacun chronométré séparément."""
        preds = []
        for name, fn in (("lbp_knn", self._knn), ("hog_svm", self._svm), ("mini_xception", self._cnn)):
            t0 = time.perf_counter()
            probs = fn(face48)
            preds.append(Prediction(name, np.asarray(probs, dtype=np.float32),
                                    (time.perf_counter() - t0) * 1000))
        return preds

    def process(self, image, detect=True, align=True):
        """image : RGB ou niveaux de gris (numpy). Si detect=False, l'image entière est
        considérée comme le visage (cas des images FER-2013 déjà recadrées)."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        if not detect:
            face = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            return FaceResult([], face, 0.0, self.predict_face(face))

        t0 = time.perf_counter()
        faces = self.detector.detect(gray)
        detect_ms = (time.perf_counter() - t0) * 1000
        if not faces:
            return FaceResult([], None, detect_ms)
        face = self.detector.crop(gray, faces[0], align=align)
        return FaceResult(faces, face, detect_ms, self.predict_face(face))

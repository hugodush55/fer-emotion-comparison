# Reconnaissance d'émotions faciales : LBP+KNN vs HOG+SVM vs mini-Xception

TD4 Apprentissage pour la robotique (ENSTA). Comparaison de trois classifieurs d'émotions sur FER-2013
(7 classes : Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral), via une interface qui applique les trois
modèles au même visage et affiche label, confiance et latence (ms).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

(`requirements-lock.txt` contient les versions exactes testées.)

Télécharger FER-2013 dans `data/` (voir [data/README.md](data/README.md)).

## Lancement

```bash
python app.py
```

## Structure

```
src/         features, entraînement des 3 modèles, pipeline visage, évaluation
models/      poids entraînés (légers uniquement)
reports/     figures (matrices de confusion, etc.)
app.py       interface Gradio
```

## Résultats

| Modèle | Accuracy test | Latence moyenne (ms) |
|---|---|---|
| LBP + KNN | à venir | à venir |
| HOG + SVM linéaire | à venir | à venir |
| mini-Xception (CNN) | à venir | à venir |

## Poids du CNN

Entraîné par nos soins sur FER-2013 (PyTorch), voir `src/train_cnn.py`.

## Démo vidéo

À venir.

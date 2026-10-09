# Reconnaissance d'émotions faciales : LBP+KNN vs HOG+SVM vs mini-Xception

TD4 Apprentissage pour la robotique (ENSTA). Trois classifieurs d'émotions entraînés sur FER-2013
(7 classes : Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) et comparés dans une interface Gradio
qui applique les trois modèles **au même visage** et affiche pour chacun le label, la confiance et la latence (ms).

**Démo vidéo :** _lien à venir_

## Résultats

| Modèle | Latence moyenne (ms) | Accuracy 50 images | Accuracy test complet (7 178 images) |
|---|---|---|---|
| LBP + KNN | **4,1** | 38 % | **39,5 %** |
| HOG + SVM linéaire (OvR) | **3,2** | 50 % | **43,5 %** |
| mini-Xception (CNN) | **8,5** (5,1 sur GPU) | 72 % | **69,4 %** |

- Latence = extraction des features + inférence, sur CPU, moyenne sur 500 images de test, premiers appels exclus
  (préchauffage). Pour le CNN, elle inclut le TTA (2 passes : image et image retournée).
- La détection Haar + recadrage, commune aux trois modèles, coûte en plus environ 12 ms sur une image 640×480.
- Détail (écart-type, p95, accuracy sur 500 images) : [reports/results.md](reports/results.md), généré par `python -m src.benchmark`.

![Matrices de confusion](reports/confusion_all.png)

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Testé avec Python 3.10, PyTorch 2.14 (CUDA 13), OpenCV 4.14, scikit-learn 1.7, Gradio 6.
Les versions exactes sont dans [requirements-lock.txt](requirements-lock.txt). OpenCV doit être en 4.x :
la version 5 a retiré `CascadeClassifier` (détection Haar).

Les poids entraînés sont fournis dans `models/` (6 Mo au total) : **l'application fonctionne sans le dataset**.
Pour réentraîner, placer FER-2013 (version Kaggle en dossiers d'images) dans `data/fer2013/` (voir [data/README.md](data/README.md)).

## Lancement

```bash
python app.py                 # CNN sur CPU
python app.py --device cuda   # CNN sur GPU
```

Puis ouvrir http://localhost:7860. Deux onglets :

- **Image** : import d'une photo (ou des exemples FER-2013 ; cocher alors « déjà recadrée », car Haar ne détecte pas
  de visage dans une image 48×48) ;
- **Webcam** : analyse en continu (environ 4 images/s).

L'interface affiche l'image avec les visages détectés (en vert celui qui est analysé, le plus grand), le visage 48×48
réellement envoyé aux modèles, puis trois cartes côte à côte : label, confiance, latence de l'image courante et moyenne
de la session, et les probabilités des 7 classes. Si aucun visage n'est détecté, l'interface l'indique.

### Réentraîner et évaluer

```bash
python -m src.train_knn                  # modèle 1 (~5 min)
python -m src.train_svm                  # modèle 2 (~10 min, LinearSVC est mono-thread)
python -m src.train_cnn --mode full --width 2 --epochs 200 --erasing 0.5 --mix-prob 0.5   # modèle 3 (~20 min sur RTX 4060)
python -m src.benchmark                  # tableau de latence / accuracy
```

## Structure

```
app.py              interface Gradio (image + webcam)
src/config.py       ordre des labels, chemins
src/data.py         chargement de FER-2013 (cache .npz), découpage train/validation
src/features.py     LBP par blocs, HOG
src/train_knn.py    modèle 1 : recherche d'hyperparamètres, entraînement, évaluation
src/train_svm.py    modèle 2
src/cnn.py          mini-Xception, augmentations, TTA
src/train_cnn.py    modèle 3
src/pipeline.py     détection Haar, alignement, recadrage, prédiction chronométrée des 3 modèles
src/evaluation.py   accuracy, rapport par classe, matrices de confusion
src/benchmark.py    mini-tableau du rendu
models/             poids entraînés
reports/            métriques, matrices de confusion, courbes, logs d'entraînement
```

## Méthode

### Données

FER-2013 : images 48×48 en niveaux de gris, 28 709 en train et 7 178 en test (split Kaggle = split officiel
Training / PublicTest + PrivateTest). Ordre des labels : 0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral.
Le jeu est déséquilibré : Happy 7 215 images d'entraînement, Disgust seulement 436.

Le split Kaggle ne fournit pas de validation : **tous les hyperparamètres sont choisis sur une validation stratifiée
de 10 % tirée du train** (graine fixe). Le test n'est utilisé que pour l'évaluation finale.

### Modèle 1 : LBP + KNN

- LBP `uniform` (`skimage.feature.local_binary_pattern`), image découpée en 4×4 = 16 blocs, un histogramme normalisé
  par bloc, concaténés (P+2 classes par bloc, soit 160 features pour P=8).
- Grille testée : (P, R) ∈ {(8,1), (8,2), (8,3), (16,2)}, k ∈ {1, 3, 5, 9, 15, 25, 35, 50}, poids uniformes ou
  inverses de la distance, distance Manhattan (p=1) ou euclidienne (p=2) : 128 configurations
  ([reports/lbp_knn_grid.json](reports/lbp_knn_grid.json)).
- **Retenu : P=8, R=2, k=50, poids = distance, Manhattan.** Validation 40,2 %, test 39,5 %.
- Toutes les configurations donnent entre 36 et 40 % : la représentation, pas le réglage, limite le modèle.
  Manhattan bat l'euclidienne presque partout, ce qui est cohérent pour comparer des histogrammes.
- Confiance = `predict_proba` (vote des voisins pondéré par la distance).

### Modèle 2 : HOG + SVM linéaire (One-vs-Rest)

- HOG (`skimage.feature.hog`) : cellules 8×8, blocs 2×2 cellules, 9 orientations, normalisation L2-Hys
  (900 features). Features standardisées, `LinearSVC` (One-vs-Rest par défaut), C ∈ {1e-4, …, 1}.
- **Retenu : C = 0,01.** Validation 44,7 %, test 43,5 %. La courbe en C est plate (44,1 à 44,7 %) : le modèle est
  limité par les features, pas par la régularisation.
- Des cellules 6×6 (1 764 features) ont aussi été testées : 44,8 % en validation et en test, soit +0,1 point en
  validation, dans le bruit. On garde les 8×8 de l'énoncé
  ([reports/hog_svm_grid_cells_6_8.json](reports/hog_svm_grid_cells_6_8.json)).
- Confiance : `LinearSVC` ne donne pas de probabilités. Les scores sont calibrés par `CalibratedClassifierCV`
  (sigmoïde, 3 folds).

### Modèle 3 : mini-Xception

**Provenance des poids : entraînés par nous**, en PyTorch, sur FER-2013 (pas de poids pré-entraînés, donc aucun
risque de décalage dans l'ordre des labels). Architecture d'après Arriaga et al. (2017) : 2 convolutions 3×3, puis
4 blocs résiduels (2 convolutions séparables en profondeur, BatchNorm, ReLU, max-pooling, raccourci 1×1 de stride 2),
une convolution vers 7 canaux et un global average pooling. Entrée 48×48 normalisée dans [0, 1], confiance = softmax.

Augmentations (calculées sur GPU, par batch) : flip horizontal, rotation ±10°, décalage ±10 %, zoom ±10 %,
luminosité et contraste ±20 %.

Le modèle a été amélioré en trois étapes, chacune choisie sur la validation :

| Version | Changements | Paramètres | Validation | Test |
|---|---|---|---|---|
| v1 | Recette du TD : Adam lr 1e-3, `ReduceLROnPlateau`, batch 64, early stopping sur la val-loss (arrêt epoch 86) | 57 k | 63,4 % | 63,7 % |
| — | v1 + label smoothing 0,1, warmup + cosinus sur 100 epochs, TTA flip | 57 k | 66,2 % | 66,0 % |
| v2 | idem, largeur ×2 (tous les canaux doublés) | 202 k | 67,4 % | 69,0 % |
| **v3** | v2 + random erasing (p=0,5) + mixup/cutmix (1 batch sur 2), 200 epochs | 202 k | 69,0 % | **69,4 %** |

- La nouvelle recette (label smoothing, schedule cosinus, TTA) apporte environ 3 points (63,4 → 66,2 % en validation),
  la largeur ×2 environ 1 point en validation (3 en test). Le TTA seul compte pour 1,5 à 2 points
  (v3 : 67,9 % sans, 69,4 % avec), pour une latence doublée.
- Les augmentations fortes de v3 gagnent 1,6 point en validation mais seulement 0,4 en test, dans l'incertitude
  statistique (environ ±0,5 point sur 7 178 images) : on a atteint le plafond de cette architecture sur ce dataset.
- Le modèle final est réentraîné sur tout le train (train + validation) avec le schedule retenu.
- Courbes d'apprentissage : [reports/](reports/) (`mini_xception_*_curves.png`), logs dans `reports/*_train.log`.

### Pipeline par image

1. Conversion en niveaux de gris, détection des visages par Haar cascade (`haarcascade_frontalface_default`) ;
   s'il y a plusieurs visages, le plus grand est analysé, s'il n'y en a aucun, l'interface l'indique.
2. Alignement : les yeux sont détectés (`haarcascade_eye`) dans la moitié haute du visage, et l'image est tournée pour
   rendre la ligne des yeux horizontale (ignoré si les deux yeux ne sont pas trouvés ou si l'angle dépasse 25°).
3. Recadrage avec une marge de 10 %, redimensionnement en 48×48.
4. Passage dans les trois modèles, chacun chronométré séparément (`time.perf_counter`).

## Analyse

### Classes confondues

| Rappel par classe | Angry | Disgust | Fear | Happy | Sad | Surprise | Neutral |
|---|---|---|---|---|---|---|---|
| LBP + KNN | 25 % | 27 % | 21 % | 69 % | 23 % | 48 % | 36 % |
| HOG + SVM | 25 % | 13 % | 18 % | 78 % | 25 % | 54 % | 43 % |
| mini-Xception | 60 % | 50 % | 47 % | 89 % | 60 % | 80 % | 71 % |

- **Disgust → Angry** : la confusion la plus marquée, qui augmente même avec le CNN (14 % pour le KNN, 23 % pour le
  SVM, 29 % pour le CNN). Les deux expressions partagent les mêmes indices (sourcils froncés, nez plissé, lèvre
  supérieure relevée). Disgust est aussi de loin la classe la plus rare (1,5 % du train) : le modèle apprend mal sa
  frontière et la rabat sur Angry.
- **Fear → Sad** (13 %, 16 %, 21 %) et **Fear → Surprise** (8 à 14 %) : Fear est la classe la plus difficile pour
  tous les modèles. Elle partage la bouche ouverte et les yeux écarquillés avec Surprise, et les sourcils relevés au
  centre avec Sad.
- **Sad ↔ Neutral** (Sad → Neutral 18 à 25 % ; Neutral → Sad 9 à 14 %) : des expressions de faible intensité, souvent
  ambiguës même pour un humain (l'accuracy humaine sur FER-2013 est estimée à environ 65 %).
- **Biais vers Happy** des modèles classiques : le KNN classe Happy 33 à 38 % des Angry, Disgust, Fear, Sad et Neutral.
  Happy est la classe majoritaire (25 % du train) et ses voisins dominent dans l'espace des histogrammes LBP. Le SVM
  atténue ce biais (21 à 29 %) et le CNN le supprime (1 à 6 %).
- Happy et Surprise, les expressions les plus marquées visuellement (sourire, bouche ouverte en O), sont les mieux
  reconnues par les trois modèles.

### Précision et vitesse

- **Le CNN est meilleur de 26 points que le meilleur modèle classique**, pour une latence du même ordre de grandeur
  (8,5 ms contre 3,2 à 4,1 ms sur CPU, 5,1 ms sur GPU). Les trois modèles sont largement temps réel : même en ajoutant
  la détection (12 ms), une image est traitée en moins de 25 ms, et c'est la détection qui domine.
- Le CNN est aussi le plus léger (0,8 Mo) : le KNN doit stocker toute sa base d'entraînement (5,1 Mo) et son coût
  d'inférence croît avec elle.
- Les features faites à la main plafonnent : 128 configurations LBP+KNN donnent toutes 36 à 40 %, et le HOG ne gagne
  que 0,1 point de validation en passant de cellules 8×8 à 6×6. Le CNN apprend des features adaptées à la tâche et bénéficie des
  augmentations, ce que les modèles classiques ne peuvent pas exploiter de la même façon.
- Le TTA du CNN illustre le compromis : +1,5 point pour une latence doublée.

### Limites

- Haar est sensible à la pose, à l'éclairage et à la résolution. Sur des visages FER-2013 agrandis et collés dans une
  image 640×480 (visages très serrés, sans contexte), il en détecte 78,5 %. Sur une webcam (visage de face, contexte
  normal), la détection est beaucoup plus fiable, mais un détecteur CNN (par exemple YuNet) serait plus robuste.
- FER-2013 contient du bruit d'annotation et quelques images vides ou sans visage, ce qui limite l'accuracy atteignable
  (les meilleurs résultats publiés, avec des réseaux de plusieurs millions de paramètres, sont autour de 73 à 76 %).
- Le recadrage Haar ne correspond pas exactement au cadrage des images FER-2013 : les performances en conditions
  réelles (webcam) peuvent être inférieures à celles mesurées sur le jeu de test.

## Références

- I. Goodfellow et al., « Challenges in Representation Learning: A report on three machine learning contests », 2013 (FER-2013).
- O. Arriaga, M. Valdenegro-Toro, P. Plöger, « Real-time Convolutional Neural Networks for Emotion and Gender Classification », 2017 (mini-Xception).
- T. Ojala, M. Pietikäinen, T. Mäenpää, « Multiresolution gray-scale and rotation invariant texture classification with local binary patterns », PAMI 2002.
- N. Dalal, B. Triggs, « Histograms of Oriented Gradients for Human Detection », CVPR 2005.
- P. Viola, M. Jones, « Rapid Object Detection using a Boosted Cascade of Simple Features », CVPR 2001.

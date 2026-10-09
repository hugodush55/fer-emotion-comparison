| Modèle | Latence moyenne (ms) | Écart-type / p95 (ms) | Accuracy 50 images | Accuracy 500 images | Accuracy test complet (7178) |
|---|---|---|---|---|---|
| LBP + KNN | **4.3** | 0.9 / 5.7 | 38% | 38.2% | **39.5%** |
| HOG + SVM linéaire | **3.5** | 0.8 / 4.7 | 46% | 43.8% | **44.8%** |
| mini-Xception (CNN) | **8.7** | 4.6 / 12.2 | 72% | 71.2% | **69.4%** |

Latences sur CPU, 500 images de test, premiers appels exclus ; latence = extraction des features + inférence (+ TTA flip pour le CNN).
Détection Haar + recadrage (image 640x480) : 12.0 ms en moyenne, commune aux 3 modèles.
mini-Xception sur GPU (NVIDIA GeForce RTX 4060 Laptop GPU) : 5.8 ms.

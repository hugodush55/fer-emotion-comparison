| Modèle | Latence moyenne (ms) | Écart-type / p95 (ms) | Accuracy 50 images | Accuracy 500 images | Accuracy test complet (7178) |
|---|---|---|---|---|---|
| LBP + KNN | **4.1** | 1.0 / 5.9 | 38% | 38.2% | **39.5%** |
| HOG + SVM linéaire | **3.2** | 0.7 / 4.3 | 50% | 41.0% | **43.5%** |
| mini-Xception (CNN) | **8.5** | 4.6 / 11.9 | 72% | 71.2% | **69.4%** |

Latences sur CPU, 500 images de test, premiers appels exclus ; latence = extraction des features + inférence (+ TTA flip pour le CNN).
Détection Haar + recadrage (image 640x480) : 11.6 ms en moyenne, commune aux 3 modèles.
mini-Xception sur GPU (NVIDIA GeForce RTX 4060 Laptop GPU) : 5.1 ms.

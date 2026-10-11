| Model | Mean latency (ms) | Std / p95 (ms) | Accuracy 50 images | Accuracy 500 images | Accuracy full test set (7178) |
|---|---|---|---|---|---|
| LBP + KNN | **3.2** | 0.5 / 3.9 | 38% | 38.2% | **39.5%** |
| HOG + linear SVM | **2.4** | 0.4 / 3.0 | 50% | 41.0% | **43.5%** |
| mini-Xception (CNN) | **6.5** | 3.6 / 7.6 | 72% | 71.2% | **69.4%** |

Latencies on CPU, 500 test images, first calls excluded; latency = feature extraction + inference (+ flip TTA for the CNN).
Haar detection + crop (640x480 image): 10.6 ms on average, shared by the 3 models (face found in 78% of the synthetic test images).
mini-Xception on GPU (NVIDIA GeForce RTX 4060 Laptop GPU): 5.8 ms.

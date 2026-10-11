---
title: Facial Emotion Comparison
emoji: 🙂
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.29.1
python_version: "3.10"
app_file: app.py
pinned: false
short_description: LBP+KNN vs HOG+SVM vs mini-Xception on FER-2013
---

# Facial emotion recognition: 3 models on the same face

Live demo of a course project (TD4, Machine Learning for Robotics, ENSTA): three classifiers trained on FER-2013
(7 emotions) are applied to the same detected face, and each one shows its label, confidence and latency.

| Model | Test accuracy (FER-2013, 7,178 images) |
|---|---|
| LBP + KNN | 39.5 % |
| HOG + linear SVM | 43.5 % |
| mini-Xception (CNN) | 69.4 % |

**How to use:** upload a frontal face photo in the *Image* tab, or open the *Webcam* tab and allow camera access.

This Space runs on a shared free CPU, so the latencies shown here are higher than the reference values
measured locally (2 to 7 ms per model). Full code, method, results and analysis:
https://github.com/hugodush55/fer-emotion-comparison

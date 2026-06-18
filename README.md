# Emotion_detection

phase 1:
tasks:
1) get datasets
2) understand classes
3) explore dataset
4) visulization

output:
1) dataset
2) EDA Notebook
3) Dataset Report
4) train.csv
5) test.csv

Phase 2:

tasks:
Preprocessing
1) resize (check the best param to resize)
2) Normalize (find the best scaling factor) (grayscale) (224*224)
3) Data-Augumentation (horizontal flip/ rotation +-15 Degree tilt/ brightness jitter/ random crop) => lower the overfitting
4) Train Validation test (70/15/15)

phase 2.2:
Baseline Model, CNN, Conv, ReLU, Pooling, Dense ==> workking baseline

phase 2.3:
Experiment

1) Deeper CNN
2) BatchNorm
3) Dropout
4) Learning rate scheduler
5) Early stopping

phase 2.4:
1) Accuracy
2) Loss Curves
3) Confusion Matrix
4) Precision
5) Recall
6) F1              ====> get best model

before stating phase 3: decide on a template Local deployment

phase 3:

Objective
1) Serve predictions.
2) Framework
3) FastAPI
4) Endpoints

Tasks
1) Load model once
2) Receive image
3) Run preprocessing
4) Predict emotion
5) Return JSON

backend/
1) api.py
2) requirements.txt
3) Swagger Documentation
FROM python:3
COPY predict.py ./
COPY requirements.txt ./
COPY binary_classifier_data/validation.csv ./
COPY knn_train.pkl ./
COPY labelencoder.pkl ./
COPY preprocess_pipeline.pkl ./
COPY rf_all.pkl ./

RUN pip install -r requirements.txt
CMD ["python", "predict.py"]

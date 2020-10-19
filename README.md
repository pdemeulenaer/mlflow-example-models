# mlflow-models-examples

The purpose of this repo is to explore the tracking and deployment of several machine learning model flavours using the MLFlow platform

rf-spark: random forest using pyspark.ml library

rf-sklearn random forest using scikit-learn library

Each folder follows the same structure:

    training.py: contains a train function which trains a model and saves the model artifact to MLFlow model registry (and MLFlow experiments)

    scoring.py:

        contains a score function which de-serializes the model from the MLFlow model registry

        contains an evaluate function which applies the score function on a (historical) test dataset, builds performance metrics and log them to the MLFlow model registry
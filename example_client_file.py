from models.knn import KNN

knn = KNN(20, "Outcome")

knn.train("/Users/alifabdullah/Collaboration/Kaggle-ML-Algorithm-Musings/datasets/diabetes.csv")
training_set = knn.get_training_set()
print(f"Training set! {training_set}")
print(f"Predicted label: {knn.predict(training_set.iloc[0])}")
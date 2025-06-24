# KNN From Scratch implementation
# This code block will contain implementations of the KNN algorithm and some of its variants.
import numpy as np
import pandas as pd
class KNN:
  def __init__(self, k):
    """
    Pass in one argument - k, the number of nearest neighbors to consider.

    Args:
      self.k - the number of nearest neighbors to consider
      self.training_set - the training set we will inevitably pass to this 
      algorithm. This variable is instantiated as an empty Pandas Dataframe
    """
    self.k = k
    self.training_set = pd.DataFrame()
    self.label_set = []
  
  def train(self, csvfile: str, target_column: str):
    """
    Sets self.training_set = csv_in_panda_form, where csv_in_panda_form is a 
    pandas dataframe that contains the information from the linked csv file.

    Args:
      csvfile - A string path to the csv file we want to take in. We desire an absolute path.
    """
    csv_in_panda_form = pd.read_csv(csvfile)
    self.training_set = csv_in_panda_form
    self.label_set = self.training_set[target_column].unique()
  
  def get_training_set(self):
    """
    Returns the training set
    """
    return self.training_set
  
  def _nearest_neighbors(self, x):
    """
    Goes through every row to find the k nearest neighbors of testing instance x
    """
    training_set_test_instance_difference = []
    training_set_rows = []
    feature_columns = self.training_set.columns.drop("Outcome")
    for index, row in self.training_set.iterrows():
      training_set_test_instance_difference.append(float((x[feature_columns] - row[feature_columns]).abs().sum()))
      training_set_rows.append(row)
    
    similarity_scores, corresponding_rows = list(zip(*sorted(zip(training_set_test_instance_difference, training_set_rows), key=lambda x: x[0])))
    

    # Get the k nearest neighbors 
    return corresponding_rows[:self.k]

  def predict(self, x):
    """
    Predicts a label, y, for the testing instance, x, based on the k nearest neighbors.
    """
    nearest_neighbors_of_x = self._nearest_neighbors(x)
    label_counts = []
    for label in self.label_set:
      count_for_current_label = 0
      for neighbor in nearest_neighbors_of_x:
        if label == neighbor["Outcome"]:
          count_for_current_label += 1
      label_counts.append(count_for_current_label)
    label_counts = np.array(label_counts)

    assert label_counts.sum() == self.k, "Total number of labels assigned and number of nearest neighbors picked do not align."
    return self.label_set[label_counts.argmax()]
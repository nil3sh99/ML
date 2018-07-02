from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import numpy as np

y_true = ["cat", "ant", "cat", "cat", "ant", "bird", "animal", "doggie"]

y_pred = ["ant", "ant", "cat", "cat", "ant", "cat", "doggie", "doggie"]

cm = confusion_matrix(y_true, y_pred, labels = ["ant", "bird", "cat"])
print(metrics.classification_report(y_true, y_pred))
print(cm)

iris = load_iris()
#print(iris)

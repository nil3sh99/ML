
# coding: utf-8

# In[6]:


from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier


iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

import graphviz
dot_data = tree.export_graphviz(clf , out_file = None)
graph = graphviz.Source(dot_data)
graph.render('iris')
dot_data = tree.export_graphviz(clf, out_file = None)

graph = graphviz.Source(dot_data)
graph



# coding: utf-8

# In[52]:


from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

dataset = datasets.load_iris()
#print(dataset.data)

model = DecisionTreeClassifier()
clf = model.fit(dataset.data, dataset.target)

#making predictions

expected = dataset.target
predicted = model.predict(dataset.data)
print(expected)
print(predicted)

#summarazing the fit of the model

print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))


# In[50]:


#check for a user input

data1= [[5.7,3.3,4.5,0.3]]

predict_Class = model.predict(data1)
print ("The flower belongs to:", predict_Class, "Class")


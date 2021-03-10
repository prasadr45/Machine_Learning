# -*- coding: utf-8 -*-
"""
@author: Prasad Gandham
"""

###################################
# this is to draw the confusion matrix form the predictions"
########################################

from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import classification_report

iris=datasets.load_iris()
target = iris.target
print(target)
data = iris.data
print(data)
x_train,x_test,y_train,y_test = train_test_split(data,target,test_size=0.10)
classifier = svm.SVC(kernel='linear',C=0.05)
classifier.fit(x_train,y_train)
y_pred = classifier.predict(x_test)
confusion_mtx = confusion_matrix(y_test,y_pred)
print(confusion_mtx)


title="Confusion Matrix"
fig,ax=plt.subplots()
ax.matshow(confusion_mtx, cmap=plt.cm.Blues)

threshold = confusion_mtx.max()/2.
for i,j in itertools.product(range(confusion_mtx.shape[0]),range(confusion_mtx.shape[1])):
    ax.text(i,j,format(confusion_mtx[i,j],'d'),
    horizontalalignment="center",
    verticalalignment="center",
    color="white" if confusion_mtx[i,j] > threshold else "black")
    plt.tight_layout()
    plt.show()
    print("classification report")
    print(classification_report(y_test,y_pred))
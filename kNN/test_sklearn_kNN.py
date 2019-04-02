# 使用sklearn库中的kNN

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets

digits = datasets.load_digits()
X = digits.data
y = digits.target

X_train,X_test,y_train,y_test = train_test_split(X,y)

knn_clf =  KNeighborsClassifier(3)
knn_clf.fit(X_train,y_train)
y_predict = knn_clf.predict(X_test)
accuracy =  knn_clf.score(X_test,y_test)

print('Origin: ' , y_test)
print('Predict:' , y_predict)
print('Accuracy:' , accuracy)
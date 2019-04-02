from kNN import kNNClassifier
from model_selection import train_test_split
from sklearn import datasets
from metrics import accuracy_score
from preprocessing import StandardScaler

#导入iris数据
iris = datasets.load_iris()
X = iris.data 
y = iris.target

standardScaler = StandardScaler()
standardScaler.fit

#将数据集拆分为train、test数据
X_train,X_test,y_train,y_test = train_test_split(X,y,0.2)

#数据的归一化（训练数据集、测试数据集的归一化）
standardScaler = StandardScaler()
standardScaler.fit(X_train)
X_train = standardScaler.transform(X_train)
X_test = standardScaler.transform(X_test)
print(X_test)

#使用上述数据进行训练和预测
kcf = kNNClassifier(3)
kcf.fit(X_train,y_train)
y_predict = kcf.predict(X_test)

#计算预测准确度
accuracy = accuracy_score(y_test,y_predict)
print('Origin: ' , y_test)
print('Predict:' , y_predict)
print('Accuracy:' , accuracy)


print(kcf.score(X_test,y_test))


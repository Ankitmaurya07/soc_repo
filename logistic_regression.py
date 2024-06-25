
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import datasets
import numpy as np

breast_cancer = datasets.load_breast_cancer() 
X, y = breast_cancer.data, breast_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



class LogisticRegression:
    def __init__(self, lr=0.01, iters=1000):
        self.lr = lr
        self.iters = iters
    
    def fit(self, X, y):
        self.m, self.n = X.shape
        self.weights = np.zeros(self.n)
        self.bias = 0
        self.X = X
        self.y = y
        
        for _ in range(self.iters):
            self.update_weights()
    
    def update_weights(self):
        linear_model = np.dot(self.X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        
        dw = (1 / self.m) * np.dot(self.X.T, (y_predicted - self.y))
        db = (1 / self.m) * np.sum(y_predicted - self.y)
        
        self.weights -= self.lr * dw
        self.bias -= self.lr * db
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)


model = LogisticRegression(lr=0.01, iters=1000)


model.fit(X_train, y_train)


y_pred = model.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')
    

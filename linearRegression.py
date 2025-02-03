import numpy as np

class LinearRegression:

    def __init__(self,lr=0.001,n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        #init parameters
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        #gradient descent
        for _ in range(self.n_iters):
            y_predicted = np.dot(X,self.weights) + self.bias

            dw = (1/n_samples) * np.dot(X.T,(y_predicted-y))
            db = (1/n_samples) * np.sum(y_predicted-y)
             
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self,X):
        y_predicted = np.dot(X,self.weights) + self.bias
        return y_predicted



# Testing
if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

    # fig = plt.figure(figsize=(10,6))
    # plt.scatter(X[:,0],y,color="b",marker="o",s=30)
    # plt.show()

    # print(X_train.shape)
    # print(y_train.shape)

    model  = LinearRegression(0.01)
    model.fit(X_train,y_train)
    predicted = model.predict(X_test)

    def mse(y_true,y_predicted):
        return np.mean((y_true-y_predicted)**2)

    mse_value = mse(y_test,predicted)
    print("MSE:",mse_value)

    cmap =plt.get_cmap("viridis")

    y_pred_line = model.predict(X)
    figure = plt.figure(figsize=(10,6))
    m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
    m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
    plt.plot(X,y_pred_line, color="black", linewidth=2, label="Prediction")
    plt.show()

    
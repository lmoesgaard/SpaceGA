import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


class RF1std:
    def __init__(self):
        self.model = RandomForestRegressor(n_jobs=-1)
        self.model.fit(np.random.randint(2, size=[1000, 1024]), np.random.randint(10, size=1000))

    def predict(self, X):
        return self.model.predict(X)
    

class NN1:
    def __init__(self):
        self.model = MLPRegressor(hidden_layer_sizes=[200, 200],
                                  tol=0.1,
                                  early_stopping=True,
                                  warm_start=True,
                                  alpha=0.01
                                  )
        self.model.fit(np.random.randint(2, size=[1000, 1024]), np.random.randint(10, size=1000))

    def predict(self, X):
        return self.model.predict(X)

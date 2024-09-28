import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from Adaline import Adaline
from sklearn.svm import SVC


# Generació del conjunt de mostres
X, y = make_classification(n_samples=400, n_features=5, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=1,
                           random_state=9)

# Separar les dades: train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Estandaritzar les dades: StandardScaler
scaler = StandardScaler()
X_train_transformed = scaler.fit_transform(X_train)
X_test_transformes = scaler.fit_transform(X_test)

# Entrenam un perceptron
perceptron = Adaline(eta=0.0005, n_iter=60)
perceptron.fit(X_train_transformed, y_train)

# Entrenam una SVM linear (classe SVC)
svc = SVC(C=1.00, kernel='linear')
svc.fit(X_train_transformed, y_train)

# Prediccio
y_predict_percep = perceptron.predict(X_test_transformes)
y_predict_svc = svc.predict(X_test_transformes)

# Metrica
def tasa_acerts(y_true, y_pred):
    acerts = np.sum(y_true == y_pred)
    mostres = len(y_true)
    tasa = acerts / mostres
    return tasa

tasa_percep = tasa_acerts(y_test, y_predict_percep)
print(f"Tasa d'encerts perceptron: {tasa_percep * 100:.2f}%")

# Es pot realitzar de dues formes diferents
# Els acerts son el sumatori de y_test == y_pred
tasa_svc = tasa_acerts(y_test, y_predict_svc)
print(f"Tasa d'encerts SVC lineal: {tasa_svc * 100:.2f}%")
# Calcular diferencia entre y_pred y y_test. Si la diferencia es 0, significa que és unn acert, sino es un erropr.
# Agafam la longitud de les prediccions i ho restam pels errors (Obtenim els acerts)
differences = (y_predict_svc - y_test)
errors = np.count_nonzero(differences)
print(f'Rati d\'acerts en el bloc de predicció: {(len(y_predict_svc)-errors)/len(y_predict_svc)}')
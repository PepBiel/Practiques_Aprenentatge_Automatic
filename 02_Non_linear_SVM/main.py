import sklearn.metrics
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.svm import SVC
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.preprocessing import MinMaxScaler

X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, n_repeated=0,
                           n_classes=2, n_clusters_per_class=1, class_sep=0.5,
                           random_state=8)
# Ja no necessitem canviar les etiquetes, Scikit ho fa per nosaltres

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Els dos algorismes es beneficien d'estandaritzar les dades

scaler = MinMaxScaler() #StandardScaler()
X_transformed = scaler.fit_transform(X_train)
X_test_transformed = scaler.transform(X_test)


def kernel_lineal(x1, x2):
    return x1.dot(x2.T)

gamma = 1.0/(X_transformed.shape[1] * X_transformed.var())

def kernel_gauss(x1, x2):
    return np.exp(-gamma * distance_matrix(x1, x2)**2)

def kernel_poly(x1, x2, degree=3):
    return (gamma + x1.dot(x2.T))**degree


# FEINA 1
svc = SVC(C=1.0, kernel='linear', random_state=33)
svc.fit(X_transformed, y_train, sample_weight=None)
y_pred_svc = svc.predict(X_test_transformed)

svc_meu = SVC(C=1.0, kernel=kernel_lineal, random_state=33)
svc_meu.fit(X_transformed, y_train, sample_weight=None)
y_pred_svc_meu = svc_meu.predict(X_test_transformed)

# Imprimim els resultats
print("Resultats per SVC amb kernel lineal de sklearn:")
svc_prec = sklearn.metrics.precision_score(y_test, y_pred_svc)
print(f"Precisió: {svc_prec:.4f}")

print("Resultats per SVC amb kernel personalitzat:")
svc_prec_meu = sklearn.metrics.precision_score(y_test, y_pred_svc_meu)
print(f"Precisió: {svc_prec_meu:.4f}")

# FEINA 2
svc = SVC(C=1.0, kernel='rbf', random_state=33)
svc.fit(X_transformed, y_train, sample_weight=None)
y_pred_svc = svc.predict(X_test_transformed)

svc_meu = SVC(C=1.0, kernel=kernel_gauss, random_state=33)
svc_meu.fit(X_transformed, y_train, sample_weight=None)
y_pred_svc_meu = svc_meu.predict(X_test_transformed)

# Imprimim els resultats
print("Resultats per SVC amb kernel rbf llibreria Scikit:")
svc_prec = sklearn.metrics.precision_score(y_test, y_pred_svc)
print(f"Precisió: {svc_prec:.4f}")

print("Resultats per SVC amb kernel gaussia personalitzat:")
svc_prec_meu = sklearn.metrics.precision_score(y_test, y_pred_svc_meu)
print(f"Precisió: {svc_prec_meu:.4f}")

#FEINA 3
svc = SVC(C=1.0, kernel='poly', random_state=33)
svc.fit(X_transformed, y_train, sample_weight=None)
y_pred_svc = svc.predict(X_test_transformed)

svc_meu = SVC(C=1.0, kernel=kernel_poly, random_state=33)
svc_meu.fit(X_transformed, y_train, sample_weight=None)
y_pred_svc_meu = svc_meu.predict(X_test_transformed)

# Imprimim els resultats
print("Resultats per SVC amb kernel poly llibreria Scikit:")
svc_prec = sklearn.metrics.precision_score(y_test, y_pred_svc)
print(f"Precisió: {svc_prec:.4f}")

print("Resultats per SVC amb kernel polynomic personalitzat:")
svc_prec_meu = sklearn.metrics.precision_score(y_test, y_pred_svc_meu)
print(f"Precisió: {svc_prec_meu:.4f}")
from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

data = load_iris()
features = data.data
target = data.target

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.scatter(features[:,0], features[:,1], c=target, marker='o', cmap='viridis')

classifier = MLPClassifier(hidden_layer_sizes=(3,8), alpha=1, max_iter=700)
classifier.fit(features, target)
pred = classifier.predict(features)

plt.subplot(2,2,3)
plt.scatter(features[:,0], features[:,1], c=pred, marker='d', cmap='viridis', s=150)
plt.scatter(features[:,0], features[:,1], c=target, marker='o', cmap='viridis', s=15)

pca = PCA(n_components=2, whiten=True, svd_solver='randomized')
pca = pca.fit(features)
pca_features = pca.transform(features)
print('Mantida %5.2f%% da informação do conjunto inicial de dados'%(sum(pca.explained_variance_ratio_)*100))

plt.subplot(2,2,2)
plt.scatter(pca_features[:,0], pca_features[:,1], c=target, marker='o', cmap='viridis')

classifier_pca = MLPClassifier(hidden_layer_sizes=(2,9), alpha=1, max_iter=1000)
classifier_pca.fit(pca_features, target)
pred_pca = classifier_pca.predict(pca_features)

plt.subplot(2,2,4)
plt.scatter(pca_features[:,0], pca_features[:,1], c=pred_pca, marker='d', cmap='viridis', s=150)
plt.scatter(pca_features[:,0], pca_features[:,1], c=target, marker='o', cmap='viridis', s=15)
plt.show()

# Compute confusion matrix for the original data
cm = confusion_matrix(target, pred, normalize='true')
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot()
plt.show()

# Compute confusion matrix for the PCA-transformed data
cm_pca = confusion_matrix(target, pred_pca, normalize='true')
disp_pca = ConfusionMatrixDisplay(confusion_matrix=cm_pca, display_labels=data.target_names)
disp_pca.plot()

plt.show()

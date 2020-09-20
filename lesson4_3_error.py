from sklearn import __version__ as sklearn_version
from distutils.version import LooseVersion
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import numpy as np

iris = datasets.load_iris()
x = iris.data[:, [2,3]]
y = iris.target
x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.3, random_state=1, stratify=y)
# print(np.bincount(y))
# print(np.bincount(y_train))
# print(np.bincount(y_test))

# this is standarded
# sc = StandardScaler()
# sc.fit(x_train)
# x_train_std = sc.transform(x_train)
# x_test_std = sc.transform(x_test)
# this is combined
x_combined = np.vstack((x_train, x_test))
y_combined = np.hstack((y_train, y_test))
# LogisticRegression
def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

    # highlight test samples
    if test_idx:
        # plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidth=1,
                    marker='o',
                    s=100,
                    label='test set')




tree = DecisionTreeClassifier(criterion='error',
                              max_depth=4,
                              random_state=1)
tree.fit(x_train, y_train)


plot_decision_regions(x_combined, y_combined,
                      classifier=tree, test_idx=range(105, 150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
#plt.savefig('images/03_20.png', dpi=300)
plt.show()

# dot_data = export_graphviz(tree,
#                            filled=True,
#                            rounded=True,
#                            class_names=['Setosa',
#                                         'Versicolor',
#                                         'Virginica'],
#                            feature_names=['petal length',
#                                           'petal width'],
#                            out_file=None)
# graph = graph_from_dot_data(dot_data)
# graph.write_png('tree.png')

print(tree.predict_proba(x_test[:45, :]))
y_pred = tree.predict(x_test)
print(y_pred)
print('Accuracy: %.2f' % + accuracy_score(y_test, y_pred))

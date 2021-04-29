from time import time
import logging
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import seaborn as sns
import numpy as np
# %matplotlib inline
class Solution:
    def __init__(self) -> None:
        self.lfw_people = fetch_lfw_people(min_faces_per_person=60)
        print('data loaded')

    def img_data(self):
        n_samples, h, w = self.lfw_people.images.shape
        X = self.lfw_people.data
        n_features = X.shape[1]
        y = self.lfw_people.target
        target_names = self.lfw_people.target_names
        n_classes = target_names.shape[0]
        print("Total dataset size:")
        print("n_samples: %d" % n_samples)
        print("n_features: %d" % n_features)
        print("n_classes: %d" % n_classes)
    
        return X,y,target_names,h,w

    def pca(self, X_train):
        n_components = 150

        print("Extracting the top %d eigenfaces from %d faces"% (n_components, X_train.shape[0]))
        t0 = time()
        pca = PCA(n_components=n_components, svd_solver='randomized',whiten=True)        
        print("done in %0.3fs" % (time() - t0))
        return pca

    def classifier(self):
        print("Fitting the classifier to the training set")
        t0 = time()
        param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5], 'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        svc = SVC(kernel='rbf', class_weight='balanced')
        clf = GridSearchCV(svc, param_grid)
        print("done in %0.3fs" % (time() - t0))
        return clf

    def predict(self, model):
        print("Predicting people's names on the test set")
        t0 = time()
        y_pred = model.predict(X_test)
        print("done in %0.3fs" % (time() - t0))
        return y_pred

    def report(self, y_pred, target_names):
        print(classification_report(y_test, y_pred, target_names=target_names))
        conf_mat = confusion_matrix(y_test, y_pred)
        sns.heatmap(conf_mat/np.sum(conf_mat), fmt='.2%', cmap='Blues', annot=True)
        plt.savefig('snsheatmap')
        # fig.savefig('confusion_matrix.pgf')

def plot_gallery(images, titles, h, w, n_row=6, n_col=4):
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        x = titles[i].split('\n')
        y = x[0].split(':')
        # print(y)
        if y[0] != 'predicted':
            plt.xticks.color = 'Red'
            plt.yticks.color = 'Red'
            plt.axes.labelcolor = 'red'
            # plt.xaxis.label.set_color('red')
            plt.tick_params(axis='x', colors='red')
            print(titles[i])
        plt.xticks(())
        plt.yticks(())
        plt.savefig('faceplot')

# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    # print("\x1b[31m\aabc\x1b[0m")
    if pred_name == true_name : 
        return 'predicted: %s\nTrue:      %s' % (pred_name, true_name)
    else:
        return 'Wrong prediction: %s\nTrue:      %s' % (pred_name, true_name)    


if __name__ == "__main__":
    sol = Solution()
    X,y,target_names,h,w = sol.img_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    pca = sol.pca(X_train)
    svc = sol.classifier()
    model = make_pipeline(pca, svc)
    t0 = time()
    model = model.fit(X_train, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best Estimator: ")
    print(svc.best_estimator_)
    y_pred = sol.predict(model)
    sol.report(y_pred,target_names)
    prediction_titles = [title(y_pred, y_test, target_names, i) for i in range(y_pred.shape[0])]
    plot_gallery(X_test, prediction_titles, h, w)


    
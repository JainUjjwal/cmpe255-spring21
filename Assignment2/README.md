# Assignment 2

SVM classifier to label [famous people's images](https://scikit-learn.org/stable/auto_examples/applications/plot_face_recognition.html#sphx-glr-auto-examples-applications-plot-face-recognition-py).

## Dataset

Used the labeled faces in the [Wild dataset](https://www.kaggle.com/c/labeled-faces-in-the-wild/overview) which consists of several thousand collated photos of the various public figures.


```python
from sklearn.datasets import fetch_lfw_people

def load_data():
    faces = fetch_lfw_people(min_faces_per_person=60)
    print('data loaded')
    print(faces.target_names)
    print(faces.images_shape)
```
### _Note: Running the code locally would only get half the data (4 classes and 820 samples instead of 8 classes and 1348 samples) from sklearn, running the python notebook on google collab obtained the full data and thus gave the output presented in the table. (the plots were all recorded locally and thus reflect accordingly)_
  
## Requirements

Each image contains [62x47] or nearly 3,000 pixels. Use each pixel value as a feature. You will use RandomizedPCA to extract 150 fundamental components to feed into your SVM model as a single pipeline.

```python
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

pca = PCA(n_components=n_components, svd_solver='randomized', whiten=True)
svc = SVC(kernel='rbf', class_weight='balanced')
model = make_pipeline(pca, svc)
```


## Tasks
1. Split the data into a training and testing set.
### Answer
```python
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
```
2. Use a [grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) cross-validation to explore combinations of [parameters](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) to determine the best model: 
   - C: margin hardness E.g. [1, 5, 10, 50]
   - gama: radial basis function kernel E.g. [0.0001, 0.0005, 0.001, 0.005]
 * precision 
 * recall
 * f1-score
 * support
### Answer
```python
best_esitmator = SVC(C=1000.0, break_ties=False, cache_size=200, class_weight='balanced', coef0=0.0, decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf', max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
```
|                  | precision | recall |  f1-score  |  support	|
|------------------|-----------|--------|------------|---------	|
|                  |           |        |            |         	|
| Ariel Sharon     |  0.92     | 0.73   |    0.81    |   15   	|
| Colin Powell     |  0.83     | 0.87   |    0.85    |   68     |
| Donald Rumsfeld  |  0.92     | 0.77   |    0.84    |   31   	|
| George W Bush    |  0.82     | 0.94   |    0.88    |   126    |
| Gerhard Schroeder|  0.94     | 0.74   |    0.83    |   23 	  |
| Hugo Chavez      |  1.00     | 0.65   |    0.79    |   20     |
| Junichiro Koizumi|  1.00     | 0.75   |    0.86    |   12 	  |
| Tony Blair       |  0.88     | 0.88   |    0.88    |   42     |
|                                                           	  |
| accuracy         |           |        |    0.86    |   337    |
| macro avg        | 0.91      | 0.79   |    0.84    |   337    |
| weighted avg     |  0.87     | 0.86   |    0.86    |   337    |

 3. Draw a 4x6 subplots of images using names as label with color black for correct instances and red for incorrect instances.
### Answer
> Filename:  
> faceplot.png
 4. Draw a confusion matrix between features in a [heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) with X-axis of 'Actual' and Y-axis of 'Predicted'.
### Answer
> Filename:  
> snsheatmap.png



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from data import pd
from data import plt

import numpy as nm
import matplotlib.pyplot as mtp
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
from matplotlib.colors import ListedColormap


def run_model_KNN(df):

    # Extracting Independent and dependent Variable
    x = df.drop(['quality'], axis=1)
    y = df['quality']

    # Splitting the dataset into training and test set.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

    # feature Scaling
    st_x = StandardScaler()
    x_train = st_x.fit_transform(x_train)
    x_test = st_x.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 )
    classifier.fit(x_train, y_train)

    y_pred = classifier.predict(x_test)

    #Creating the Confusion matrix
    cm= confusion_matrix(y_test, y_pred)
    print(cm)


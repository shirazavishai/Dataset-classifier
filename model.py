from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import confusion_matrix

def run_model_KNN(df, splitData):
    x_train = splitData[0]
    x_test = splitData[1]
    y_train = splitData[2]
    y_test = splitData[3]

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print("KNN Algorithm")

    K_model = KNeighborsClassifier(n_neighbors=17, p=2)
    K_model.fit(x_train, y_train)

    y_test_pred = K_model.predict(x_test)

    print("Score the X-train with Y-train is : ", K_model.score(x_train, y_train))
    print("Score the X-test  with Y-test  is : ", K_model.score(x_test, y_test))
    print(" Model Evaluation K Neighbors Classifier : accuracy score ", accuracy_score(y_test, y_test_pred))
    print(" Model Evaluation RandomForest Classifier : f1_score score ", f1_score(y_test, y_test_pred,average='weighted'))

    print(confusion_matrix(y_pred=y_test_pred, y_true=y_test))


def run_model_random_forest(df, splitData):
    x_train = splitData[0]
    x_test = splitData[1]
    y_train = splitData[2]
    y_test = splitData[3]

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print("RandomForest Algorithm")

    rfc = RandomForestClassifier()
    rfc.fit(x_train, y_train)
    rfc_pred = rfc.predict(x_test)
    print("Score the X-train with Y-train Random Forest is:", rfc.score(x_train, y_train))
    print("Score the X-test  with Y-test Random Forest is:", rfc.score(x_test, y_test))
    print(" Model Evaluation RandomForest Classifier : accuracy score ", accuracy_score(y_test, rfc_pred))
    print(" Model Evaluation RandomForest Classifier : f1_score score ", f1_score(y_test, rfc_pred,average='weighted'))

    print(confusion_matrix(y_pred=rfc_pred, y_true=y_test))


def run_model_adaboost(df, splitData):
    x_train = splitData[0]
    x_test = splitData[1]
    y_train = splitData[2]
    y_test = splitData[3]

    print("|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
    print("AdaBoost Algorithm")

    abc = AdaBoostClassifier()
    abc.fit(x_train, y_train)
    abc_pred = abc.predict(x_test)
    abc_acc = abc.score(x_test, y_test)
    print("Score the X-test  with Y-test for AdaBoost is:", abc.score(x_train, y_train)*100, "%")
    print("Score the X-test  with Y-test for AdaBoost is:", abc_acc * 100, "%")
    print(" Model Evaluation AdaBoost : accuracy score ", accuracy_score(y_test, abc_pred))
    print(" Model Evaluation AdaBoost : f1_score score ", f1_score(y_test, abc_pred, average='weighted'))

    print(confusion_matrix(y_pred=abc_pred, y_true=y_test))

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC


class Model:
    classifiers = {
    "LogisiticRegression": LogisticRegression(),
    "KNearest": KNeighborsClassifier(),
    "Support Vector Classifier": SVC(),
    "DecisionTreeClassifier": DecisionTreeClassifier()
    }

    def __init__(self,x, y):
        self.x = x 
        self.y = y
        self.model_train = {}

    
    def train(self):
        for key, classifier in Model.classifiers.items():
            classifier.fit(self.x, self.y)
            training_score = cross_val_score(classifier, self.x, self.y, cv=5)
            self.model_train[classifier.__class__.__name__] = classifier
            print("Classifiers: ", classifier.__class__.__name__, "Has a training score of", round(training_score.mean(), 2) * 100, "% accuracy score")
        print('-----------------------------------\n')



    
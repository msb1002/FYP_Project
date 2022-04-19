  #Fitting the regression

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC


class tuning:
    def __init__(self):
        return 
    def svr(self):
        X = np.array(df.drop('Close',axis=1))
        y = np.array(df['Close'])

        train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 1002)

        tuned_parameters = [{'kernel': ['sigmoid','poly', 'rbf'],'C': [1,100,1000]}]

        clf = GridSearchCV(SVR(), tuned_parameters, scoring='accuracy')
        clf.fit(train_X, train_y)

        clf.best_params_

import pandas as pan
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


class PerceptronAnalyzer:
    def __init__(self, train_data_source, test_data_source):
        self.train_data = pan.read_csv(train_data_source, header=None)
        self.test_data = pan.read_csv(test_data_source, header=None)

    def __test_perceptron(self, normalized):
        clf = Perceptron()
        X_train = self.train_data.iloc[:, 1:]
        y_train = self.train_data.iloc[:, 0]
        X_test = self.test_data.iloc[:, 1:]
        y_test = self.test_data.iloc[:, 0]
        if normalized:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        return accuracy_score(y_test, predictions)

    def get_answers(self):
        answers = []
        answers.append(self.__test_perceptron(True) - self.__test_perceptron(False))
        return answers


train_data_source = "../../../resources/week2/perceptron-train.csv"
test_data_source = "../../../resources/week2/perceptron-test.csv"
percenptron_analyzer = PerceptronAnalyzer(train_data_source, test_data_source)

print percenptron_analyzer.get_answers()

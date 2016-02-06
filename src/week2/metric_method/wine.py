from sklearn.cross_validation import KFold
from sklearn.neighbors import KNeighborsClassifier
import sklearn.cross_validation as cv
import sklearn.preprocessing as pp
import numpy as np


class WineAnalyzer:
    # Extract data form dataset
    @staticmethod
    def __extract_data(datasource):
        classes = []
        features = []
        with open(datasource, "r") as f:
            for line in f:
                numbers = line.split(",")
                index = 0
                cur_features = []
                for number in numbers:
                    if index == 0:
                        classes.append(int(number))
                    else:
                        if number[0] == '.':
                            number = '0' + number
                        cur_features.append(float(number))
                    index += 1
                features.append(cur_features)
            f.close()
        return {'classes': classes, 'features': features}

    def __create_Kfold(self):
        self.kfold = KFold(len(self.data['classes']), n_folds=5, shuffle=True, random_state=42)

    def __qualities(self, scale):
        qualities = []
        X = self.data['features']
        if scale:
            X = pp.scale(X)
        y = self.data['classes']
        for k in range(1, 51):
            nclf = KNeighborsClassifier(n_neighbors=k)
            result = cv.cross_val_score(nclf, X, y,cv=self.kfold)
            qualities.append(result.mean())
        return qualities

    @staticmethod
    def __max_element(list):
        max_index = 1
        max_value = list[0]
        index = max_index
        for element in list:
            if element > max_value:
                max_value = element
                max_index = index
            index += 1
        return [max_index, max_value]

    def __init__(self, datasource):
        self.data = self.__extract_data(datasource)
        self.__create_Kfold()
        self.qualities = self.__qualities(False)
        self.qualities_n = self.__qualities(True)
        self.optimal_k = self.__max_element(self.qualities)
        self.optimal_k_n = self.__max_element(self.qualities_n)


datasource = "../../../resources/week2/wine.data"
wine_analyzer = WineAnalyzer(datasource)

print wine_analyzer.optimal_k
print wine_analyzer.optimal_k_n
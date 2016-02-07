import numpy as np
import sklearn.cross_validation as cv
import sklearn.datasets as ds
import sklearn.neighbors as nghb
import sklearn.preprocessing as pp


class BostonAnalyzer:
    def __init__(self):
        self.data_frame = ds.load_boston()
        self.__generate_answers()

    def __calc_best_param(self):
        data_frame = self.data_frame
        scaled_features = pp.scale(data_frame.data)
        steps = np.linspace(1, 10, 200)

        kfold = cv.KFold(len(data_frame.data), n_folds=5, random_state=42, shuffle=True)
        scores = {}
        for step in steps:
            nrgr = nghb.KNeighborsRegressor(n_neighbors=5, weights='distance', p=step)
            score = cv.cross_val_score(nrgr, scaled_features, data_frame.target, cv=kfold,
                                       scoring='mean_squared_error').max()
            scores[step] = score
        best_score = scores.values()[0]
        for key in scores.keys():
            if scores[key] > best_score:
                best_score = score

        return best_score

    def __generate_answers(self):
        self.answers = []
        self.answers.append(self.__calc_best_param())


boston_analyzer = BostonAnalyzer()
print boston_analyzer.answers

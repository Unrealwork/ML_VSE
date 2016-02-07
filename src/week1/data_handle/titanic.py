import math
from collections import Counter

import numpy as np
import pandas as pan
from sklearn.tree import DecisionTreeClassifier


class TitanicSource:
    # get data from csv file
    @staticmethod
    def __calc_survived_percent(data):
        survive_values = data['Survived'].value_counts()
        return survive_values[1] / len(data) * 100

    @staticmethod
    def __calc_first_class_percent(data):
        pclass_data = data['Pclass'].value_counts()
        return pclass_data[1] / len(data) * 100

    @staticmethod
    def __calc_average_age(data):
        list = data['Age'].tolist()
        list = [elem for elem in list if not (math.isnan(elem))]
        return {'average': np.average(list), 'median': np.median(list)}

    @staticmethod
    def __calc_pirson_correlation(data):
        sibsp_list = data['SibSp'].tolist()
        parch_list = data['Parch'].tolist()
        return np.corrcoef(sibsp_list, parch_list)

    @staticmethod
    def __calc_most_popular_name(data):
        female_name_list = data[data.Sex == 'female']['Name'].tolist()

        def get_first_name(name):
            return name.split(' ')[-1]

        first_name_list = []
        for name in female_name_list:
            first_name_list.append(get_first_name(name))
        return Counter(first_name_list).most_common(1)[0]

    @staticmethod
    def __calc_most_important_sign(data):
        filtered_data = data[['Pclass', 'Fare', 'Age', 'Sex']]
        X = filtered_data[~np.isnan(filtered_data['Age'])].as_matrix()
        for row in X:
            if (row[3] == 'male'):
                row[3] = 1
            else:
                row[3] = 0
        y = data[~np.isnan(data['Age'])]['Survived'].as_matrix()
        clf = DecisionTreeClassifier(random_state=241)
        clf.fit(X, y)
        print(clf.feature_importances_)
        return filtered_data

    def __init__(self, csv_path):
        data = pan.read_csv(csv_path)
        self.data = data
        self.sex_values = data['Sex'].value_counts()
        self.survive_percent = self.__calc_survived_percent(self.data)
        self.first_class_percent = self.__calc_first_class_percent(self.data)
        self.average_age = self.__calc_average_age(self.data)
        self.pearson_correlation_sib_parch = self.__calc_pirson_correlation(self.data)
        self.most_popular_female_name = self.__calc_most_popular_name(self.data)
        self.most_important_signs = self.__calc_most_important_sign(self.data)


def generate_answers():
    titanic = TitanicSource("../../../resources/week1/titanic.csv")


generate_answers()

class WineAnalyzer:

    # Extract data form dataset
    @staticmethod
    def __extract_data(data_source):
        classes = []
        features = []
        with open(data_source, "r") as f:
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

    def __init__(self, data_source):
        self.data = self.__extract_data(data_source)


data_source = "../../../resources/week2/wine.data"
wine_analyzer = WineAnalyzer(data_source)

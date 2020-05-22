from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import numpy as np
import linear_regression as lr


def to_np_num(list_of_str):
    temp = np.array(list_of_str)
    ans = temp.astype(np.float)
    return ans

def run(features_file_name, tags_file_name):
    features_file = open(features_file_name, 'r', encoding="utf8")
    lines = features_file.read().splitlines()
    training_inputs = list(map(lambda line: list(line.split(",")), lines))

    tags_file = open(tags_file_name, 'r', encoding="utf8")
    labels = tags_file.read().splitlines()

    train, test, train_tags, test_tags = model_selection.train_test_split(training_inputs, labels, test_size=0.3)
    num_of_features = len(training_inputs[0])

    train = np.array(list(map((lambda line: to_np_num(line)), train)))
    test = np.array(list(map((lambda line: to_np_num(line)), test)))


    Encoder = LabelEncoder()
    train_tags = Encoder.fit_transform(train_tags)
    test_tags = Encoder.fit_transform(test_tags)

    linearRegression = lr.linearRegression(num_of_features)
    linearRegression.train(train, np.array(train_tags))
    predictions_lr = list()
    for vec in test:
        p = linearRegression.predict(vec)
        predictions_lr.append(p)
    sum = 0
    for index, pred in enumerate(predictions_lr):
        diff = abs(pred - test_tags[index])
        if (test_tags[index] != 0):
            sum += diff / test_tags[index]
    print(100 - sum/predictions_lr.__len__())

run("features/mars.csv","tags/mars.csv");
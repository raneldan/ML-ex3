from sklearn import model_selection
from sklearn.metrics import accuracy_score
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
        if p > 0.5: p=1
        else: p=0;
        predictions_lr.append(p)

    return accuracy_score(predictions_lr, test_tags) * 100
    # print("Perceptron Accuracy Score -> ",accuracy_score(predictions_perceptron, test_tags)*100)

sms_features_file_name = "old/features.csv"
sms_tags_file_name = "old/tags.csv"

email_features_file_name = "old/featuresEmail.csv"
email_tags_file_name = "old/tagsEmail.csv"

ans = run(sms_features_file_name,sms_tags_file_name)
print(ans)
#Then accuracy for sms in linear regression is - 88.57655502392345
#Then accuracy for sms in linear perceptron is - 97

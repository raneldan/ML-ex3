import csv
from typing import List
from rowToVector import RowToVector

filenameSms = "SMSSpamCollection"
filenameEmail = "spam_ham_dataset.csv"


def create_files(ignore_features: List[int] = []):
    create_email_files(ignore_features)
    create_sms_files()


def create_email_files(ignore_features: List[int]):
    with open(filenameEmail, 'r', encoding="utf8") as csvin, \
            open('tagsEmail.csv', 'w', newline='', encoding='utf-8') as csvoutForTags, \
            open('dataEmail.csv', 'w', newline='', encoding='utf-8') as csvoutForData, \
            open('featuresEmail.csv', 'w', newline='', encoding='utf-8') as csvoutForFeatures:
        csvin = csv.reader(csvin, delimiter=',')
        csvoutForTags = csv.writer(csvoutForTags)
        csvoutForData = csv.writer(csvoutForData)
        csvoutForFeatures = csv.writer(csvoutForFeatures)
        x = 0
        for row in csvin:
            if (x == 71):
                x = 71
            x += 1
            csvoutForTags.writerow([row[1]])
            csvoutForData.writerow([row[2]])
            unfiltered_feature_vec = RowToVector([row[2]], is_email=True).vector
            feature_vec = []
            for index, item in enumerate(unfiltered_feature_vec):
                if index not in ignore_features:
                    feature_vec.append(unfiltered_feature_vec[index])
            csvoutForFeatures.writerow(feature_vec)


def create_sms_files():
    with open(filenameSms, 'r', encoding="utf8") as csvin, \
            open('tags.csv', 'w', newline='', encoding='utf-8') as csvoutForTags, \
            open('data.csv', 'w', newline='', encoding='utf-8') as csvoutForData, \
            open('features.csv', 'w', newline='', encoding='utf-8') as csvoutForFeatures:
        csvin = csv.reader(csvin, delimiter='\t')
        csvoutForTags = csv.writer(csvoutForTags)
        csvoutForData = csv.writer(csvoutForData)
        csvoutForFeatures = csv.writer(csvoutForFeatures)
        for row in csvin:
            csvoutForTags.writerow([row[0]])
            csvoutForData.writerow(row[1:])
            csvoutForFeatures.writerow(RowToVector(row[1:]).vector)

create_sms_files()
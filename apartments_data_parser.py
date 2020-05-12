import csv
import os

apartments_file_name = "data/kc_house_data.csv"
apartments_tags_file_name = "tags\\apartments.csv"
apartments_features_file_name = "features\\apartments.csv"


def create_files():
    if os.path.exists(apartments_tags_file_name):
        os.remove(apartments_tags_file_name)
    if os.path.exists(apartments_features_file_name):
        os.remove(apartments_features_file_name)
    create_apartments_files()


def create_apartments_files():
    with open(apartments_file_name, 'r', encoding="utf8") as csvin, \
            open(apartments_tags_file_name, 'w', newline='', encoding='utf-8') as csvoutForTags, \
            open(apartments_features_file_name, 'w', newline='', encoding='utf-8') as csvoutForFeatures:
        csvin = csv.reader(csvin, delimiter=',')
        csvoutForTags = csv.writer(csvoutForTags)
        csvoutForFeatures = csv.writer(csvoutForFeatures)
        itercsvin = iter(csvin)
        next(itercsvin)
        for row in itercsvin:
            price = float(row[2])
            csvoutForTags.writerow([price])
            csvoutForFeatures.writerow(row[3:])

create_files()
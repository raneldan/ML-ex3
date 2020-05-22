import csv
import os

mars_file_name = "data/martians_data.csv"
mars_tags_file_name = "tags\\mars.csv"
mars_features_file_name = "features\\mars.csv"


def create_files():
    if os.path.exists(mars_tags_file_name):
        os.remove(mars_tags_file_name)
    if os.path.exists(mars_features_file_name):
        os.remove(mars_features_file_name)
    create_mars_files()


def create_mars_files():
    with open(mars_file_name, 'r', encoding="utf8") as csvin, \
            open(mars_tags_file_name, 'w', newline='', encoding='utf-8') as csvoutForTags, \
            open(mars_features_file_name, 'w', newline='', encoding='utf-8') as csvoutForFeatures:
        csvin = csv.reader(csvin, delimiter=',')
        csvoutForTags = csv.writer(csvoutForTags)
        csvoutForFeatures = csv.writer(csvoutForFeatures)
        itercsvin = iter(csvin)
        next(itercsvin)
        for row in itercsvin:
            weight = float(row[2])
            age = float(row[1])
            csvoutForTags.writerow([weight])
            csvoutForFeatures.writerow([age])

create_files()
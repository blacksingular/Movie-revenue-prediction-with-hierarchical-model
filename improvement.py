import pandas as pd
import csv
import ast
import re
import numpy as np
from sklearn import preprocessing
import pdb
from collections import defaultdict

imdb_path = './tables/movie_info.txt'


class PreProcessing:
    def __init__(self):
        self.men_dict = dict()
        self.train_X = None
        self.train_y = None

    def data_cleaning(self):
        params = ["Title", "Year", "Director", "Writer", "Actors", "Genre", "Language", "Country", "Runtime", "BoxOffice"]
        valid_csv_file = open('./new_tables/omdb_full_valid.csv', 'w', newline='')
        train_csv_file = open('./new_tables/omdb_full_train.csv', 'w', newline='')
        test_csv_file = open('./new_tables/omdb_full_test.csv', 'w', newline='')

        valid_writer = csv.writer(valid_csv_file)
        train_writer = csv.writer(train_csv_file)
        test_writer = csv.writer(test_csv_file)
        valid_writer.writerow(params)
        train_writer.writerow(params)
        test_writer.writerow(params)
        with open(imdb_path, 'r') as f:
            for line in f:
                tmp = []

                # open txt and get valid data and split it into three parts
                meta_dict = ast.literal_eval(line[line.find("\t") + 1:])
                condition = [meta_dict[p] != 'N/A' for p in params]
                if meta_dict['Type'] == 'movie' and 2008 <= int(meta_dict['Year']) <= 2019 and all(condition):
                    for p in params:
                        tmp.append(meta_dict.get(p, ""))
                    # print(tmp[-1])
                    # pdb.set_trace()
                    if tmp[-1].find(".") != -1: tmp[-1] = tmp[-1][:tmp[-1].find(".")]
                    tmp[-1] = ''.join(re.findall(r'\d+', tmp[-1]))  # remove "$"s and "million"s
                    tmp[-2] = tmp[-2][:tmp[-2].find(" ")]  # remove "min"s
                    for i in range(5, 8):
                        if tmp[i].find(",") != -1: tmp[i] = tmp[i][:tmp[i].find(",")]  # genre first
                        if tmp[i].find("(") != -1: tmp[i] = tmp[i][:tmp[i].find("(")]  # country first
                    if 2008 <= int(meta_dict['Year']) <= 2013:
                        train_writer.writerow(tmp)
                    elif 2013 < int(meta_dict['Year']) <= 2015:
                        valid_writer.writerow(tmp)
                    else:
                        test_writer.writerow(tmp)
        valid_csv_file.close()
        train_csv_file.close()
        test_csv_file.close()
        # clean duplicates
        df = pd.read_csv('./new_tables/omdb_full_valid.csv')
        df = df.drop_duplicates(subset='Title', keep='last')
        df.to_csv('./new_tables/omdb_full_valid.csv')
        df = pd.read_csv('./new_tables/omdb_full_train.csv')
        df = df.drop_duplicates(subset='Title', keep='last')
        df.to_csv('./new_tables/omdb_full_train.csv')
        df = pd.read_csv('./new_tables/omdb_full_test.csv')
        df = df.drop_duplicates(subset='Title', keep='last')
        df.to_csv('./new_tables/omdb_full_test.csv')
        # test the uniqueness
        # df = pd.read_csv('./new_tables/omdb_full_train.csv')
        # print(df.Title.describe())


    def numerical(self):
        params = ["Year", "Runtime", "BoxOffice"]
        train_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_train.csv"))
        year = np.array(list(map(float, train_df['Year'].dropna()))).reshape(-1, 1)
        boxoffice = np.array(list(map(float, train_df['BoxOffice'].dropna()))).reshape(-1, 1)
        runtime = np.array(list(map(float, train_df['Runtime'].dropna()))).reshape(-1, 1)
        self.train_X = np.c_[year, runtime]
        print(self.train_X.shape)
        self.train_y = boxoffice

    def one_hot(self):
        params = ["Genre", "Language", "Country"]

        def transform(string):
            train_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_train.csv"))
            enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
            # append genre, country and language to the feature after encoding
            self.train_X = np.c_[self.train_X, enc.fit_transform(np.array(train_df[string].dropna()).reshape(-1, 1)).toarray()]

        for p in params:
            transform(p)

    def men_representation(self):
        params = ["Director", "Writer", "Actors"]
        train_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_train.csv"))
        for p in params:
            # loop for directors, writers, actors
            men = []
            for t, revenue in zip(train_df[p], train_df['BoxOffice']):
                try:
                    for m in t.split(', '):
                        men.append((re.sub(r'\([^)]*\)', '', m), revenue))
                except AttributeError:
                    print(t)
            for man, revenue in men:
                if man not in self.men_dict:
                    self.men_dict[man] = [revenue]
                elif revenue not in self.men_dict[man]:
                    self.men_dict[man].append(revenue)
                else:
                    continue
        print(self.men_dict['Robert Downey Jr.'])
        for k, v in self.men_dict.copy().items():
            self.men_dict[k] = sum(v) / len(v)
        print(sorted(self.men_dict.items(), key=lambda x: x[1], reverse=True))

    def evaluation(self):
        error_smape_train = 0
        error_mae_train = 0
        error_smape_test = 0
        error_mae_test = 0
        # train_loss
        for i in range(len(self.train_y)):
            error_smape_train += abs(self.train_y[i] - self.y_pre_train[i]) * 2 / (
                        self.train_y[i] + abs(self.y_pre_train[i]))
            error_mae_train += abs(self.train_y[i] - self.y_pre_train[i])
        for i in range(len(self.test_y)):
            error_smape_test += abs(self.test_y[i] - self.y_pre_test[i]) * 2 / (self.test_y[i] + abs(self.y_pre_test[i]))
            error_mae_test += abs(self.test_y[i] - self.y_pre_test[i])
            # pdb.set_trace()
        error_smape_train /= len(self.y_pre_train)
        error_mae_train /= len(self.y_pre_train)
        error_smape_test /= len(self.y_pre_test)
        error_mae_test /= len(self.y_pre_test)
        print("train loss are: ", error_smape_train, error_mae_train)
        print("test loss are: ", error_smape_test, error_mae_test)
        return str(error_smape_train) + ' ' + str(error_mae_train) + ' ' + str(error_smape_test) + ' ' + str(error_mae_test)


if __name__ == "__main__":
    Pre = PreProcessing()
    # Pre.data_cleaning()
    Pre.numerical()
    Pre.one_hot()
    Pre.men_representation()
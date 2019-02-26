import pandas as pd
import csv
import ast
import re
import numpy as np
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LoR
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import pdb
from collections import defaultdict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

imdb_path = './tables/movie_info.txt'


class PreProcessing:
    def __init__(self):
        self.men_dict = dict()
        self.train_X = None
        self.train_y = None
        self.valid_X = None
        self.valid_y = None

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

    def men_representation(self):
        params = ["Director", "Writer", "Actors"]
        train_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_train.csv"))
        for p in params:
            # loop for directors, writers, actors
            men = []
            for t, revenue in zip(train_df[p], train_df['BoxOffice']):
                try:
                    for m in t.split(', '):
                        men.append((re.sub(r'\([^)]*\)', '', m).strip(), revenue))
                except AttributeError:
                    print(t)
            for man, revenue in men:
                if man not in self.men_dict:
                    self.men_dict[man] = [revenue]
                elif revenue not in self.men_dict[man]:
                    self.men_dict[man].append(revenue)
                else:
                    continue
        for k, v in self.men_dict.copy().items():
            self.men_dict[k] = sum(v) / len(v)  # average revenue to represent men
            # self.men_dict[k] = v[-1]  # use the latest revenue to represent men
        return self.train_X, self.train_y, self.valid_X, self.valid_y

    def numerical(self):
        params = ["Year", "Runtime", "BoxOffice"]
        train_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_train.csv"))
        valid_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_valid.csv"))
        year = np.array(list(map(float, train_df['Year'].dropna()))).reshape(-1, 1)
        boxoffice = np.array(list(map(float, train_df['BoxOffice'].dropna()))).reshape(-1, 1)
        runtime = np.array(list(map(float, train_df['Runtime'].dropna()))).reshape(-1, 1)
        self.train_X = np.c_[year, runtime]
        self.train_y = boxoffice
        year = np.array(list(map(float, valid_df['Year'].dropna()))).reshape(-1, 1)
        boxoffice = np.array(list(map(float, valid_df['BoxOffice'].dropna()))).reshape(-1, 1)
        runtime = np.array(list(map(float, valid_df['Runtime'].dropna()))).reshape(-1, 1)
        self.valid_X = np.c_[year, runtime]
        self.valid_y = boxoffice

    def one_hot(self):
        params = ["Genre", "Language", "Country"]

        def transform(string):
            train_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_train.csv"))
            valid_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_valid.csv"))
            enc = preprocessing.OneHotEncoder(handle_unknown='ignore')
            enc.fit(np.array(train_df[string].dropna()).reshape(-1, 1))
            # append genre, country and language to the feature after encoding
            self.train_X = np.c_[self.train_X, enc.transform(np.array(train_df[string].dropna()).reshape(-1, 1)).toarray()]
            self.valid_X = np.c_[self.valid_X, enc.transform(np.array(valid_df[string].dropna()).reshape(-1, 1)).toarray()]

        for p in params:
            transform(p)

    def categorical(self):
        params = ["Director", "Writer", "Actors"]
        weights = [10, 10, 1, 1]  # weights assigned to every actor
        with open('./result.txt', 'a+') as f:
            f.writelines(list(map(str, weights)))
        avg = sum(self.men_dict.values()) / len(self.men_dict)  # average revenue of all men, for unseen data
        train_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_train.csv"))
        valid_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_valid.csv"))

        for p in params:
            men = []

        # get vector of all params in train
            for t in train_df[p]:
                tmp = []
                for m in t.split(', '):
                    tmp.append(self.men_dict[re.sub(r'\([^)]*\)', '', m).strip()])
                men.append(sum([x * w for x, w in zip(tmp, weights)]))
            men = np.array(men).reshape(-1, 1)
            scaler = preprocessing.StandardScaler().fit(men)
            self.train_X = np.c_[self.train_X, scaler.transform(men)]

            men = []
        # get vector of all params in valid
            for t in valid_df[p]:
                tmp = []
                for m in t.split(', '):
                    tmp.append(self.men_dict[re.sub(r'\([^)]*\)', '', m).strip()] if re.sub(r'\([^)]*\)', '', m).strip() in self.men_dict else avg)
                men.append(sum([x * w for x, w in zip(tmp, weights)]))
            men = np.array(men).reshape(-1, 1)
            self.valid_X = np.c_[self.valid_X, scaler.transform(men)]
        return self.train_X, self.train_y, self.valid_X, self.valid_y


class Regression:
    def __init__(self, train_X, train_y, valid_X, valid_y):
        self.train_X = train_X
        self.train_y = train_y
        self.valid_X = valid_X
        self.valid_y = valid_y

    def shuffle(self):
        data = np.c_[self.train_X, self.train_y]
        np.random.shuffle(data)
        self.train_X = data[:, :-1]
        self.train_y = data[:, -1]

    def GBDT(self, n, step):
        best_params = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'huber'}
        params = {'n_estimators': n, 'max_depth': step, 'min_samples_split': 2,
                  'learning_rate': 0.01, 'loss': 'huber'}
        model = GBR(**params)
        model.fit(self.train_X, self.train_y)
        w = model.get_params()
        self.y_pre_train = model.predict(self.train_X)
        self.y_pre_valid = model.predict(self.valid_X)
        return w
        # gsearch1 = GridSearchCV(estimator=model)
        # gsearch1.fit(self.train_X, self.train_y)
        # print(gsearch1.best_params_, gsearch1.best_score_)
    
    def line_regression(self):
        model = LR()
        model.fit(self.train_X, self.train_y)
        # importance = lr.feature_importances_
        # print(importance)
        # exit()
        self.y_pre_valid = model.predict(self.valid_X)
        self.y_pre_train = model.predict(self.train_X)
        
    def logistic_regression(self, c):
        model = LoR(C=c)
        # print(self.train_X)
        model.fit(self.train_X, self.train_y)
        self.y_pre_valid = model.predict(self.valid_X)
        self.y_pre_train = model.predict(self.train_X)
        # return self.y_pre_test, self.y_pre_train

    def RandomForest_regression(self):
        model = RFR(n_estimators=1000, max_depth=10)
        model.fit(self.train_X, self.train_y)
        path = model.decision_path(self.train_X)
        self.y_pre_train = model.predict(self.train_X)
        self.y_pre_valid = model.predict(self.valid_X)
        # print(path)
        
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
        for i in range(len(self.valid_y)):
            error_smape_test += abs(self.valid_y[i] - self.y_pre_valid[i]) * 2 / (self.valid_y[i] + abs(self.y_pre_valid[i]))
            error_mae_test += abs(self.valid_y[i] - self.y_pre_valid[i])
            # pdb.set_trace()
        error_smape_train /= len(self.y_pre_train)
        error_mae_train /= len(self.y_pre_train)
        error_smape_test /= len(self.y_pre_valid)
        error_mae_test /= len(self.y_pre_valid)
        print("train loss are: ", error_smape_train, error_mae_train)
        print("test loss are: ", error_smape_test, error_mae_test)
        with open('./result.txt', 'a+') as f:
            f.writelines(list(map(str, [error_smape_train, error_mae_train, error_smape_test, error_mae_test])))
        return str(error_smape_train) + ' ' + str(error_mae_train) + ' ' + str(error_smape_test) + ' ' + str(error_mae_test)

    def plot(self):
        plt.figure()
        plt.plot(self.train_y, self.y_pre_train, 'ro')
        plt.plot(np.linspace(0, 1e9, 100), np.linspace(0, 1e9, 100))
        plt.plot(self.valid_y, self.y_pre_valid, 'bx')
        plt.show()

if __name__ == "__main__":
    Pre = PreProcessing()
    # Pre.data_cleaning()
    Pre.numerical()
    Pre.one_hot()
    Pre.men_representation()
    train_X, train_y, valid_X, valid_y = Pre.categorical()
    
    Reg = Regression(train_X, train_y, valid_X, valid_y)
    Reg.shuffle()
    # Reg.line_regression()
    w = Reg.GBDT(1000, 10)
    # Reg.logistic_regression(1)
    # Reg.RandomForest_regression()
    Reg.plot()
    Reg.evaluation()
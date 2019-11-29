import pandas as pd
import csv
import ast
import re
import numpy as np
import pickle as pkl
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LoR
from sklearn.svm import SVC, SVR
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns;

sns.set(color_codes=True)
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

imdb_path = './tables/movie_info.txt'
week_11 = True
week_12 = False
med = 6960000


class PreProcessing:
    def __init__(self, mode='all'):
        self.men_dict = dict()
        self.train_X = None
        self.train_y = None
        self.valid_X = None
        self.valid_y = None
        self.train_df = None
        self.valid_df = None
        self.test_df = None
        self.mode = mode

    def data_cleaning(self):
        params = ["Title", "Year", "Director", "Writer", "Actors", "Genre", "Language", "Country", "Runtime",
                  "BoxOffice"]
        valid_csv_file = open('./new_tables/omdb_full_valid.csv', 'w', newline='')
        train_csv_file = open('./new_tables/omdb_full_train.csv', 'w', newline='')
        test_csv_file = open('./new_tables/omdb_full_test.csv', 'w', newline='')
        all_csv_file = open('./new_tables/omdb_full.csv', 'w', newline='')
        writer = csv.writer(all_csv_file)
        valid_writer = csv.writer(valid_csv_file)
        train_writer = csv.writer(train_csv_file)
        test_writer = csv.writer(test_csv_file)
        writer.writerow(params)
        valid_writer.writerow(params)
        train_writer.writerow(params)
        test_writer.writerow(params)
        with open(imdb_path, 'r') as f:
            for line in f:
                tmp = []

                # open txt and get valid data and split it into three parts
                meta_dict = ast.literal_eval(line[line.find("\t") + 1:])
                if self.mode == 'us':
                    condition = [meta_dict[p] != 'N/A' for p in params] + [meta_dict['Country'] == 'USA']
                elif self.mode == 'all':
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
                    writer.writerow(tmp)
                    if 2008 <= int(meta_dict['Year']) <= 2013:
                        train_writer.writerow(tmp)
                    elif 2013 < int(meta_dict['Year']) <= 2015:
                        valid_writer.writerow(tmp)
                    else:
                        test_writer.writerow(tmp)
        valid_csv_file.close()
        train_csv_file.close()
        test_csv_file.close()
        all_csv_file.close()
        # clean duplicates
        df = pd.read_csv('./new_tables/omdb_full_valid.csv')
        df = df.drop_duplicates(subset='Title', keep='last')
        df.to_csv('./new_tables/omdb_full_valid.csv')
        df = pd.read_csv('./new_tables/omdb_full.csv')
        df = df.drop_duplicates(subset='Title', keep='last')
        df.to_csv('./new_tables/omdb_full.csv')
        df = pd.read_csv('./new_tables/omdb_full_train.csv')
        df = df.drop_duplicates(subset='Title', keep='last')
        df.to_csv('./new_tables/omdb_full_train.csv')
        df = pd.read_csv('./new_tables/omdb_full_test.csv')
        df = df.drop_duplicates(subset='Title', keep='last')
        df.to_csv('./new_tables/omdb_full_test.csv')
        # test the uniqueness
        # df = pd.read_csv('./new_tables/omdb_full.csv')
        # print(df.Title.describe())
        # exit()

    def men_representation(self):
        params = ["Director", "Writer", "Actors"]
        data_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full.csv"))
        self.train_df = data_df[:int(0.6 * len(data_df))]
        self.valid_df = data_df[int(0.6 * len(data_df)):int(0.8 * len(data_df))]
        self.test_df = data_df[int(0.8 * len(data_df)):]
        train_df = self.train_df
        # exit()
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

    def men_representation_old(self):
        params = ["Director", "Writer", "Actors"]
        train_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_train.csv"))
        valid_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_valid.csv"))
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
            if 0:
                men = []
                for t, revenue in zip(valid_df[p], valid_df['BoxOffice']):
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
            # self.men_dict[k] = sum(v) / len(v)  # average revenue to represent men
            # self.men_dict[k] = v[-1]  # use the latest revenue to represent men
            self.men_dict[k] = np.median(np.array(v))
        return self.train_X, self.train_y, self.valid_X, self.valid_y

    def numerical(self):
        params = ["Year", "Runtime", "BoxOffice"]
        train_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_train.csv"))
        valid_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_valid.csv"))
        # train_df = self.train_df
        # valid_df = self.valid_df
        year = np.array(list(map(float, train_df['Year'].dropna()))).reshape(-1, 1)
        boxoffice = np.array(list(map(float, train_df['BoxOffice'].dropna()))).reshape(-1, 1)
        # mm = preprocessing.MinMaxScaler()
        runtime = np.array(list(map(float, train_df['Runtime'].dropna()))).reshape(-1, 1)
        self.train_X = np.c_[year, runtime]
        self.train_y_c = train_df['Label']
        self.train_y_log = np.log(boxoffice)
        self.train_y_log10 = np.log10(boxoffice)
        self.train_y = boxoffice
        year = np.array(list(map(float, valid_df['Year'].dropna()))).reshape(-1, 1)
        boxoffice = np.array(list(map(float, valid_df['BoxOffice'].dropna()))).reshape(-1, 1)
        runtime = np.array(list(map(float, valid_df['Runtime'].dropna()))).reshape(-1, 1)
        self.valid_X = np.c_[year, runtime]
        self.valid_y = boxoffice
        self.valid_y_c = valid_df['Label']
        self.valid_y_log = np.log(boxoffice)
        self.valid_y_log10 = np.log10(boxoffice)
        self.valid_y = boxoffice
        # return mm

    def one_hot(self):
        params = ["Genre", "Language", "Country"]

        def transform(string):
            train_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_train.csv"))
            valid_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_valid.csv"))
            # train_df = self.train_df
            # valid_df = self.valid_df
            enc = preprocessing.LabelEncoder()
            enc.fit(np.r_[
                        np.array(train_df[string].dropna()).reshape(-1, 1), np.array(valid_df[string].dropna()).reshape(
                            -1, 1)])
            # print(list(enc.classes_))
            # pdb.set_trace()
            # append genre, country and language to the feature after encoding
            self.train_X = np.c_[self.train_X, enc.transform(np.array(train_df[string].dropna()).reshape(-1, 1))]
            self.valid_X = np.c_[self.valid_X, enc.transform(np.array(valid_df[string].dropna()).reshape(-1, 1))]
            return enc

        encs = []
        for p in params:
            encs.append(transform(p))
        return self.train_X, self.valid_X, encs

    def categorical(self):
        params = ["Director", "Writer", "Actors"]
        weights = [1, 1, 1, 1]  # weights assigned to every actor
        with open('./result.txt', 'a+') as f:
            f.writelines(list(map(str, weights)))

        avg = sum(self.men_dict.values()) / len(self.men_dict)  # average revenue of all men, for unseen data
        train_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_train.csv"))
        valid_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_valid.csv"))
        # train_df = self.train_df
        # valid_df = self.valid_df
        for p in params:
            men = []

            # get vector of all params in train
            for t in train_df[p]:
                tmp = []
                for m in t.split(', '):
                    tmp.append(self.men_dict[re.sub(r'\([^)]*\)', '', m).strip()])
                # tmp = [x * w for x, w in zip(tmp, weights)]
                # men.append(tmp)
                men.append(sum(tmp) / len(tmp))
            men = np.array(men).reshape(-1, 1)
            # mm = preprocessing.MinMaxScaler()
            self.train_X = np.c_[self.train_X, np.log(men)]

            men = []
            # get vector of all params in valid
            for t in valid_df[p]:
                found = True
                tmp = []
                if week_11:
                    for m in t.split(', ')[:2]:
                        if re.sub(r'\([^)]*\)', '', m).strip() in self.men_dict:
                            tmp.append(self.men_dict[re.sub(r'\([^)]*\)', '', m).strip()])
                        else:
                            found = False
                            break
                    if found:
                        men.append(sum(tmp) / len(tmp))
                    else:
                        men.append(1)
                if week_12:
                    found = True
                    for m in t.split(', ')[:2]:
                        if re.sub(r'\([^)]*\)', '', m).strip() in self.men_dict:
                            tmp.append(self.men_dict[re.sub(r'\([^)]*\)', '', m).strip()])
                        else:
                            tmp.append(med)
                            found = False
                    if found:
                        men.append(sum(tmp) / len(tmp))
                    else:
                        men.append(-sum(tmp) / len(tmp))
            men = np.array(men).reshape(-1, 1)

            self.valid_X = np.c_[self.valid_X, np.log(men)]

        # # try to embedding unseen data using other information in valid
        # men = []
        # for row in valid_df[['Actors', 'Director', 'Writer']].iterrows():
        #     tmp = []  # store every single person
        #     meta = []  # store all person
        #     writer = row[1]['Writer']
        #     actor = row[1]['Actors']
        #     director = row[1]['Director']
        #     for m in writer.split(', '):
        #         tmp.append(self.men_dict[re.sub(r'\([^)]*\)', '', m).strip()] if re.sub(r'\([^)]*\)', '', m).strip() in self.men_dict else 0)
        #     meta.append(tmp)
        #     tmp = []
        #     for m in actor.split(', '):
        #         tmp.append(self.men_dict[re.sub(r'\([^)]*\)', '', m).strip()] if re.sub(r'\([^)]*\)', '', m).strip() in self.men_dict else 0)
        #     meta.append(tmp)
        #     tmp = []
        #     for m in director.split(', '):
        #         tmp.append(self.men_dict[re.sub(r'\([^)]*\)', '', m).strip()] if re.sub(r'\([^)]*\)', '', m).strip() in self.men_dict else 0)
        #     meta.append(tmp)
        #     alter = 0
        #     valid = 0
        #     max_stuff = 0
        #     max_mode = False
        #     avg_mode = True
        #     # find the average(max) of all valid stuff
        #     for cate in meta:
        #         for p in cate:
        #             if avg_mode:
        #                 if p != 0:
        #                     alter += p
        #                     valid += 1
        #             if max_mode:
        #                 alter = max(alter, p)
        #
        #     if alter == 0:
        #         meta = [avg] * 3
        #         men.append(meta)
        #         continue
        #
        #         # alter invalid stuff with average calculated
        #     meta = [[p if p != 0 else alter for p in cate] for cate in meta]
        #     meta = [sum(cate) / len(cate) for cate in meta]
        #     men.append(meta)
        #
        # men = np.array(men)
        # self.valid_X = np.c_[self.valid_X, np.log(men)]
        print(self.train_y_c.shape, self.train_y.shape)
        return self.train_X, np.c_[self.train_y_c, self.train_y, self.train_y_log, self.train_y_log10], self.valid_X, \
               np.c_[self.valid_y_c, self.valid_y, self.valid_y_log, self.valid_y_log10], self.men_dict


class Classification:
    def __init__(self, train_x, train_y, valid_x, valid_y, easy_train_X, easy_valid_X):
        self.train_X = train_x
        self.train_y = train_y
        self.valid_X = valid_x
        self.valid_y = valid_y
        self.easy_train_X = easy_train_X
        self.easy_valid_X = easy_valid_X
        self.ref = None
        self.easy_ref = None

    def shuffle(self):
        data = np.c_[self.train_X, self.easy_train_X, self.train_y]
        np.random.shuffle(data)
        self.train_X = data[:, :8]
        self.easy_train_X = data[:, 8:-4]
        self.train_y_c = data[:, -4]
        self.train_y_log = data[:, -3:]
        self.valid_y_c = self.valid_y[:, 0]
        self.valid_y_log = self.valid_y[:, 1:]

    def RandomForest_classification(self):
        model = RFC()
        easy_model = RFC()
        model.fit(self.train_X, self.train_y_c)
        easy_model.fit(self.easy_train_X, self.train_y_c)

        # path = model.decision_path(self.train_X)
        # self.y_pre_train = model.predict(self.train_X)
        # self.y_pre_valid = model.predict(self.valid_X)
        valid_X = []
        easy_valid_X = []
        valid_y = []
        easy_valid_y = []
        for com, easy, label in zip(list(self.valid_X), list(self.easy_valid_X), list(self.valid_y)):
            if 0 in com[-3:]:
                easy_valid_X.append(easy)
                easy_valid_y.append(label)
            else:
                valid_X.append(com)
                valid_y.append(label)
        self.easy_valid_X = np.array(easy_valid_X)
        self.valid_X = np.array(valid_X)
        self.valid_y = np.array(valid_y)
        self.easy_valid_y = np.array(easy_valid_y)
        print(self.easy_valid_y.shape)
        # print(valid_X.shape, easy_valid_X.shape, valid_y.shape, easy_valid_y.shape)

        print('roc score of complex on train:', roc_auc_score(self.train_y_c, model.predict(self.train_X)))
        print('roc score of easy on train:', roc_auc_score(self.train_y_c, easy_model.predict(self.easy_train_X)))
        print('roc score of complex on valid:', roc_auc_score(self.valid_y[:, 0], model.predict(valid_X)))
        print('roc score of easy on valid:', roc_auc_score(self.easy_valid_y[:, 0], easy_model.predict(easy_valid_X)))
        print('accuracy of complex on train:', model.score(self.train_X, self.train_y_c))
        print('accuracy of easy on train:', easy_model.score(self.easy_train_X, self.train_y_c))
        print('accuracy of complex on valid:', model.score(valid_X, self.valid_y[:, 0]))
        print('accuracy of easy on valid:', easy_model.score(easy_valid_X, self.easy_valid_y[:, 0]))

        self.ref = model.predict(valid_X)
        self.easy_ref = easy_model.predict(easy_valid_X)
        return model, easy_model

    def split(self):
        self.valid_X_low = []
        self.valid_X_high = []
        self.valid_y_low = []
        self.valid_y_high = []
        self.train_X_low = []
        self.train_X_high = []
        self.train_y_low = []
        self.train_y_high = []
        self.etrain_X_low = []
        self.etrain_y_low = []
        self.etrain_X_high = []
        self.etrain_y_high = []
        self.evalid_X_low = []
        self.evalid_y_low = []
        self.evalid_X_high = []
        self.evalid_y_high = []
        for x, y, label in zip(self.easy_train_X, self.train_y_log, self.train_y_c):
            if label == 0:
                self.etrain_X_low.append(x)
                self.etrain_y_low.append(y)
            else:
                self.etrain_X_high.append(x)
                self.etrain_y_high.append(y)
        for x, y, label in zip(self.train_X, self.train_y_log, self.train_y_c):
            if label == 0:
                self.train_X_low.append(x)
                self.train_y_low.append(y)
            else:
                self.train_X_high.append(x)
                self.train_y_high.append(y)
        for x, y, label in zip(self.easy_valid_X, self.easy_valid_y, self.easy_ref):
            if label == 0:
                self.evalid_X_low.append(x)
                self.evalid_y_low.append(y[1:])
            else:
                self.evalid_X_high.append(x)
                self.evalid_y_high.append(y[1:])
        for x, y, label in zip(self.valid_X, self.valid_y, self.ref):
            if label == 0:
                self.valid_X_low.append(x)
                self.valid_y_low.append(y[1:])
            else:
                self.valid_X_high.append(x)
                self.valid_y_high.append(y[1:])
        return self.valid_X_low, self.valid_X_high, self.valid_y_low, self.valid_y_high, \
               self.train_X_low, self.train_X_high, self.train_y_low, self.train_y_high, \
               self.evalid_X_low, self.evalid_X_high, self.evalid_y_low, self.evalid_y_high, \
               self.etrain_X_low, self.etrain_X_high, self.etrain_y_low, self.etrain_y_high

    def SVM(self):
        model = SVC()
        model.fit(self.train_X, self.train_y)
        # path = model.decision_path(self.train_X)
        # self.y_pre_train = model.predict(self.train_X)
        # self.y_pre_valid = model.predict(self.valid_X)
        print(roc_auc_score(self.train_y, model.predict(self.train_X)))
        print(roc_auc_score(self.valid_y, model.predict(self.valid_X)))
        print(model.score(self.train_X, self.train_y))
        print(model.score(self.valid_X, self.valid_y))


class ClassificationNew(Classification):
    def __init__(self, train_x, train_y, valid_x, valid_y, easy_train_X, easy_valid_X):
        super().__init__(train_x, train_y, valid_x, valid_y, easy_train_X, easy_valid_X)
        self._valid_x_complete_high = []
        self._valid_x_complete_low = []
        self._valid_x_missing_high = []
        self._valid_x_missing_low = []
        self._valid_y_complete_high = []
        self._valid_y_complete_low = []
        self._valid_y_missing_high = []
        self._valid_y_missing_low = []
        self._train_x_complete_high = []
        self._train_x_complete_low = []
        self._train_x_missing_high = []
        self._train_x_missing_low = []
        self._train_y_low = []
        self._train_y_high = []
        self.x_for_cls = self.valid_X[:]

    def RandomForest_classification(self):
        model = RFC()
        model.fit(self.train_X, self.train_y_c)
        # print(valid_X.shape, easy_valid_X.shape, valid_y.shape, easy_valid_y.shape)

        print('roc score of complex on train:', roc_auc_score(self.train_y_c, model.predict(self.train_X)))
        print('roc score of complex on valid:', roc_auc_score(self.valid_y[:, 0], model.predict(valid_X)))
        print('accuracy of complex on train:', model.score(self.train_X, self.train_y_c))
        print('accuracy of complex on valid:', model.score(valid_X, self.valid_y[:, 0]))

        self.ref = model.predict(valid_X)

    def split(self):
        for x, y, label in zip(self.train_X, self.train_y_log, self.train_y_c):
            if label == 1:
                self._train_x_complete_high.append(x)
                self._train_x_missing_high.append(x[:5])
                self._train_y_high.append(y)
            else:
                self._train_x_complete_low.append(x)
                self._train_x_missing_low.append(x[:5])
                self._train_y_low.append(y)

        for x, y, label in zip(self.valid_X, self.valid_y, self.ref):
            if label == 1:
                for item in x:
                    if item < 0:
                        self._valid_x_missing_high.append([abs(v) for v in x[:5]])
                        self._valid_y_missing_high.append(y[1:])
                self._valid_x_complete_high.append(x)
                self._valid_y_complete_high.append(y[1:])
            else:
                for item in x:
                    if item < 0:
                        self._valid_x_missing_low.append([abs(v) for v in x[:5]])
                        self._valid_y_missing_low.append(y[1:])
                self._valid_x_complete_low.append(x)
                self._valid_y_complete_low.append(y[1:])
        return np.c_[np.array(self._train_x_complete_high), np.array(self._train_y_high)], \
               np.c_[np.array(self._train_x_missing_high), np.array(self._train_y_high)], \
               np.c_[np.array(self._train_x_complete_low), np.array(self._train_y_low)], \
               np.c_[np.array(self._train_x_missing_low), np.array(self._train_y_low)], \
               np.c_[np.array(self._valid_x_complete_high), np.array(self._valid_y_complete_high)], \
               np.c_[np.array(self._valid_x_complete_low), np.array(self._valid_y_complete_low)], \
               np.c_[np.array(self._valid_x_missing_high), np.array(self._valid_y_missing_high)], \
               np.c_[np.array(self._valid_x_missing_low), np.array(self._valid_y_missing_low)]


class Regression:
    def __init__(self, vxl, vxh, vyl, vyh, txl, txh, tyl, tyh, men_info, mode):
        self.vxl = np.array(vxl)
        self.vxh = np.array(vxh)
        self.vyl = np.array(vyl)
        self.vyh = np.array(vyh)
        self.txl = np.array(txl)
        self.txh = np.array(txh)
        self.tyl = np.array(tyl)
        self.tyh = np.array(tyh)
        self.men_dict = men_info
        self.mode = mode

    def shuffle_2(self):
        data = np.r_[np.c_[self.train_X, self.train_y], np.c_[self.valid_X, self.valid_y]]
        l = len(data)
        np.random.shuffle(data)
        self.train_X = data[:int(0.75 * l), :-1]
        self.train_y = data[:int(0.75 * l), -1]
        self.valid_X = data[int(0.75 * l):, :-1]
        self.valid_y = data[int(0.75 * l):, -1]

    def shuffle(self):
        data = np.c_[self.txl, self.tyl]
        np.random.shuffle(data)
        self.txl = data[:, :-3]
        self.tyllog = data[:, -1]
        self.tyl = data[:, -3]
        data = np.c_[self.txh, self.tyh]
        np.random.shuffle(data)
        self.txh = data[:, :-3]
        self.tyhlog = data[:, -2]
        self.tyh = data[:, -3]
        self.vyllog = self.vyl[:, -1]
        self.vyl = self.vyl[:, -3]
        self.vyhlog = self.vyh[:, -2]
        self.vyh = self.vyh[:, -3]

    def GBDT(self, n, step):
        print(len(self.vxl), len(self.vxh))
        best_params = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 2,
                       'learning_rate': 0.01, 'loss': 'huber'}
        params_high = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 2,
                       'learning_rate': 0.01, 'loss': 'huber'}
        if self.mode == 'complex':
            model_low = GBR(**params_high)
        else:
            model_low = GBR()
        # model = GBR()
        param_test1 = {'max_depth': range(3, 6, 3)}
        model_low.fit(self.txl, self.tyllog)
        self.y_pre_train_log = model_low.predict(self.txl).reshape(-1, 1)
        self.y_pre_train = [10 ** x for x in model_low.predict(self.txl).reshape(-1, 1)]
        self.y_pre_valid_log = model_low.predict(self.vxl).reshape(-1, 1)
        self.y_pre_valid = [10 ** x for x in model_low.predict(self.vxl).reshape(-1, 1)]

        if self.mode == 'complex':
            model_high = GBR(**params_high)
        else:
            model_high = GBR()
        # # model = GBR()
        # param_test1 = {'max_depth': range(3, 6, 3)}
        model_high.fit(self.txh, self.tyhlog)
        self.y_pre_train_log = np.r_[self.y_pre_train_log, model_high.predict(self.txh).reshape(-1, 1)]
        self.y_pre_train = np.r_[self.y_pre_train, np.exp(model_high.predict(self.txh).reshape(-1, 1))]
        self.y_pre_valid_log = np.r_[self.y_pre_valid_log, model_high.predict(self.vxh).reshape(-1, 1)]
        return model_high
        # self.y_pre_valid = np.r_[self.y_pre_valid, np.exp(model_high.predict(self.vxh).reshape(-1, 1))]
        # self.y_pre_train_log = model_high.predict(self.txh).reshape(-1, 1)
        # self.y_pre_train = np.exp(model_high.predict(self.txh).reshape(-1, 1))
        # self.y_pre_valid_log = model_high.predict(self.vxh).reshape(-1, 1)
        # self.y_pre_valid = np.exp(model_high.predict(self.vxh).reshape(-1, 1))
        if 0:
            print(np.argsort(model_low.feature_importances_))
            x = ['Language', 'Year', 'Country', 'Genre', 'Runtime', 'Actor', 'Director', 'Writer']
            y = sorted(model_low.feature_importances_)
            plt.bar(x, y)
            plt.title('Feature Importance')
            plt.xlabel('Feature')
            plt.ylabel('Importance')
            plt.show()
        # gsearch1 = GridSearchCV(estimator=model, param_grid=param_test1)
        # gsearch1.fit(self.train_X, self.train_y)
        # print(gsearch1.best_params_, gsearch1.best_score_)

    def linear_regression(self):
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
        flag = 0
        # train_loss
        # print(self.tyl[0], self.tyllog[0], self.tyhlog[0], self.tyh[0], self.vyl[0], self.vyllog[0], self.vyhlog[0], self.vyh[0])
        for i in range(len(self.y_pre_train)):
            error_smape_train += abs(np.r_[self.tyl, self.tyh][i] - self.y_pre_train[i]) * 2 / (
                    np.r_[self.tyl, self.tyh][i] + abs(self.y_pre_train[i]))
            error_mae_train += abs(np.r_[self.tyl, self.tyh][i] - self.y_pre_train[i])
            # error_smape_train += abs(self.tyl[i] - self.y_pre_train[i]) * 2 / (
            #             self.tyl[i] + abs(self.y_pre_train[i]))
            # error_mae_train += abs(self.tyl[i] - self.y_pre_train[i])
        for i in range(len(self.y_pre_valid)):
            error_smape_test += abs(np.r_[self.vyl, self.vyh][i] - self.y_pre_valid[i]) * 2 / (
                    np.r_[self.vyl, self.vyh][i] + abs(self.y_pre_valid[i]))
            error_mae_test += abs(np.r_[self.vyl, self.vyh][i] - self.y_pre_valid[i])
            # error_smape_test += abs(self.vyl[i] - self.y_pre_valid[i]) * 2 / (
            #         self.vyl[i] + abs(self.y_pre_valid[i]))
            # error_mae_test += abs(self.vyl[i] - self.y_pre_valid[i])
            # pdb.set_trace()
        error_smape_train /= len(self.y_pre_train)
        error_mae_train /= len(self.y_pre_train)
        error_smape_test /= len(self.y_pre_valid)
        error_mae_test /= len(self.y_pre_valid)
        print("train loss are: ", error_smape_train, error_mae_train)
        print("test loss are: ", error_smape_test, error_mae_test)
        with open('./result.txt', 'a+') as f:
            f.writelines(list(map(str, [error_smape_train, error_mae_train, error_smape_test, error_mae_test])))
        return [np.r_[self.tyllog, self.tyhlog], self.y_pre_train_log], [np.r_[self.vyllog, self.vyhlog],
                                                                         self.y_pre_valid_log], \
               [error_mae_train, error_smape_train, error_mae_test, error_smape_test, len(self.y_pre_valid)]

    def plot(self):
        plt.figure()
        print(self.tyllog.shape, self.tyhlog.shape, self.y_pre_train_log.shape)
        plt.plot(np.r_[self.tyllog, self.tyhlog], self.y_pre_train_log, 'ro')
        plt.plot(np.linspace(0, 25, 100), np.linspace(0, 25, 100))
        plt.plot(np.r_[self.vyllog, self.vyhlog], self.y_pre_valid_log, 'bx')
        # plt.plot(self.tyllog, self.y_pre_train_log, 'ro')
        # plt.plot(np.linspace(0, 25, 100), np.linspace(0, 25, 100))
        # plt.plot(self.vyllog, self.y_pre_valid_log, 'bx')
        plt.xlabel('y_true')
        plt.ylabel('y_pre')
        plt.title('Relation between labels and predictions')
        plt.show()
        # sns.regplot(x=self.train_X[:, 7], y=self.train_y_log.reshape(-1, ), x_estimator=np.mean, robust=True)
        # plt.title('Actor')
        # plt.xlabel('Actor')
        # plt.ylabel('boxoffice')
        # plt.show()
        # sns.regplot(x=np.r_[self.train_X[:, 1], self.valid_X[:, 1]], y=np.r_[self.train_y_log.reshape(-1, ), self.valid_y_log.reshape(-1, )], robust=True)
        # plt.title('runtime')
        # plt.xlabel('runtime')
        # plt.ylabel('boxoffice')
        # plt.show()

    def storePkl(self):
        with open('./new_tables/men.pkl', 'wb') as f:
            pkl.dump(self.men_dict, f)


class RegressionNew:
    def __init__(self, th, tl, vh, vl):
        print(vl.shape)
        self.vxl = np.array(vl[:, :-3])
        self.vxh = np.array(vh[:, :-3])
        self.vyl = np.array(vl[:, -3:])
        self.vyh = np.array(vh[:, -3:])
        self.txl = np.array(tl[:, :-3])
        self.txh = np.array(th[:, :-3])
        self.tyl = np.array(tl[:, -3:])
        self.tyh = np.array(th[:, -3:])

    def shuffle(self):
        data = np.c_[self.txl, self.tyl]
        np.random.shuffle(data)
        self.txl = data[:, :-3]
        self.tyllog = data[:, -1]
        self.tyl = data[:, -3]
        data = np.c_[self.txh, self.tyh]
        np.random.shuffle(data)
        self.txh = data[:, :-3]
        self.tyhlog = data[:, -2]
        self.tyh = data[:, -3]
        self.vyllog = self.vyl[:, -1]
        self.vyl = self.vyl[:, -3]
        self.vyhlog = self.vyh[:, -2]
        self.vyh = self.vyh[:, -3]

    def GBDT(self, n, step):
        best_params = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 2,
                       'learning_rate': 0.01, 'loss': 'huber'}
        params_high = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 2,
                       'learning_rate': 0.01, 'loss': 'huber'}
        model_low = GBR()
        print(self.txl.shape, self.tyllog.shape, self.vxl.shape)
        model_low.fit(self.txl, self.tyllog)
        self.y_pre_train_log = model_low.predict(self.txl).reshape(-1, 1)
        self.y_pre_train = [10 ** x for x in model_low.predict(self.txl).reshape(-1, 1)]
        self.y_pre_valid_log = model_low.predict(self.vxl).reshape(-1, 1)
        self.y_pre_valid = [10 ** x for x in model_low.predict(self.vxl).reshape(-1, 1)]

        model_high = GBR()
        model_high.fit(self.txh, self.tyhlog)
        self.y_pre_train_log = np.r_[self.y_pre_train_log, model_high.predict(self.txh).reshape(-1, 1)]
        self.y_pre_train = np.r_[self.y_pre_train, np.exp(model_high.predict(self.txh).reshape(-1, 1))]
        self.y_pre_valid_log = np.r_[self.y_pre_valid_log, model_high.predict(self.txh).reshape(-1, 1)]
        self.y_pre_valid = np.r_[self.y_pre_valid, np.exp(model_high.predict(self.vxh).reshape(-1, 1))]
        # self.y_pre_train_log = model_high.predict(self.txh).reshape(-1, 1)
        # self.y_pre_train = np.exp(model_high.predict(self.txh).reshape(-1, 1))
        # self.y_pre_valid_log = model_high.predict(self.vxh).reshape(-1, 1)
        # self.y_pre_valid = np.exp(model_high.predict(self.vxh).reshape(-1, 1))

    def linear_regression(self):
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
        flag = 0
        # train_loss
        # print(self.tyl[0], self.tyllog[0], self.tyhlog[0], self.tyh[0], self.vyl[0], self.vyllog[0], self.vyhlog[0], self.vyh[0])
        for i in range(len(self.y_pre_train)):
            error_smape_train += abs(np.r_[self.tyl, self.tyh][i] - self.y_pre_train[i]) * 2 / (
                    np.r_[self.tyl, self.tyh][i] + abs(self.y_pre_train[i]))
            error_mae_train += abs(np.r_[self.tyl, self.tyh][i] - self.y_pre_train[i])
            # error_smape_train += abs(self.tyl[i] - self.y_pre_train[i]) * 2 / (
            #         self.tyl[i] + abs(self.y_pre_train[i]))
            # error_mae_train += abs(self.tyl[i] - self.y_pre_train[i])
        for i in range(len(self.y_pre_valid)):
            error_smape_test += abs(np.r_[self.vyl, self.vyh][i] - self.y_pre_valid[i]) * 2 / (
                    np.r_[self.vyl, self.vyh][i] + abs(self.y_pre_valid[i]))
            error_mae_test += abs(np.r_[self.vyl, self.vyh][i] - self.y_pre_valid[i])
            # error_smape_test += abs(self.vyl[i] - self.y_pre_valid[i]) * 2 / (
            #         self.vyl[i] + abs(self.y_pre_valid[i]))
            # error_mae_test += abs(self.vyl[i] - self.y_pre_valid[i])
            # pdb.set_trace()
        error_smape_train /= len(self.y_pre_train)
        error_mae_train /= len(self.y_pre_train)
        error_smape_test /= len(self.y_pre_valid)
        error_mae_test /= len(self.y_pre_valid)
        print("train loss are: ", error_smape_train, error_mae_train)
        print("test loss are: ", error_smape_test, error_mae_test)
        with open('./result.txt', 'a+') as f:
            f.writelines(list(map(str, [error_smape_train, error_mae_train, error_smape_test, error_mae_test])))
        return [np.r_[self.tyllog, self.tyhlog], self.y_pre_train_log], [np.r_[self.vyllog, self.vyhlog],
                                                                         self.y_pre_valid_log], \
               [error_mae_train, error_smape_train, error_mae_test, error_smape_test, len(self.y_pre_valid)]

    def plot(self):
        plt.figure()
        plt.plot(np.r_[self.tyllog, self.tyhlog], self.y_pre_train_log, 'ro')
        plt.plot(np.linspace(0, 25, 100), np.linspace(0, 25, 100))
        plt.plot(np.r_[self.vyllog, self.vyhlog], self.y_pre_valid_log, 'bx')
        # plt.plot(self.tyllog, self.y_pre_train_log, 'ro')
        # plt.plot(np.linspace(0, 25, 100), np.linspace(0, 25, 100))
        # plt.plot(self.vyllog, self.y_pre_valid_log, 'bx')
        plt.xlabel('y_true')
        plt.ylabel('y_pre')
        plt.title('Relation between labels and predictions')
        plt.show()


if __name__ == "__main__":
    Pre = PreProcessing()
    # Pre.data_cleaning()
    Pre.men_representation_old()
    Pre.numerical()
    easy_train_X, easy_valid_X, enc = Pre.one_hot()
    train_X, train_y, valid_X, valid_y, men_info = Pre.categorical()

    if week_11:
        Cla = Classification(train_X, train_y, valid_X, valid_y, easy_train_X, easy_valid_X)
        Cla.shuffle()
        model, easy_model = Cla.RandomForest_classification()
        vxl, vxh, vyl, vyh, txl, txh, tyl, tyh, evxl, evxh, evyl, evyh, etxl, etxh, etyl, etyh = Cla.split()

        Reg = Regression(vxl, vxh, vyl, vyh, txl, txh, tyl, tyh, men_info, mode='complex')
        Reg.shuffle()
        # Reg.linear_regression()
        model_high = Reg.GBDT(1000, 10)
        # Reg.logistic_regression(1)
        # Reg.RandomForest_regression()
        Reg.plot()
        Train_para, Valid_para, Eval = Reg.evaluation()

        eReg = Regression(evxl, evxh, evyl, evyh, etxl, etxh, etyl, etyh, men_info, mode='simple')
        eReg.shuffle()
        eReg.GBDT(1000, 10)
        eReg.plot()
        eTrain_para, eValid_para, eEval = eReg.evaluation()

        print('mae_train:', (eEval[0] + Eval[0]) / 2)
        print('smape_train:', (Eval[1] + eEval[1]) / 2)
        print('mae_valid:', (eEval[2] * eEval[-1] + Eval[2] * Eval[-1]) / (eEval[-1] + Eval[-1]))
        print('smape_valid:', (eEval[3] * eEval[-1] + Eval[3] * Eval[-1]) / (eEval[-1] + Eval[-1]))

        # predict revenue for captain marvel
        one_hot = np.r_[enc[0].transform(['Action']), enc[1].transform(['English']), enc[2].transform(['USA'])]

        men = []
        directors = ['Anna Boden', 'Ryan Fleck']
        tmp = []
        for director in directors:
            if director in men_info:
                tmp.append(men_info[director])
        men.append(sum(tmp) / len(tmp))

        writers = ['Anna Boden', 'Ryan Fleck']
        tmp = []
        for writer in writers:
            if writer in men_info:
                tmp.append(men_info[writer])
        men.append(sum(tmp) / len(tmp))

        actors = ['Brie Larson', 'Samuel L. Jackson']
        tmp = []
        for actor in actors:
            if actor in men_info:
                tmp.append(men_info[actor])
        men.append(sum(tmp) / len(tmp))
        captain_marvel = np.r_[[2019, 124], one_hot, men].reshape(1, -1)
        print(np.exp(model_high.predict(captain_marvel)))

        # predict revenue for infinity war
        one_hot = np.r_[enc[0].transform(['Action']), enc[1].transform(['English']), enc[2].transform(['USA'])]

        men = []

        writers = ['Christopher Markus', 'Stephen McFeely']
        tmp = []
        for writer in writers:
            if writer in men_info:
                tmp.append(men_info[writer])
            else:
                print(writer)
                tmp.append(med)
        men.append(sum(tmp) / len(tmp))

        actors = ['Robert Downey Jr.', 'Chris Hemsworth']
        tmp = []
        for actor in actors:
            if actor in men_info:
                tmp.append(men_info[actor])
            else:
                print(actor)
                tmp.append(med)
        men.append(sum(tmp) / len(tmp))
        infinity_war = np.r_[[2018, 149], one_hot, [259766572], men].reshape(1, -1)
        print(np.exp(model_high.predict(infinity_war)))

        # predict revenue for fantastic beasts
        one_hot = np.r_[enc[0].transform(['Adventure']), enc[1].transform(['English']), enc[2].transform(['UK'])]

        men = []

        directors = ['David Yates']
        tmp = []
        for director in directors:
            if director in men_info:
                tmp.append(men_info[director])
        men.append(sum(tmp) / len(tmp))
        writers = ['J.K. Rowling']
        tmp = []
        for writer in writers:
            if writer in men_info:
                tmp.append(men_info[writer])
            else:
                print(writer)
                tmp.append(med)
        men.append(sum(tmp) / len(tmp))

        actors = ['Eddie Redmayne', 'Sam Redford']
        tmp = []
        for actor in actors:
            if actor in men_info:
                tmp.append(men_info[actor])
            else:
                print(actor)
                tmp.append(med)
        men.append(sum(tmp) / len(tmp))
        infinity_war = np.r_[[2016, 133], one_hot, men].reshape(1, -1)
        print(np.exp(model_high.predict(infinity_war)))


        def plot(Train_para, Valid_para, eTrain_para, eValid_para):
            plt.figure()
            plt.plot(Train_para[0], Train_para[1], 'ro')
            plt.plot(eTrain_para[0], eTrain_para[1], 'ro')
            plt.plot(np.linspace(0, 25, 100), np.linspace(0, 25, 100))
            plt.plot(Valid_para[0], Valid_para[1], 'bx')
            plt.plot(eValid_para[0], eValid_para[1], 'bx')
            plt.xlabel('y_true')
            plt.ylabel('y_pre')
            plt.title('Relation between labels and predictions')
            plt.show()


        plot(Train_para, Valid_para, eTrain_para, eValid_para)

    if week_12:
        NCla = ClassificationNew(train_X, train_y, valid_X, valid_y, easy_train_X, easy_valid_X)
        NCla.shuffle()
        NCla.RandomForest_classification()
        tch, tmh, tcl, tml, vch, vcl, vmh, vml = NCla.split()
        print('complete shapes are', tch.shape, tcl.shape, vch.shape, vcl.shape)
        print('missing shapes are', tmh.shape, tml.shape, vmh.shape, vml.shape)
        complete_reg = RegressionNew(tch, tcl, vch, vcl)
        complete_reg.shuffle()
        # Reg.linear_regression()
        complete_reg.GBDT(1000, 10)
        # Reg.logistic_regression(1)
        # Reg.RandomForest_regression()
        # complete_reg.plot()
        complete_train_para, complete_valid_para, complete_eval = complete_reg.evaluation()

        missing_reg = RegressionNew(tmh, tml, vmh, vml)
        missing_reg.shuffle()
        missing_reg.GBDT(1000, 10)
        missing_train_para, missing_valid_para, missing_eval = missing_reg.evaluation()

        print('mae_train:', (missing_eval[0] + complete_eval[0]) / 2)
        print('smape_train:', (missing_eval[1] + complete_eval[1]) / 2)
        print('mae_valid:', (missing_eval[2] * complete_eval[-1] + complete_eval[2] * missing_eval[-1]) / (
                    complete_eval[-1] + missing_eval[-1]))
        print('smape_valid:', (missing_eval[3] * complete_eval[-1] + complete_eval[3] * missing_eval[-1]) / (
                    complete_eval[-1] + missing_eval[-1]))


        def plot(Train_para, Valid_para, eTrain_para, eValid_para):
            plt.figure()
            plt.plot(Train_para[0], Train_para[1], 'ro')
            plt.plot(eTrain_para[0], eTrain_para[1], 'ro')
            plt.plot(np.linspace(0, 25, 100), np.linspace(0, 25, 100))
            plt.plot(Valid_para[0], Valid_para[1], 'bx')
            plt.plot(eValid_para[0], eValid_para[1], 'bx')
            plt.xlabel('y_true')
            plt.ylabel('y_pre')
            plt.title('Relation between labels and predictions')
            plt.show()


        plot(complete_train_para, complete_valid_para, missing_train_para, missing_valid_para)


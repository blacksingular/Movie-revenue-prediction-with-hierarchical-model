import pickle as pkl
import pandas as pd
import collections
import numpy as np
import re
import os
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn import mixture
from sklearn import preprocessing
import warnings

warnings.filterwarnings('ignore')



years = range(2008, 2019)  # all of the valid years
complex_params = ["Director", "Writer", "Actors"]
easy_params = ["Year", "Runtime"]
categorical_params = ["Genre", "Language", "Country"]


def split_by_years():
    try: raw_df = pd.read_csv('./new_tables/omdb_full.csv')  # raw data
    except FileNotFoundError: print('File not exist')
    # split data by year and restore in csv format
    for year in years:
        raw_df[raw_df['Year'] == year].to_csv('./data/' + str(year) + '.csv')


def men_presentation():
    params = ["Director", "Writer", "Actors"]
    men_dict = collections.defaultdict(list)

    # form men information dict for every year
    for year in years:
        try: train_df = pd.read_csv('./data/' + str(year) + '.csv')
        except FileNotFoundError: print('File not found')
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
                # print(men_dict[man])
                if revenue not in men_dict[man]:
                    men_dict[man].append(revenue)
        prefix = men_dict.copy()
        for k, v in men_dict.items():
            prefix[k] = sum(v)
        with open('./pkls/' + str(year) + '_men.pkl', 'wb') as f:
            pkl.dump(prefix, f)


def form_feature():
    # restore easy features in pickle format
    easy = collections.defaultdict(list)
    complex_ = collections.defaultdict(list)
    for year in years[:-1]:
        df = pd.read_csv('./data/' + str(year + 1) + '.csv')
        with open('./pkls/' + str(year) + '_men.pkl', 'rb') as f:
            men_dict = pkl.load(f)
        empty = True
        for param in easy_params:
            if empty:
                value = np.c_[np.log10(df['BoxOffice'].values), df[param].values]
                empty = False
            else:
                value = np.c_[value, df[param].values]
        print(len(men_dict))

        encoders = {}
        for p in categorical_params:
            enc = preprocessing.LabelEncoder()
            enc.fit(np.array(df[p].dropna()).reshape(-1, 1))
            # append genre, country and language to the feature after encoding
            value = np.c_[value, enc.transform(np.array(df[p].dropna()).reshape(-1, 1))]
            encoders[p] = enc
        with open('./pkls/' + str(year + 1) + '_encoder.pkl', 'wb') as f:
            pkl.dump(encoders, f)

        def complex(idx):
            for person in complex_params:
                for p in df[person][idx].split(', ')[:2]:
                    if re.sub(r'\([^)]*\)', '', p).strip() not in men_dict:
                        return False
            return True

        print('original length of data', len(value))
        complete_idx = []  # idx of movie whose information is complete
        for i in range(len(value)):
            if complex(i):
                complete_idx.append(i)
        complex_feature = np.array([value[x] for x in complete_idx])
        for person in complex_params:
            complex_writers = np.ndarray(0)
            for idx in complete_idx:
                tmp = 0
                flag = 0
                for p in df[person][idx].split(', ')[:2]:
                    tmp += men_dict[re.sub(r'\([^)]*\)', '', p).strip()]
                    flag += 1
                tmp //= flag
                complex_writers = np.r_[complex_writers, tmp]
            complex_feature = np.c_[complex_feature, complex_writers]
        # print(complex_feature)
        value = np.delete(value, complete_idx, 0)
        print('after deletion', len(value))
        easy[year + 1] = list(value)
        complex_[year + 1] = complex_feature
        with open('./pkls/' + str(year + 1) + '_easy_feature.pkl', 'wb') as f:
            pkl.dump(easy, f)
        with open('./pkls/' + str(year + 1) + '_complex_feature.pkl', 'wb') as f:
            pkl.dump(complex_, f)


def store_classification_model():
    _years = range(2010, 2019)
    for year in _years:
        with open('./pkls/' + str(year - 1) + '_easy_feature.pkl', 'rb') as f:
            _easy = pkl.load(f)
        with open('./pkls/' + str(year - 1) + '_complex_feature.pkl', 'rb') as f:
            _complex = pkl.load(f)
        easy_data = np.ndarray(0)
        for k in _easy:
            if len(easy_data) > 0:
                easy_data = np.r_[easy_data, np.array(_easy[k])]
            else:
                easy_data = np.array(_easy[k])
        print('length of easy data for classification is:', len(easy_data))
        np.random.shuffle(easy_data)
        easy_test = easy_data[:, 0]
        easy_train = easy_data[:, 1:]
        model = mixture.GaussianMixture(n_components=2)
        model.fit(easy_test.reshape(-1, 1))
        easy_test = model.predict(easy_test.reshape(-1, 1))

        complex_data = np.ndarray(0)
        for k in _complex:
            if len(complex_data) > 0:
                complex_data = np.r_[complex_data, np.array(_complex[k])]
            else:
                complex_data = np.array(_complex[k])
        print('length of complex data for classification is:', len(complex_data))
        np.random.shuffle(complex_data)
        complex_test = complex_data[:, 0]
        complex_train = complex_data[:, 1:]
        model = mixture.GaussianMixture(n_components=2)
        model.fit(complex_test.reshape(-1, 1))
        complex_test = model.predict(complex_test.reshape(-1, 1))

        model = RFC()
        easy_model = RFC()
        model.fit(complex_train, complex_test)
        easy_model.fit(easy_train, easy_test)
        with open('./pkls/' + str(year) + '_easy_class.pkl', 'wb') as f:
            pkl.dump(easy_model, f)
        with open('./pkls/' + str(year) + '_complex_class.pkl', 'wb') as f:
            pkl.dump(model, f)
        # print(len(complex_train))


def store_regression_model():
    _years = range(2010, 2019)

    for year in _years:
        with open('./pkls/' + str(year - 1) + '_easy_feature.pkl', 'rb') as f:
            _easy = pkl.load(f)
        with open('./pkls/' + str(year - 1) + '_complex_feature.pkl', 'rb') as f:
            _complex = pkl.load(f)
        with open('./pkls/' + str(year) + '_easy_class.pkl', 'rb') as f:
            _easy_class = pkl.load(f)
        with open('./pkls/' + str(year) + '_complex_class.pkl', 'rb') as f:
            _complex_class = pkl.load(f)

        easy_data_one = np.ndarray(0)
        easy_data_zero = np.ndarray(0)
        complex_data_one = np.ndarray(0)
        complex_data_zero = np.ndarray(0)

        def split(model, data, t1, t2, path):
            for k in data:
                # print(len(data[k]))
                for row in data[k]:
                    if model.predict(row[1:].reshape(1, -1)):
                        # print(row[1:].reshape(1, -1))
                        if len(t1) > 0:
                            # print(t1.shape)
                            t1 = np.r_[t1, np.array(row).reshape(1, -1)]
                        else:
                            t1 = np.array(row).reshape(1, -1)
                    else:
                        if len(t2) > 0:
                            # print(t2.shape)
                            t2 = np.r_[t2, np.array(row).reshape(1, -1)]
                        else:
                            t2 = np.array(row).reshape(1, -1)
            print(len(t1) + len(t2))
            if len(t1) > 0:
                np.random.shuffle(t1)
                test1 = t1[:, 0]
                train1 = t1[:, 1:]
                params_high = {'n_estimators': 1000, 'max_depth': 10, 'min_samples_split': 2,
                               'learning_rate': 0.01, 'loss': 'huber'}
                # one_model = GBR(**params_high)
                one_model = RFR()
                one_model.fit(train1, test1.T)
                with open('./pkls/' + str(year) + path + '_1.pkl', 'wb') as f:
                    pkl.dump(one_model, f)
            if len(t2) > 0:
                np.random.shuffle(t2)
                test2 = t2[:, 0]
                train2 = t2[:, 1:]
                # zero_model = GBR(**params_high)
                zero_model = RFR()
                zero_model.fit(train2, test2.T)
                with open('./pkls/' + str(year) + path + '_0.pkl', 'wb') as f:
                    pkl.dump(zero_model, f)
        split(_easy_class, _easy, easy_data_one, easy_data_zero, '_easy_regression')
        split(_complex_class, _complex, complex_data_one, complex_data_zero, '_complex_regression')
        print(len(easy_data_zero) + len(easy_data_one))

def test():
    year = 2014
    mode = '_complex'
    num = 20
    with open('./pkls/' + str(year) + mode + '_feature.pkl', 'rb') as f:
        features = pkl.load(f)
    with open('./pkls/' + str(year) + mode + '_class.pkl', 'rb') as f:
        class_model = pkl.load(f)
    print('true: ', features[year][num])
    with open('./pkls/' + str(year) + mode + '_regression_' + str(class_model.predict(features[year][num][1:].reshape(1, -1))[0]) + '.pkl', 'rb') as f:
        print('prediction: ', np.exp(pkl.load(f).predict(features[year][num][1:].reshape(1, -1))))

def GUI(year=2018, directors=['David Yates'], writers=['J.K. Rowling'], actors = ['Eddie Redmayne', 'Katherine Waterston'], genre='Action', language='English', country='UK', runtime=134):

    with open('./pkls/' + str(min(year, 2018)) + '_men.pkl', 'rb') as f:
        men_info = pkl.load(f)
    with open('./pkls/' + str(min(year, 2018)) + '_encoder.pkl', 'rb') as f:
        enc = pkl.load(f)

    one_hot = np.r_[enc['Genre'].transform([genre]), enc['Language'].transform([language]), enc['Country'].transform([country])]
    mode = '_complex'
    men = []
    tmp = []
    for director in directors:
        if director in men_info:
            tmp.append(men_info[director])
        else:
            mode = '_easy'
    men.append(sum(tmp) / len(tmp))

    tmp = []
    for writer in writers:
        if writer in men_info:
            tmp.append(men_info[writer])
        else:
            mode = '_easy'
    men.append(sum(tmp) / len(tmp))

    tmp = []
    for actor in actors:
        if actor in men_info:
            tmp.append(men_info[actor])
        else:
            mode = '_easy'
    men.append(sum(tmp) / len(tmp))
    with open('./pkls/' + str(min(year, 2018)) + mode + '_class.pkl', 'rb') as f:
        class_model = pkl.load(f)
    if mode == '_complex':
        vector = np.r_[[year, runtime], one_hot, men].reshape(1, -1)
    else:
        vector = np.r_[[year, runtime], one_hot].reshape(1, -1)
    # print(vector)
    with open('./pkls/' + str(min(year, 2018)) + mode + '_regression_' + str(class_model.predict(vector)[0]) + '.pkl', 'rb') as f:
        model = pkl.load(f)
        print('prediction: ', 10 ** model.predict(vector) * 1.7)
        prediction = 10 ** model.predict(vector) * 1.7
        return prediction
def get_length():
    for year in range(2009, 2019):
        with open('./pkls/' + str(year) + '_easy_feature.pkl', 'rb') as f:
            feature = pkl.load(f)
        l = np.ndarray(0)
        for k in feature:
            thisyear = feature[k]
            for sample in thisyear:
                if len(l) > 0:
                    l = np.r_[l, sample]
                else:
                    l = np.array(sample)
        print(len(l))


if __name__ == '__main__':
    # split_by_years()
    # men_presentation()
    # form_feature()
    # store_classification_model()
    # store_regression_model()

    # test()
    prediction = GUI()
    # get_length()
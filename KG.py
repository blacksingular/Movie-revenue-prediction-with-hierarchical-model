from __future__ import print_function
import imp

import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt;
# plt.rcdefaults()
# from matplotlib.ticker import FuncFormatter

import numpy as np
import pdb
import pandas as pd
import ast
import csv
import matplotlib
from sklearn.svm import SVR 

from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import LogisticRegression as LoR
from sklearn.linear_model import SGDRegressor as SGD

from sklearn.neural_network import MLPRegressor as MLP
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.ensemble import GradientBoostingRegressor as GBR
from utils import *
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.losses import mean_absolute_error
from keras.optimizers import Adam
from keras import metrics
from numpy import loadtxt
from xgboost import XGBClassifier
import keras as K
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
from keras.callbacks import EarlyStopping
# from knowledge_graph.KnowledgeGraph import evaluation as EVA

# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop
# matplotlib.use('agg')


credit_path = './TMDB/tmdb_5000_credits.csv'
movies_path = './TMDB/tmdb_5000_movies.csv'
imdb_path = './tables/movie_info.txt'


class KnowledgeGraph():
    def __init__(self):
        self.movie_meta_dict = dict()
        self.movie_chosen_dict = dict()
        self.actor_chosen_dict = dict()
        self.director_chosen_dict = dict()
        self.genre_node_dict = dict()
        self.genre_type_dict = dict()
        self.homo_edge_list = []
        self.home_recount = 1
        self.home_node_dict = dict()
        
    def datacleaning(self):
        # this the data cleaning for KG (only EN )
        # only care about the genre, id
        movie_df = pd.DataFrame(pd.read_csv(movies_path))
        # first filter the 'en' 
        movie_recount = 0
        for row in movie_df[['id','genres','original_language']].iterrows():
            movie_orig_lang = row[1]['original_language']
            movie_id = row[1]['id']
            if  movie_orig_lang == 'en':
                if movie_id not in self.movie_chosen_dict:
                    # for heterogenous
                    self.movie_chosen_dict[movie_id] = 'm'+str(movie_recount)
                    movie_recount +=1 
    
        # print(self.genre_chosen_list)   
        # print(len(self.movie_chosen_dict))# total 4505
    def datacleaning_imdb(self):
        csv_file = open('./tables/omdb.csv','w')

        with open(imdb_path, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                movie_id = line[0]
                # movie_meta_list += (movie_id+'\t')
                movie_meta_dict =  ast.literal_eval(line[1])
                if movie_meta_dict['Type'] == 'movie' and 2008 <= int(movie_meta_dict['Year']) <= 2019: # filer the movie out 
                    self.movie_meta_dict[movie_id] = movie_meta_dict
        #             for k,v in movie_meta_dict.items():
        #                 movie_meta_list += str(v)+'\t'
        #         csv_file.write(movie_meta_list[:-1]+'\n')
        # csv_file.close()          
            # print(len(self.movie_meta_dict))
        # convet into csv file 


        # split the feature and label

    def feature_analysis(self):
        # feature_selection = ['Year', 'Rated','Runtime','Genre', 'Director', 'Writer','Actors','Language', 'Country','Ratings','imdbRating','imdbVotes','Type','BoxOffice','Production']
        # continues feature covariance
        # descrete feature distribution 

        # first split feature and label
        
        self.train_movieid_label = dict()
        self.test_movieid_label = dict()

        self.year_dict = dict()
        self.genre_dict = dict()
        self.lang_dict = dict()
        self.country_dict = dict()
        self.product_dict = dict()
        self.rate_dict = dict()
        for movie, meta in self.movie_meta_dict.items():
            # split train test and feature, and label 
            if int(meta['Year']) <= 2015:
                self.train_movieid_label[movie] = float(meta['BoxOffice'].split('$')[1].replace(',',''))
            else:
                self.test_movieid_label[movie] = float(meta['BoxOffice'].split('$')[1].replace(',',''))

            # year_distribution
            if meta['Year'] not in self.year_dict:
                self.year_dict[meta['Year']] = 1 
            else:
                self.year_dict[meta['Year']] += 1 

            # genre distribution 
            genre_list = meta['Genre'].split(', ')
            for genre in genre_list:
                if genre not in self.genre_dict:
                    self.genre_dict[genre] = 1
                else:
                    self.genre_dict[genre] += 1
            
            # lang_dict = dict()
            lang_list = meta['Language'].split(', ')
            for lang in lang_list:
                if lang not in self.lang_dict:
                    self.lang_dict[lang] = 1
                else:
                    self.lang_dict[lang] += 1
            # country_dict 
            country_dict = meta['Country'].split(', ')
            for count in country_dict:
                if count not in self.country_dict:
                    self.country_dict[count] = 1 
                else:
                    self.country_dict[count] += 1 
            
            # produce_dict 
            product_dict = meta['Production'].split('/')
            for prod in product_dict:
                if prod[0]==' ':
                    prod = prod[1:]
                if prod[-1]==' ':
                    prod = prod[:-2]
                if prod not in self.product_dict:
                    self.product_dict[prod] = 1 
                else:
                    self.product_dict[prod] += 1 

            # rate_dict  
            if meta['Rated'] not in self.rate_dict:
                self.rate_dict[meta['Rated']] = 1 
            else:
                self.rate_dict[meta['Rated']] += 1 
        print(self.year_dict)
        print(self.genre_dict)
        print(self.lang_dict)
        print(self.country_dict)
        print(self.rate_dict)
        print(len(self.train_movieid_label.values()))
        print(len(self.test_movieid_label.values()))

        # print(self.product_dict)


    def feature_correalation_coefficient(self,feature_string):
        x_list = []
        y_list = []
        for movie_id, box_office in self.train_movieid_label.items():
            x_list.append(self.movie_meta_dict[movie_id][feature_string])
            y_list.append(box_office)
            # print(float(x_list),y_list)
        plt.boxplot(x_list, y_list,'x')
        plt.savefig('./figures/{}_budget.png'.format(feature_string))

        



    def genre_director_actor_node(self):
        credit_df = pd.DataFrame(pd.read_csv(credit_path))
        actor_recount = 0
        dirctor_recount = 0
        genre_recount = 0
        for row in credit_df[['movie_id','cast','crew']].iterrows():
            movie_id = row[1]['movie_id']
            if movie_id in self.movie_chosen_dict: # only care about en
                for dic in ast.literal_eval(row[1]['cast']):   # collecte first 3 actors 
                    if dic['order'] < 3: 
                        if dic['id'] not in self.actor_chosen_dict:
                            self.actor_chosen_dict[dic['id']] = 'a' + str(actor_recount)
                            actor_recount += 1 
                for dic in ast.literal_eval(row[1]['crew']):
                    if dic['job'] == 'Director' : 
                        if dic['id'] not in self.director_chosen_dict:
                            self.director_chosen_dict[dic['id']] = 'd' + str(dirctor_recount)
                            dirctor_recount +=1 
        # genre movie 1.node collection 2. edge collection 
        # 1. node collection in self.genre_node_dict
        movie_df = pd.DataFrame(pd.read_csv(movies_path))
        for row in movie_df[['id','genres']].iterrows():
            movie_id = row[1]['id']
            if movie_id in self.movie_chosen_dict:   # only care about en
                for dic in ast.literal_eval(row[1]['genres']):
                    # genre_dictribution in genre_type_dict
                    if dic['name'] not in self.genre_type_dict:  #
                        self.genre_type_dict[dic['name']] = 1 
                    else:
                        self.genre_type_dict[dic['name']] +=1
                    # genre_node in genre_node_dict
                    if dic['id'] not in self.genre_node_dict:
                        self.genre_node_dict[dic['id']] = 'g' + str(genre_recount)
                        genre_recount += 1 
    
    def homo_node_normalization(self):
        for movie_ori, m_id in KG.movie_chosen_dict.items():
            self.home_node_dict[m_id] = self.home_recount
            self.home_recount += 1 
        for actor_ori, a_id in KG.actor_chosen_dict.items():
            self.home_node_dict[a_id] = self.home_recount
            self.home_recount += 1

        for direc_ori, d_id in KG.director_chosen_dict.items():
            self.home_node_dict[d_id] = self.home_recount
            self.home_recount += 1
        
        for genre_ori, g_id in KG.genre_node_dict.items():
            self.home_node_dict[g_id] = self.home_recount
            self.home_recount += 1

    def homo_edge(self):
        credit_df = pd.DataFrame(pd.read_csv(credit_path))
        movie_df = pd.DataFrame(pd.read_csv(movies_path))

        f = open('./TMDB/movie_homo_edge.txt','w')
        # firs form movie-genre edge
        
        for row in movie_df[['id','genres']].iterrows():
            movie_id = row[1]['id']
            if movie_id in self.movie_chosen_dict:   # only care about en
                for dic in ast.literal_eval(row[1]['genres']):
                    movie_global = self.home_node_dict[self.movie_chosen_dict[movie_id]]
                    genre_global = self.home_node_dict[self.genre_node_dict[dic['id']]]
                    f.write('{}\t{}\t1\n'.format(movie_global, genre_global))
                    f.write('{}\t{}\t1\n'.format(genre_global, movie_global))
        # form movie- actor movie -director edge
        for row in credit_df[['movie_id','cast','crew']].iterrows():
            movie_id = row[1]['movie_id']
            if movie_id in self.movie_chosen_dict: # only care about en
                movie_global = self.home_node_dict[self.movie_chosen_dict[movie_id]]
                for dic in ast.literal_eval(row[1]['cast']):   # collecte first 3 actors 
                    if dic['order'] < 3:
                        actor_global = self.home_node_dict[self.actor_chosen_dict[dic['id']]]
                        f.write('{}\t{}\t1\n'.format(movie_global, actor_global))
                        f.write('{}\t{}\t1\n'.format(actor_global, movie_global))

                for dic in ast.literal_eval(row[1]['crew']):
                    if dic['job'] == 'Director':
                        director_global = self.home_node_dict[self.director_chosen_dict[dic['id']]]
                        f.write('{}\t{}\t1\n'.format(movie_global, director_global))
                        f.write('{}\t{}\t1\n'.format(director_global, movie_global))
        f.close()

 

class Regression():
    #this is 
    def __init__(self):
        self.movie_chosen_dict = dict()
        self.train_movie_chosen_dict = dict()
        self.test_movie_chosen_dict = dict()
        self.instance = dict()
        self.train_X = dict()
        self.train_y = dict()
        self.test_X = dict()
        self.test_y = dict()
        self.homo_node_embedding = dict()
        self.md_relation_tr = dict()
    def datacleaning(self):
        # only get en, from 2005 - 2016, have budge above 10000, have box office above 10000

        # 1. en
        movie_df = pd.DataFrame(pd.read_csv(movies_path))

        for row in movie_df[['id','original_language','revenue','budget','release_date']].iterrows():
            movie_orig_lang = row[1]['original_language']
            movie_id = row[1]['id']
            movie_revenue = row[1]['revenue']
            movie_budget = row[1]['budget']
            movie_release_date = row[1]['release_date']#2009-12-10
            
            if  movie_orig_lang == 'en' and movie_revenue >= 10000 and movie_budget >= 10000 and int(movie_release_date.split('-')[0]) >=2005:
                self.movie_chosen_dict[movie_id] = 1

    def train_test_split(self):
        movie_df = pd.DataFrame(pd.read_csv(movies_path))
        for row in movie_df[['id','release_date']].iterrows():
            if row[1]['id'] in self.movie_chosen_dict:
                if int(row[1]['release_date'].split('-')[0]) < 2015:
                    self.train_movie_chosen_dict[row[1]['id']] = 1
                else:
                    self.test_movie_chosen_dict[row[1]['id']] = 1
                
                
        print('num of train movie: '+ str(len(self.train_movie_chosen_dict)))
        print('num of test movie: '+ str(len(self.test_movie_chosen_dict)))
    
    def embedding_instance(self, home_node_dict, movie_chosen_dict, actor_chosen_dict, director_chosen_dict, genre_node_dict):
        # pass
        movie_df = pd.DataFrame(pd.read_csv(movies_path))
        credit_df = pd.DataFrame(pd.read_csv(credit_path))

        with open('./TMDB/vec.txt') as f:
            f.readline()
            for line in f:
                line = line.strip().split(' ')
                node = line[0]
                vec = [float(x) for x in line[1:]]
                self.homo_node_embedding[node] = vec

        def embedding(X, y, chosen_dict):
            for row in movie_df[['id','revenue','budget','popularity','runtime','release_date','vote_average','vote_count','genres','keywords','production_companies']].iterrows():
                if row[1]['id'] in chosen_dict:
                    
                    X[row[1]['id']] = ([row[1]['budget'],row[1]['popularity'],row[1]['runtime'],row[1]['vote_average'],row[1]['vote_count']])
                    y[row[1]['id']] = float(row[1]['revenue'])
                    genre_vec = [0] * 100
                    for dic in ast.literal_eval(row[1]['genres']):
                        g_vec = self.homo_node_embedding[str(home_node_dict[genre_node_dict[dic['id']]])]
                        genre_vec = [g_vec[i] + genre_vec[i] for i in range(len(genre_vec))]
                    genre_vec = [x/len(ast.literal_eval(row[1]['genres']))  for x in genre_vec]
                    X[row[1]['id']] += genre_vec

            for row in credit_df[['movie_id','cast','crew']].iterrows():
                movie_id = row[1]['movie_id']

                if movie_id in X: # only care about e
                    # self.train_X[row[1]['movie_id']].extent()
                    act_vec = [0] * 100
                    for dic in ast.literal_eval(row[1]['cast']):   # collecte first 3 actors 
                        if dic['order'] < 3:
                            actor_vec = self.homo_node_embedding[str(home_node_dict[actor_chosen_dict[dic['id']]])]
                            act_vec = [act_vec[i] + actor_vec[i] for i in range(len(actor_vec))]
                            '''TODO  normalize'''
                    act_vec = [x/3  for x in act_vec]
                    X[movie_id] += act_vec

                    director_vec = [0] * 100
                    for dic in ast.literal_eval(row[1]['crew']):
                        if dic['job'] == 'Director':
                            dir_vec = self.homo_node_embedding[str(home_node_dict[director_chosen_dict[dic['id']]])]
                            director_vec = [dir_vec[i] + director_vec[i] for i in range(len(director_vec))]
                            '''TODO  normalize'''
                    X[movie_id] += director_vec
            embedding(self.train_X, self.train_y, self.train_movie_chosen_dict)
            embedding(self.test_X, self.test_y, self.test_movie_chosen_dict)
            self.train_X = [y[1] for y in sorted(self.train_X.items(), key=lambda x: x[0])]
            self.train_y = [y[1] for y in sorted(self.train_y.items(), key=lambda x: x[0])]
            self.test_X = [y[1] for y in sorted(self.test_X.items(), key=lambda x: x[0])]
            self.test_y = [y[1] for y in sorted(self.test_y.items(), key=lambda x: x[0])]
            print(type(self.train_X), type(self.train_y))
            # print(len(self.test_X))
            # print(len(self.train_X[19995]))


    # def avg_regression(self):
    #     credit_df = pd.DataFrame(pd.read_csv(credit_path))
    #     revenue_df = pd.DataFrame(pd.read_csv(movie_path))
    #     # user_df = pd.DataFrame(pd.read_csv(user_rating_path, low_memory=False))
    #     for row in credit_df[['cast', 'crew', 'id']].iterrows():
    #         movie_id = row[1]['id']
    #         if str(movie_id) in self.chosen_train_dict:
    #             for dic in ast.literal_eval(row[1]['crew']):
    #                 if dic['job'] == 'Director':
    #                     if dic['id'] in self.md_relation_tr:
    #                         self.md_relation_tr[dic['id']].append(movie_id)
    #                     else:
    #                         self.md_relation_tr[dic['id']] = [movie_id]
    #         if str(movie_id) in self.chosen_test_dict:
    #             for dic in ast.literal_eval(row[1]['crew']):
    #                 if dic['job'] == 'Director':
    #                     if dic['id'] in self.md_relation_te:
    #                         self.md_relation_te[dic['id']].append(movie_id)
    #                     else:
    #                         self.md_relation_te[dic['id']] = [movie_id]
    #     for row in revenue_df[['revenue', 'id']].iterrows():
    #         movie_id = row[1]['id']
    #         if str(movie_id) in self.chosen_train_dict or str(movie_id) in self.chosen_test_dict:
    #             self.mf_relation[movie_id] = int(row[1]['revenue'])
    #     movies = 0

    #     for (k, v) in self.md_relation_tr.items():
    #         total = 0
    #         flag = 0
    #         for id in v:
    #             if str(id) in self.mf_relation:
    #                 total += int(self.mf_relation[str(id)])
    #                 flag += 1
    #         self.revenues[k] = total / flag
    #     my_list = sorted(self.revenues.items(), key=lambda x: x[1])
    #     self.mean = sum(list(zip(*my_list))[1]) / len(my_list)
    def line_regression(self):
        lr = LR()
        lr.fit(self.train_X, self.train_y)
        self.y_pre_test = lr.predict(self.test_X)
        self.y_pre_train = lr.predict(self.train_X)

    def logistic_regression(self):
        lor = LoR(C =1 )
        lor.fit(self.train_X, self.train_y)
        self.y_pre_test = lor.predict(self.test_X)
        self.y_pre_train = lor.predict(self.train_X)

    def svm_regression(self):
        c = [0.5,0.6,0.8,1.0,1.2,1.5,1.7,3]
        svr = SVR(kernel='linear')
        print(svr)
        svr.fit(self.train_X, self.train_y)
        self.y_pre_train = svr.predict(self.train_X)
        self.y_pre_test = svr.predict(self.test_X)

    def SGD_regression(self):
        sgd = SGD(eta0=0.1, learning_rate='adaptive')
        sgd.fit(self.train_X, self.train_y)
        self.y_pre_train = sgd.predict(self.train_X)
        self.y_pre_test = sgd.predict(self.test_X)

    def RandomForest_regression(self):
        rfr = RFR(n_estimators=1000, max_depth = 4)
        rfr.fit(self.train_X, self.train_y)
        path = rfr.decision_path(self.train_X)
        self.y_pre_train = rfr.predict(self.train_X)
        self.y_pre_test = rfr.predict(self.test_X)
        # print(path)

    def GradientBoosting_regression(self):
        params = {'n_estimators': 1000, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'lad'}
        gbr = GBR(**params)
        gbr.fit(self.train_X, self.train_y)
        self.y_pre_train = gbr.predict(self.train_X)
        self.y_pre_test = gbr.predict(self.test_X)



    def MLP_ours(self):
        batch_size = 128
        epochs = 1000
        input_size = 305
        self.train_X = np.array(self.train_X)
        self.train_y = np.array(self.train_y)
        self.test_X = np.array(self.test_X)
        self.test_y = np.array(self.test_y)
        model = Sequential()
        model.add(Dense(521, activation='relu', input_shape=(input_size,)))
        # model.add(Dropout(0.2))
        model.add(Dense(521, activation='relu'))
        # model.add(Dropout(0.2))
        model.add(Dense(1, activation='softmax'))
        model.summary()

        model.compile(loss='mean_absolute_error',optimizer=Adam(),metrics=['mae'])

        history = model.fit(self.train_X, self.train_y,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        )
        self.y_pre_train = model.predict(self.train_X)
        self.y_pre_test = model.predict(self.test_X)
        # score_train = model.evaluate(self.train_X, self.train_y, verbose=0)
        # score_test = model.evaluate(self.test_X, self.test_y, verbose=0)
        # print('Train loss:', score_train)
        # print('Test :', score_test)

    def MLP_sklearn(self):
#         mlpr = MLP(hidden_layer_sizes=(200,100,10), ## 隐藏层的神经元个数
#                     activation='tanh', 
#                     solver='adam', 
#                     alpha=0.001,   ## L2惩罚参数
#                     max_iter=5200, 
#                     random_state=123,
#                     learning_rate_init= 0.01
#                     # early_stopping=True, ## 是否提前停止训练
#                     # validation_fraction=0.1, ## 20%作为验证集
# #                     tol=1e-8,
#                    )
#         mlpr.fit(self.train_X, self.train_y)
#         plt.figure()
#         plt.plot(mlpr.loss_curve_)
#         plt.show()
#         self.y_pre_train = mlpr.predict(self.train_X)
#         self.y_pre_test = mlpr.predict(self.test_X)
        self.train_X = np.array(self.train_X)
        self.train_y = np.array(self.train_y)
        self.test_X = np.array(self.test_X)
        self.test_y = np.array(self.test_y)


        model = Sequential()
        model.add(Dense(2,input_dim=305,activation="tanh",name="full_1"))
        # model.add(Dense(200,activation="tanh",name="full_2"))
        # model.add(Dense(200,activation="tanh",name="full_3"))
        model.add(Dense(1, activation="linear"))
        model.summary()
        early_stopping = EarlyStopping(monitor='val_loss', patience=1000)
        model.compile(loss='mean_squared_error', optimizer='adam')
        model_fit = model.fit(self.train_X, self.train_y, batch_size=100,
                            epochs=2000, verbose=0,
                            validation_split=0.2,
                            callbacks=[early_stopping])
        self.y_pre_train = model.predict(self.train_X)
        self.y_pre_test = model.predict(self.test_X)
        ## 可视化
        plt.figure()
        plt.plot(model_fit.history["loss"])
        plt.plot(model_fit.history["val_loss"])
        plt.xlabel("iters")
        plt.ylabel("loss")
        plt.show()
    

    def evaluation(self):
        error_smape_train = 0
        error_mae_train = 0
        error_smape_test = 0
        error_mae_test = 0
        # train_loss
        for i in range(len(self.train_y)):
            error_smape_train += abs(self.train_y[i] - self.y_pre_train[i]) * 2 / (self.train_y[i] + abs(self.y_pre_train[i]))
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
   
    def analysis(self):
        plt.figure(figsize=(15,6))
        for ii,name in enumerate(data.columns):
            plt.subplot(2,5,ii+1)
            plt.hist(data.iloc[:,ii],25,color="green",alpha = 0.5)
            plt.title(name)
            plt.subplots_adjust()
            plt.show()

if __name__ == "__main__":
    KG = KnowledgeGraph()
    KG.datacleaning_imdb()
    KG.descrete_feature_plot()
    exit()
    KG.feature_analysis()
    KG.feature_correalation_coefficient('Year')
    # KG.feature_correalation_coefficient('Metascore')
    # KG.feature_correalation_coefficient('imdbRating')
    # KG.feature_correalation_coefficient('Metascore')
    exit()
    KG.datacleaning()
    KG.genre_director_actor_node()
    KG.homo_node_normalization()
    KG.homo_edge()


    Reg = Regression()
    Reg.datacleaning()
    Reg.train_test_split()
    Reg.embedding_instance(KG.home_node_dict, KG.movie_chosen_dict, KG.actor_chosen_dict, KG.director_chosen_dict, KG.genre_node_dict)

    with open('./result.txt', 'w') as f:
        print('Logisitic')
        Reg.logistic_regression()
        f.write(Reg.evaluation())

        print('SVM')
        # for c in [0.5,0.6,0.8,1.0,1.2,1.5,1.7,3]:
        Reg.svm_regression()
        f.write(Reg.evaluation())

        # print('SGD')
        # Reg.SGD_regression()
        # Reg.evaluation()
        print('Linear')
        Reg.line_regression()
        f.write(Reg.evaluation())
        
        print('RFR')
        Reg.RandomForest_regression()
        f.write(Reg.evaluation())

        print('GBR')
        Reg.GradientBoosting_regression()
        f.write(Reg.evaluation())


    # Reg.MLP_ours()
    # print('MLP_ours')
    # Reg.evaluation()
    # print('MLP_his')
    # Reg.MLP_sklearn()
    # Reg.evaluation()
    # print(sorted(KG.movie_chosen_dict.items(), key=lambda item: item[1]))
    # print(sorted(KG.actor_chosen_dict.items(), key=lambda item: item[1]))
    # print(sorted(KG.director_chosen_dict.items(), key=lambda item: item[1]))
    # print(sorted(KG.genre_node_dict.items(), key=lambda item: item[1]))


  

import pandas as pd
from imdb import IMDb
from tqdm import tqdm
import numpy as np
from scipy.stats import kstest
import matplotlib.pyplot as plt
from sklearn import mixture
# credit_path = '/run/media/yuanga/16C1-237D/to be moved/datasets/credits.csv'
# genre_path = '/run/media/yuanga/16C1-237D/to be moved/datasets/movies_metadata.csv'
# deleted_actor_path = '/tables/deleted_actor.csv'
# movie_actor_path = './tables/movie_actor.csv'
# movie_director_path = './tables/movie_director.csv'
# movie_genre_path = './tables/movie_genre.csv'


class Filter:
    def __init__(self):
        pass

    def get_valid_id(self): # 2019.3.20 get id using to crawl plot from IMDB website
        with open('./tables/movie_id.txt', 'r') as f:
            movieID = {}
            for line in f:
                line = line.strip().split('\t')
                try:
                    movieID[line[1]] = line[0].strip()
                except:
                    continue
        print('num of movie:', len(movieID))
        data_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full.csv"))

        # ensure that every movie in excel is in the txt
        if 0:
            notInTxt = []
            for title in list(data_df['Title']):
                if title not in list(movieID.values()):
                    notInTxt.append(title)
            print('number of movies in excel not in txt:', len(notInTxt))
            print(movieID['6333080'], notInTxt[0])
            print(notInTxt[:10])
            exit()

        # filter out unseen id in txt
        invalid = set()
        for k, v in movieID.items():
            if v not in list(data_df['Title']):
                invalid.add(k)
        for k in invalid:
            movieID.pop(k)
        print('after filtering:', len(movieID))
        return movieID


class Crawl:
    def __init__(self, reference):
        self.reference = reference

    def crawlPlot(self):  # 2019.3.20 Crawling plot data using IMDB API
        ia = IMDb()
        exist = set()
        with open('./tables/PlotID.txt', 'r') as f:
            for line in f:
                exist.add(line.strip().split('\t')[0])
        with open('./tables/PlotID.txt', 'a+') as f:
            for k, v in tqdm(self.reference.items()):
                if k in exist:
                    continue
                movie = ia.get_movie(k)
                try:
                    plot = movie['plot'][0]
                    f.write(k + '\t' + plot.split('::')[0] + '\n')
                except:
                    print(k, v)


class Cal:
    def __init__(self, log):
        self.log = log

    def cal_distribution_of_revenue(self):  # 2019.3.20 calculate the distribution of revenue
        data_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full_train.csv"))
        if self.log:
            dataTen = np.array(list(map(lambda x: np.log10(x), list(data_df['BoxOffice'])))).reshape(-1, 1)
            dataE = np.array(list(map(lambda x: np.log(x), list(data_df['BoxOffice'])))).reshape(-1, 1)
        else:
            data = list(data_df['BoxOffice'])
        # print(kstest(data, 'norm'))
        # print('var is:', np.var(data))
        # print('median is:', np.median(data))
        # print('average is:', np.mean(data))
        # print('min is:', np.min(data))
        # print('max is:', np.max(data))
        # counts = np.bincount(data)
        # print('most is:', np.argmax(counts))

        # using GMM to estimate the distribution
        modelTen = mixture.GaussianMixture(n_components=2)
        modelE = mixture.GaussianMixture(n_components=2)
        modelTen.fit(dataTen)
        modelE.fit(dataE)
        print('ten_mu:', modelTen.means_)
        print('ten_sigma:', modelTen.covariances_)
        print('e_mu:', modelE.means_)
        print('e_sigma:', modelE.covariances_)
        labelTen = modelTen.predict(dataTen)
        zeros = []
        ones = []
        for data, label in zip(dataTen, labelTen):
            if label:
                ones.append(data)
            else:
                zeros.append(data)
        print(len(ones), len(zeros))
        print(min(10 ** (max(zeros)), 10 ** (max(ones))))
        # add a new label to the training data
        new_train_df = pd.concat([data_df, pd.DataFrame(columns=['Label'])])
        for i in range(len(new_train_df)):
            new_train_df['Label'][i] = 1 if dataTen[i] in ones else 0
        print(list(new_train_df['Label']))
        new_train_df.to_csv('./new_tables/new_full_train.csv')

        # add a new label to the valid data
        new_valid_df = pd.concat([pd.DataFrame(pd.read_csv("./new_tables/omdb_full_valid.csv")), pd.DataFrame(columns=['Label'])])
        revenue = [np.log10(x) for x in list(new_valid_df['BoxOffice'])]
        # revenue = np.array(np.log10(x) for x in [list(new_valid_df['BoxOffice'])]).reshape(-1,1)
        print(revenue)
        revenue = np.array(revenue).reshape(-1,1)
        prdict_valid = list(modelTen.predict(revenue))
        print(prdict_valid)
        for i in range(len(prdict_valid)):
            new_valid_df['Label'][i] = prdict_valid[i]
        new_valid_df.to_csv('./new_tables/new_full_valid.csv')

        # labelE = modelE.predict(dataE)
        # dif = [1 if a == b else 0 for a, b in zip(labelTen, labelE)]
        # print(sum(dif), len(labelTen))
        # meanTen = modelTen.means_
        # varTen = modelTen.covariances_
        # print(data)
        # print(revenue)
        # plot distribution
        # print(10**revenue)
        # print(min(10**(revenue)),max(10**(revenue)))
        def plot(revenue):
            plt.hist(revenue, bins=50, range=(0,10),facecolor="blue", edgecolor="black", alpha=0.7)
            plt.xlabel('Revenue in loge space')
            plt.ylabel('Frequency')
            plt.title('Revenue Distribution')
            

            
            # h1 = 520
            # h2 = 180
            # mean = modelTen.means_
            # var = modelTen.covariances_ 
            # print(mean,var)
            # g1 = np.array([h1 / (var[0] * np.sqrt(2 * np.pi)) * np.exp(-(x - mean[0]) ** 2 / (2 * var[0] ** 2)) for x in
            #       np.linspace(-100, 23, 200)]).reshape(-1, 1)
            # g2 = np.array([h2 / (var[1] * np.sqrt(2 * np.pi)) * np.exp(-(x - mean[1]) ** 2 / (2 * var[1] ** 2)) for x in
            #       np.linspace(-100, 23, 200)]).reshape(-1, 1)
            # plt.plot(np.linspace(-100, 23, 200), g1, 'r', linewidth=5.0)
            # plt.plot(np.linspace(-100, 23, 200), g2, 'y', linewidth=5.0)
            plt.show()
        plot(dataTen)
        plot(revenue)


class Classification:
    def __init__(self):
        pass

    def model(self):
        data_df = pd.DataFrame(pd.read_csv("./new_tables/new_full_train.csv"))


if __name__ == "__main__":
    # f = Filter()
    # movieID = f.get_valid_id()
    # c = Crawl(movieID)
    # c.crawlPlot()
    cal = Cal(True)
    cal.cal_distribution_of_revenue()
# coding: utf-8

from utils import *
import pdb
import pandas as pd
import ast
import csv
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt; plt.rcdefaults()


class KnowledgeGraph:
    def __init__(self):
        self.actors = {}
        self.directors = {}
        self.cast = []
        self.crew = []
        self.actor_nodes = []
        self.director_nodes = []
        self.genre_nodes = []
        self.trash = []
        self.chosen_list = []

    def data_cleaning(self):
        credit_df = pd.DataFrame(pd.read_csv(credit_path))
        genre_df = pd.DataFrame(pd.read_csv(genre_path))
        for row in credit_df[['cast', 'crew', 'id']].iterrows():
            movie_id = row[1]['id']
            if not row[1]['cast'] or not row[1]['crew']:
                self.trash.append(movie_id)
        for row in genre_df[['id', 'genres', 'title']].iterrows():
            movie_id = row[1]['id']
            if not row[1]['genres'] or not row[1]['title']:
                self.trash.append(movie_id)
        print(self.trash)

    def movie_actor(self):
        # export movie_actor csv
        credit_df = pd.DataFrame(pd.read_csv(credit_path))
        for row in credit_df[['cast', 'crew', 'id']].iterrows():
            movie_id = row[1]['id']
            if movie_id in self.trash or str(movie_id) not in self.chosen_list:
                continue
            for dic in ast.literal_eval(row[1]['cast']):
                if dic['order'] < 3:
                    self.actor_nodes.append([movie_id, dic['name']])
        # with open(movie_actor_path, 'w') as f:
        #     for edge in self.actor_nodes:
        #         f.write(str(edge[0]) + ',' + str(edge[1]) + "\n")
        with open(movie_actor_path, 'w') as csvfile:
            csv.writer(csvfile).writerow(['movie_id', 'actor'])
            csv.writer(csvfile).writerows(self.actor_nodes)
        pdb.set_trace()

    def movie_director(self, mode='init'):
        # export movie_director csv
        credit_df = pd.DataFrame(pd.read_csv(credit_path))
        self.director_nodes = []
        for row in credit_df[['cast', 'crew', 'id']].iterrows():
            movie_id = row[1]['id']
            if mode == 'init':
                if movie_id in self.trash:
                    continue
                for dic in ast.literal_eval(row[1]['crew']):
                    if dic['job'] == 'Director':
                        self.director_nodes.append([movie_id, dic['name']])
            else:
                if movie_id in self.trash or str(movie_id) not in self.chosen_list:
                    continue
                for dic in ast.literal_eval(row[1]['crew']):
                    if dic['job'] == 'Director':
                        self.director_nodes.append([movie_id, dic['name']])
        # with open(movie_director_path, 'w') as f:
        #     for edge in self.director_nodes:
        #         f.write(str(edge[0]) + ',' + str(edge[1]) + "\n")
        with open(movie_director_path, 'w') as csvfile:
            csv.writer(csvfile).writerow(['movie_id', 'director'])
            csv.writer(csvfile).writerows(self.director_nodes)

    # for graph visualization
    def part_of_data(self):
        with open(movie_director_path, newline='') as csvfile:
            f = list(csv.reader(csvfile))
        np.random.shuffle(f)
        self.chosen_list = list(list(zip(*f))[0])[:100]

    def movie_genre(self):
        # export movie_genre csv
        genre_df = pd.DataFrame(pd.read_csv(genre_path, low_memory=False))
        for row in genre_df[['id', 'genres', 'title']].iterrows():
            movie_id = row[1]['id']
            # if movie_id in self.trash or str(movie_id) not in self.chosen_list:
            #     continue
            movie_title = row[1]['title']
            for dic in ast.literal_eval(row[1]['genres']):
                genre_id = dic['id']
                genre_name = dic['name']
                self.genre_nodes.append([movie_id, movie_title, genre_id, genre_name])
        # with open(movie_genre_path, 'w') as f:
        #     for edge in self.genre_nodes:
        #         f.write(str(edge[0]) + ',' + str(edge[1]) + ',' + str(edge[2]) + ',' + str(edge[3]) + '\n')
        with open(movie_genre_path, 'w') as csvfile:
            csv.writer(csvfile).writerow(['movie_id', 'movie_title', 'genre_id', 'genre'])
            csv.writer(csvfile).writerows(self.genre_nodes)
        # plot the distribution of the genre 
        genre_dict = dict()

        for instance in self.genre_nodes:
            if instance[3] not in genre_dict:
                movie_list = []
                movie_list.append(instance[1])
                genre_dict[instance[3]] = movie_list
            else:
                movie_list = genre_dict[instance[3]]
                if instance[1] not in movie_list:
                    movie_list.append(instance[1])
                    genre_dict[instance[3]] = movie_list
        # pie of distribution 
        genre_list= []
        num_list = []
        for genre, movie_list in genre_dict.items():
            genre_dict[genre] = len(movie_list)
        genre_sorted = sorted(genre_dict.items(), key=lambda kv: kv[1])
        for genre in genre_sorted:
            if genre[1] >1 :
                genre_list.append(genre[0])
                num_list.append(genre[1])
        print(num_list)
       
        objects = genre_list
        y_pos = np.arange(len(objects))
        performance = num_list
        plt.barh(y_pos, performance, align='center', alpha=0.5)
        plt.yticks(y_pos, objects)
        plt.xlabel('# Movies')
        plt.title('genre_distribution')
        plt.savefig('./genre_distribution.png')
        
        
    def delete_few_people(self):
        credit_df = pd.DataFrame(pd.read_csv(credit_path))
        for row in credit_df[['cast', 'crew', 'id']].iterrows():
            for dic in ast.literal_eval(row[1]['cast']):
                if dic['order'] < 5:
                    if dic['name'] in self.actors:
                        self.actors[dic['name']] += 1
                    else:
                        self.actors[dic['name']] = 1
                    # flag += 1
                    # print(flag)
                    # if flag == 100:
                    #     pdb.set_trace()
            for dic in ast.literal_eval(row[1]['crew']):
                if dic['job'] == 'Director':
                    if dic['name'] in self.directors:
                        self.directors[dic['name']] += 1
                    else:
                        self.directors[dic['name']] = 1
        self.cast = sorted(self.actors.items(), key=lambda x: x[1], reverse=True)
        self.crew = sorted(self.directors.items(), key=lambda x: x[1], reverse=True)
        return self.actors, self.directors

    def actor_distribution(self):
        self.actor_dict = dict()
        for row in self.actor_nodes:
            if row[1] not in self.actor_dict:
                movie_list = []
                movie_list.append(row[0])
                self.actor_dict[row[1]] = movie_list
            else:
                movie_list = self.actor_dict[row[1]]
                movie_list.append(row[0])
                self.actor_dict[row[1]] = movie_list
        # conut the degree
        degree_dict = dict()
        for actor, movie in self.actor_dict.items():
            if len(movie) not in degree_dict:
                degree_dict[len(movie)] = 1
            else:
                degree_dict[len(movie)] += 1
        # sort the dict 
        sorted_by_value = sorted(degree_dict.items(), key=lambda kv: kv[0])
        degree_list = []
        num_list = []
        for row in sorted_by_value:
            degree_list.append(np.log(row[0]))
            num_list.append(np.log(row[1]))
        plt.figure()
        plt.plot(degree_list,num_list,'x')
        plt.xlabel('num of movie acted(log ')
        plt.ylabel('num of actors(log)')
        plt.savefig('./actor_distribution.png')

    def director_distribution(self):
        self.direct_dict = dict()
        for row in self.director_nodes:
            if row[1] not in self.direct_dict:
                movie_list = []
                movie_list.append(row[0])
                self.direct_dict[row[1]] = movie_list
            else:
                movie_list = self.direct_dict[row[1]]
                movie_list.append(row[0])
                self.direct_dict[row[1]] = movie_list
        # conut the degree
        degree_dict = dict()
        for actor, movie in self.direct_dict.items():
            if len(movie) not in degree_dict:
                degree_dict[len(movie)] = 1
            else:
                degree_dict[len(movie)] += 1
        # sort the dict 
        sorted_by_value = sorted(degree_dict.items(), key=lambda kv: kv[0])
        degree_list = []
        num_list = []
        for row in sorted_by_value:
            degree_list.append(np.log(row[0]))
            num_list.append(np.log(row[1]))
        plt.figure()
        plt.plot(degree_list, num_list, 'x')
        plt.xlabel('num of movie directed(log ')
        plt.ylabel('num of director(log)')
        plt.savefig('./director_distribution.png')
        

def main():
    KG = KnowledgeGraph()
    # KG.data_cleaning()
    # KG.movie_director('init')
    # KG.part_of_data()
    KG.movie_genre()
    # KG.movie_actor()
    # KG.movie_director('delete')
    # KG.actor_distribution()
    # KG.director_distribution()
    # actors, directors = KG.delete_few_people()
    # print('Top 30 actors are: ', actors[:30])
    # print('Top 30 directors are: ', directors[:30])


if __name__ == "__main__":
    main()



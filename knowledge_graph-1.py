# coding: utf-8

from utils import *
import pdb
import pandas as pd
import ast
import csv
import matplotlib
matplotlib.use('agg')
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
        self.user_movielist_dict = {}
        self.actor_node_type = {} # this is the dict of node:type
        self.director_node_type = {}
        self.movie_node_type = {}
        self.genre_node_type = {}
        self.user_node_type = {}
        self.edge_type = [] # this is the dict of edge:type
        self.year_movie = {}
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
        print('Generate Movie-Actor graph table:')
        credit_df = pd.DataFrame(pd.read_csv(credit_path))
        for row in credit_df[['cast', 'crew', 'id']].iterrows():
            movie_id = row[1]['id']
#            if movie_id in self.trash or str(movie_id) not in self.chosen_list:
#                continue
            for dic in ast.literal_eval(row[1]['cast']):
                if dic['order'] < 3:
                    self.actor_nodes.append([movie_id, dic['id'], 1])
                    self.actor_nodes.append([dic['id'], movie_id, 1])
                    self.actor_node_type[('a:'+str(dic['id']))] = 1
                    self.movie_node_type['m:'+str(movie_id)] = 1
                    self.edge_type.append('a:'+str(dic['id'])+' '+'m:'+str(movie_id)+' '+'1 a')
        # with open(movie_actor_path, 'w') as f:
        #     for edge in self.actor_nodes:
        #         f.write(str(edge[0]) + ',' + str(edge[1]) + "\n")
        with open(movie_actor_path, 'w') as csvfile:
#            csv.writer(csvfile).writerow(['movie_id', 'actor'])
            csv.writer(csvfile,delimiter='\t').writerows(self.actor_nodes)
                
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
                plt.savefig('./output/actor_distribution.png')

    def movie_director(self, mode='init'):
        # export movie_director csv
        print('Generate movie-director graph table:')
        credit_df = pd.DataFrame(pd.read_csv(credit_path))
        self.director_nodes = []
        for row in credit_df[['cast', 'crew', 'id']].iterrows():
            movie_id = row[1]['id']
            if mode == 'init':
                if movie_id in self.trash:
                    continue
                for dic in ast.literal_eval(row[1]['crew']):
                    if dic['job'] == 'Director':
                        self.director_nodes.append([movie_id, dic['id'],1])
                        self.director_nodes.append([dic['id'],movie_id, 1])
                        self.director_node_type[('d:'+str(dic['id']))] = 1
                        self.movie_node_type['m:'+str(movie_id)] = 1
                        self.edge_type.append('d:' + str(dic['id']) + ' '+'m:'+str(movie_id)+' '+'1 d')
            else:
                if movie_id in self.trash or str(movie_id) not in self.chosen_list:
                    continue
                for dic in ast.literal_eval(row[1]['crew']):
                    if dic['job'] == 'Director':
                        self.director_nodes.append([movie_id, dic['id'],1])
                        self.director_nodes.append([dic['id'],movie_id,1])
                        self.director_node_type[('d:'+str(dic['id']))] = 1
                        self.movie_node_type['m:'+str(movie_id)] = 1
                        self.edge_type.append('d:' + str(dic['id']) +' ' + 'm:' + str(movie_id) + ' ' + '1 d')
        # with open(movie_director_path, 'w') as f:
        #     for edge in self.director_nodes:
        #         f.write(str(edge[0]) + ',' + str(edge[1]) + "\n")
        with open(movie_director_path, 'w') as csvfile:
#            csv.writer(csvfile).writerow(['movie_id', 'director','weight'])
            csv.writer(csvfile,delimiter='\t').writerows(self.director_nodes)
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
        plt.savefig('./output/director_distribution.png')

    # for graph visualization
    def part_of_data(self):
        with open(movie_director_path, newline='') as csvfile:
            f = list(csv.reader(csvfile))
        np.random.shuffle(f)
        self.chosen_list = list(list(zip(*f))[0])[:100]

    def movie_genre(self):
        print('Generate the movie-genre table')
        # export movie_genre csv
        year_nummovies = {}
        genre_df = pd.DataFrame(pd.read_csv(genre_path, low_memory=False))
        for row in genre_df[['id', 'genres', 'title','release_date']].iterrows():
            movie_id = row[1]['id']
            # if movie_id in self.trash or str(movie_id) not in self.chosen_list:
            #     continue
            movie_title = row[1]['title']

            release_year = str(row[1]['release_date']).split('-')[0]
            for dic in ast.literal_eval(row[1]['genres']):
                genre_id = dic['id']
                genre_name = dic['name']
                self.genre_nodes.append([movie_id, genre_id, 1])
                self.genre_nodes.append([genre_id, movie_id, 1])
                self.genre_node_type[('g:'+str(genre_id))] = 1
                self.movie_node_type['m:'+str(movie_id)] = 1
                self.edge_type.append('m:'+ str(movie_id) +' '+'g:'+str(genre_id)+' '+'1 b')
                
                if release_year not in self.year_movie:
                    movie_list = []
                    movie_list.append(movie_id)
                    self.year_movie[release_year] = movie_list
                
                else:
                    movie_list = self.year_movie[release_year]
                    movie_list.append(movie_id)
                    self.year_movie[release_year] = movie_list
        for year,movie_list in self.year_movie.items():
            year_nummovies[year] = len(movie_list)
#        print(sorted(year_nummovies.items(), key=lambda kv: kv[1]))
        sorted_by_value = sorted(year_nummovies.items(), key=lambda kv: kv[0])
        print(sorted_by_value)
        year_list = []
        num_list = []
        for row in sorted_by_value:
            year_list.append((row[0]))
            num_list.append((row[1]))
        plt.figure()
        plt.plot(year_list,num_list,'x')
        plt.xlabel('year')
        plt.ylabel('num of movies')
        plt.savefig('./output/year_movies.png')
        
        # with open(movie_genre_path, 'w') as f:
        #     for edge in self.genre_nodes:
        #         f.write(str(edge[0]) + ',' + str(edge[1]) + ',' + str(edge[2]) + ',' + str(edge[3]) + '\n')
        
        with open(movie_genre_path, 'w') as csvfile:
#            csv.writer(csvfile).writerow(['movie_id', 'movie_title', 'genre_id', 'genre'])
            csv.writer(csvfile,delimiter='\t').writerows(self.genre_nodes)
        # plot the distribution of the genre
        
        
#        genre_dict = dict()
#        for instance in self.genre_nodes:
#            if instance[3] not in genre_dict:
#                movie_list = []
#                movie_list.append(instance[1])
#                genre_dict[instance[3]] = movie_list
#            else:
#                movie_list = genre_dict[instance[3]]
#                if instance[1] not in movie_list:
#                    movie_list.append(instance[1])
#                    genre_dict[instance[3]] = movie_list
#        # pie of distribution
#        genre_list= []
#        num_list = []
#        for genre, movie_list in genre_dict.items():
#            genre_dict[genre] = len(movie_list)
#        genre_sorted = sorted(genre_dict.items(), key=lambda kv: kv[1])
#        for genre in genre_sorted:
#            if genre[1] >1 :
#                genre_list.append(genre[0])
#                num_list.append(genre[1])
##        print(num_list)
#
#        objects = genre_list
#        y_pos = np.arange(len(objects))
#        performance = num_list
#
#        plt.barh(y_pos, performance, align='center', alpha=0.5)
#        plt.yticks(y_pos, objects)
#        plt.xlabel('# Movies')
#        plt.title('genre_distribution')
#        plt.savefig('./output/genre_distribution.png')

        
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



    def user_rating(self):
        # we would like to see the distribution of user rating
        #userId	movieId	rating	timestamp
        user_num_dict = {}
        
        user_df = pd.DataFrame(pd.read_csv(user_rating_path, low_memory=False))
        for row in user_df[['userid', 'movieid', 'rating','timestamp']].iterrows():
            user_id = row[1]['userid']
            movie_id = row[1]['movieid']
            self.user_node_type['u:'+str(user_id)]  = 1
            self.edge_type.append('u:'+str(user_id)+' '+'m:'+str(movie_id)+ ' ' + ' ')
            if user_id not in self.user_movielist_dict:
                movie_list = []
                movie_list.append(movie_id)
                self.user_movielist_dict[user_id] = movie_list
            else:
                movie_list = self.user_movielist_dict[user_id]
                movie_list.append(movie_id)
                self.user_movielist_dict[user_id] = movie_list
        for user, movie_list in self.user_movielist_dict.items():
            user_num_dict[user] = len(movie_list)
        # distribution 
        user_degree = dict()
        
        for user, degree in user_num_dict.items():
            if degree not in user_degree:
                user_degree[degree] = 1 
            else:
                user_degree[degree] += 1 
        
        sorted_by_value = sorted(user_degree.items(), key=lambda kv: kv[0])
        degree_list = []
        num_list = []
        for row in sorted_by_value:
            degree_list.append((row[0]))
            num_list.append((row[1]))
        plt.figure()
        plt.plot(degree_list, num_list, 'x')
        plt.xlabel('num of comments')
        plt.ylabel('num of user')
        plt.savefig('./output/user_rating_distribution.png')
    def generate_HIN(self):
        with open('./output/attibutr', 'w') as f:
            for node in self.actor_node_type:
                f.write(node)
                f.write('\n')
            for node in self.director_node_type:
                f.write(node)
                f.write('\n')
            for node in self.genre_node_type:
                f.write(node)
                f.write('\n')
#        with open('./output/director.node','w') as f:

        with open('./output/movie.node','w') as f:
            for node in self.movie_node_type:
                f.write(node)
                f.write('\n')
#        with open('./output/genre.node','w') as f:

        with open('./output/HIN.hin','w') as f:
            for edge in self.edge_type:
                f.write(edge)
                f.write('\n')
    def data_split(self):



def main():
    KG = KnowledgeGraph() # init the KG
    # KG.data_cleaning()
    KG.movie_director('init')
    # KG.part_of_data()
    KG.movie_genre()
    KG.movie_actor()
    # KG.movie_director('delete')

#    KG.user_rating()
    # actors, directors = KG.delete_few_people()
    # print('Top 30 actors are: ', actors[:30])
    # print('Top 30 directors are: ', directors[:30])
    KG.generate_HIN()

if __name__ == "__main__":
    main()


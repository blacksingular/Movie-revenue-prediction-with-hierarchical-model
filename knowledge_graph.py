# coding: utf-8

from utils import *
import pdb
import pandas as pd
import ast
import csv


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

        def movie_actor_director(self):
            # export movie_actor csv
            credit_df = pd.DataFrame(pd.read_csv(credit_path))
            for row in credit_df[['cast', 'crew', 'id']].iterrows():
                movie_id = row[1]['id']
                if movie_id in self.trash:
                    continue
                for dic in ast.literal_eval(row[1]['cast']):
                    self.actor_nodes.append([movie_id, dic['name']])
            # with open(movie_actor_path, 'w') as f:
            #     for edge in self.actor_nodes:
            #         f.write(str(edge[0]) + ',' + str(edge[1]) + "\n")
            with open(movie_actor_path, 'w') as csvfile:
                csv.writer(csvfile).writerow(['movie_id','actor'])
                csv.writer(csvfile).writerows(self.actor_nodes)

            # export movie_director csv
            for row in credit_df[['cast', 'crew', 'id']].iterrows():
                movie_id = row[1]['id']
                if movie_id in self.trash:
                    continue
                for dic in ast.literal_eval(row[1]['crew']):
                    if dic['job'] == 'Director':
                        self.director_nodes.append([movie_id, dic['name']])
            # with open(movie_director_path, 'w') as f:
            #     for edge in self.director_nodes:
            #         f.write(str(edge[0]) + ',' + str(edge[1]) + "\n")
            with open(movie_director_path, 'w') as csvfile:
                csv.writer(csvfile).writerow(['movie_id','director'])
                csv.writer(csvfile).writerows(self.director_nodes)

        def movie_genre(self):
            # export movie_genre csv
            genre_df = pd.DataFrame(pd.read_csv(genre_path))
            for row in genre_df[['id', 'genres', 'title']].iterrows():
                movie_id = row[1]['id']
                if movie_id in self.trash:
                    continue
                movie_title = row[1]['title']
                for dic in ast.literal_eval(row[1]['genres']):
                    genre_id = dic['id']
                    genre_name = dic['name']
                    self.genre_nodes.append([movie_id, movie_title, genre_id, genre_name])
            # with open(movie_genre_path, 'w') as f:
            #     for edge in self.genre_nodes:
            #         f.write(str(edge[0]) + ',' + str(edge[1]) + ',' + str(edge[2]) + ',' + str(edge[3]) + '\n')
            with open(movie_genre_path, 'w') as csvfile:
                csv.writer(csvfile).writerow(['movie_id','movie_title','genre_id','genre'])
                csv.writer(csvfile).writerows(self.genre_nodes)

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


def main():
    KG = KnowledgeGraph()
    KG.data_cleaning()
    KG.movie_genre()
    KG.movie_actor_director()
    # actors, directors = KG.delete_few_people()
    # print('Top 30 actors are: ', actors[:30])
    # print('Top 30 directors are: ', directors[:30])


if __name__ == "__main__":
    main()



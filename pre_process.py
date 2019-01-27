from util import *
import mysql.connector
import pandas as pd 
import csv 
import ast

class KnowledgeGraph():
    def __init__(self):
        self.edge_list = [] 
    
    def Movie_genre(self):
        data = pd.read_csv(movie_path)
        df = pd.DataFrame(data)
        columns = df.columns
        for row in df[['id', 'genres', 'title']].iterrows():
            movie_id = row[1]['id']
            movie_title = row[1]['title']
            for dic in ast.literal_eval(row[1]['genres']):
                genre_id = dic['id']
                genre_name = dic['name']
                self.edge_list.append([movie_id,movie_title,genre_id,genre_name])
        with open(file_path+'Movie_Genre_table.csv','w') as f:
            for edge in self.edge_list:
                f.write(str(edge[0])+','+str(edge[1])+','+str(edge[2])+','+str(edge[3])+'\n')
        # print(self.edge_list)

    
class DataBase():
    def __init__(self,host_name,user_name):
        # init the database 
        mydb = mysql.connector.connect(
            host=host_name,
            user= user_name)
        mycursor = mydb.cursor()
        mycursor.execute("CREATE DATABASE mydatabase")

def main():
    KG = KnowledgeGraph()
    KG.Movie_genre()
    # print(KG.edge_list)
    # mydb = DataBase('localhost','')

if __name__ == "__main__":
    main()
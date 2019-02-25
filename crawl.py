# import wptools
import pickle
# import omdb
import os
import urllib
from urllib import request
import requests
import ast
# from imdb import IMDb
from unidecode import unidecode
from bs4 import BeautifulSoup
from urllib.parse import unquote
from tqdm import tqdm


home_page = 'https://en.wikipedia.org/wiki/Category:Films_by_year'
wiki_home = 'https://en.wikipedia.org'
year_range = [year for year in range(2008, 2019)]


# get movie list over given years
def extract_movielist():
    year_movielist = dict()
    for year in year_range:
        print("collect movie for: "+ str(year))
        year_first_url = '/wiki/Category:{}_films'.format(year)
        movie_peryear = []
        next_url = year_first_url
        first_url_falg = 1
        final_url_falg = 1
        while final_url_falg:
            this_url = wiki_home + next_url
            # print(this_url)
            html_page = urllib.request.urlopen(this_url)
            soup = BeautifulSoup(html_page,features="lxml")
            link_list = [link.get('href')  for link in soup.findAll('a')[1:]]
            pre_next_list = []
            for link in soup.findAll('a')[1:]: # the first one is NoneType
                # print(link.get('href'))
                if '/w/index.php?' in link.get('href') and link.get('href').find('/w/index.php?') == 0:
                    pre_next_list.append(link.get('href'))
                    # print(link.get('href'))
            # print(pre_next_list)
            if pre_next_list[0] == pre_next_list[1]:
                if first_url_falg:
                    # print('first page:')
                    next_url = pre_next_list[1]
                    indices_start = [i for i, x in enumerate(link_list) if x == next_url][0]
                    indices_end = [i for i, x in enumerate(link_list) if x == next_url][1]
                    movie_list = link_list[indices_start+1:indices_end]
                    movie_peryear.extend(movie_list)
                    first_url_falg = 0
                    # print(movie_list)
                else:
                    # print('final page:')
                    previous_url = pre_next_list[1]
                    indices_start = [i for i, x in enumerate(link_list) if x == previous_url][0]
                    indices_end = [i for i, x in enumerate(link_list) if x == previous_url][1]
                    movie_list = link_list[indices_start+1:indices_end]
                    movie_peryear.extend(movie_list)
                    final_url_falg = 0
                    # print(movie_list)
            else:
                previous_url = pre_next_list[0]
                next_url = pre_next_list[1]
                indices_start = [i for i, x in enumerate(link_list) if x == previous_url][0]
                indices_end = [i for i, x in enumerate(link_list) if x == previous_url][1]
                movie_list = link_list[indices_start+2:indices_end]
                movie_peryear.extend(movie_list)
                # print(movie_list)
        year_movielist[year] = movie_peryear
        with open('./tables/year_movielist.pkl','wb') as f:
            pickle.dump(year_movielist,f)

    return year_movielist


# crawl using wikipedia
def extract_infobox(year_movielist):
    movie_infobox = {}
    invalid_movies = []
    exist_elements = {}
    # exist_flag = False
    # idx = -1
    #if os.path.exists('./tables/infobox.pkl'):
    #    with open('./tables/infobox.pkl','rb') as f:
    #        exist_elements = pickle.load(f)
    for year, movie_list in year_movielist.items():
        # print(year,len(movie_list))
        if os.path.exists('./tables/'+ str(year) + '.pkl'):
            continue
    #         while not exist_flag:
    #             page = wptools.page(movie_list[idx].split('/')[-1])
    #             try:
    #                 infobox = page.get_parse().data['infobox']
    #             except LookupError:
    #                 idx -= 1
    #                 continue
    #             if infobox in exist_elements[year]:
    #                 exist_flag = True
    #                 break
    #             else:
    #                 break
    #     if exist_flag:
    #         continue
        for movie in movie_list:
            page = wptools.page(movie.split('/')[-1])
            # print("################page is:", movie.split('/')[-1])
            try:
                infobox = page.get_parse().data['infobox']
            except LookupError:
                invalid_movies.append(movie.split('/')[-1])
                print('invalid title')
            # print(infobox)
            if year not in movie_infobox:
                movie_infobox[year] = [infobox]
            else:
                movie_infobox[year].append(infobox)
        #  restore movie_lists per year
        with open('./tables/'+str(year)+'.pkl', 'wb') as f:
            pickle.dump(movie_infobox, f)
            print('pickle dump completed')
        #with open('/tables/invalid_ones.pkl', 'wb') as f:
        #   pickle.dump(invalid_movies, f)
        print(len(invalid_movies), len(movie_list), len(movie_infobox))


# crawl using IMDB API
def imdb_API(clean_moviename_list):

    # create an instance of the IMDb class
    ia = IMDb()
    # get a movie
    # movie_id_list = []
    exists_list = set()
    if os.path.exists('./tables/movie_id.txt'):
        with open('./tables/movie_id.txt','r') as f:
            exists_list = set([x.strip().split('\t')[0] for x in f.readlines()])
        print((exists_list))
    clean_moviename_list = clean_moviename_list.difference(exists_list)
    print(len(clean_moviename_list))

    # print(clean_moviename_list[0], exists_list[0])
    file = open('./tables/movie_id.txt','a+')

    for movie in tqdm(clean_moviename_list):

        # print(movie)
        # movie = ia.search_movie(movie)[0]
        # print(ia.search_movie(movie)[0].movieID)
        try:
            # movie_id_list.append(ia.search_movie(movie)[0].movieID)
            file.write(movie + '\t' + ia.search_movie(movie)[0].movieID + '\n')
            file.flush()
        except:
            print(movie)
    file.close()


def get_movie_list():
    # imdb_API()
    if os.path.exists('./tables/year_movielist.pkl'):
        print('Year movielist exist')
        with open('./tables/year_movielist.pkl', 'rb') as f:
            year_movielist = pickle.load(f)
    else:
        year_movielist = extract_movielist()

    clean_moviename_list = set()
    for year, movie_list in (year_movielist.items()):
        for movie in movie_list:
            movie = (unidecode(unquote(movie)))
            movie = movie.split('/')[2]
            movie = movie.replace('_', ' ')
            movie = movie.split('(')[0]
            clean_moviename_list.add(movie)
            # print(unidecode(unquote(movie)))
            # print(movie)
    print(len(clean_moviename_list))
    movie_id_list = imdb_API(clean_moviename_list)

    print(len(movie_id_list))

    # extract_infobox(year_movielist)
    # with open('./tables/2018.pkl','rb') as f :
    #     movie_infobox = pickle.load(f,protocol=3)

    # for year, movie_list in movie_infobox.items():
    #     print(year,len(movie_list))
    

    # box_office_mojo = bom.BoxOfficeMojo()
    # box_office_mojo.crawl_for_urls()


    # movie = box_office_mojo.get_movie_summary("titanic")
    # movie.clean_data()
    # print(movie.to_json())

    # weekly = box_office_mojo.get_weekly_summary("titanic")
    # weekly.clean_data()
    # print(weekly.to_json())

# crawl using OMDB API
def omdb_request():
    my_id = "efc4c1c1"
    page = "http://www.omdbapi.com/?i="
    #file_path = ('./tables/movie_id.txt')
    file_path = ('./tables/movie_id_2000to2007and2019.txt')
#    out_file_path = ('./tables/movie_info.txt')
    out_file_path = ('./tables/movie_info_2.txt')
    assert os.path.exists(file_path) == True
    with open(file_path, 'r') as f:
        movie_id = ['tt' + x.strip().split('\t')[1] for x in f.readlines()]
    f = open(out_file_path, 'a+')
    for i in tqdm(range(len(movie_id))):
        info = requests.get(page + movie_id[i] + "&apikey=" + my_id).content.decode("utf-8")
        if not ast.literal_eval(info).get('BoxOffice', 0) or ast.literal_eval(info)["BoxOffice"] == 'N/A':         # Box office non-exist
            continue
        else:
            f.write(movie_id[i] + "\t" + unidecode(unquote(info)) + "\n")
            f.flush()
    f.close()


if __name__ == "__main__":
    omdb_request()

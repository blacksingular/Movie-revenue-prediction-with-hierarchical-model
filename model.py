import urllib
from  urllib import *
import ast
from bs4 import BeautifulSoup 
import re
import wptools
import pickle
import os
home_page = 'https://en.wikipedia.org/wiki/Category:Films_by_year'
wiki_home = 'https://en.wikipedia.org'
year_range = [year for year in range(2008,2019)]

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
if os.path.exists('./tables/year_movielist.pkl'):
    print('Year movielist exist')
    with open('./tables/year_movielist.pkl', 'rb') as f:
        year_movielist = pickle.load(f)
else:   
    year_movielist = extract_movielist()


extract_infobox(year_movielist)

import tkinter as tk
import tkinter.messagebox
import pandas as pd
from storage import *





# 这里是窗口的内容
movie_search_dict = dict()
def data_set():
    params = ["BoxOffice", "Title"]
    data_df = pd.DataFrame(pd.read_csv("./new_tables/omdb_full.csv"))
    for title,boxoffice in zip(data_df['Title'],data_df['BoxOffice']):
        movie_search_dict[title] = boxoffice
    # print(movie_search_dict)


def GUI_interface():
    #####windows 
    # welcome image

    window = tk.Tk()
    window.title('Movies Box office')
    window.geometry('600x400')
    canvas = tk.Canvas(window, height=80, width=80)#创建画布
    image_file = tk.PhotoImage(file='./img/UMSI.png')#加载图片文件
    image = canvas.create_image(0,10,anchor='nw', image=image_file)#将图片置于画布上
    canvas.pack(side='top')#放置画布（为上端）
    tk.Label(window, text='Team Lucy:Jiazhao, Yuan').place(x=400, y=300)#创建一个`label`名为`User name: `置于坐标（50,150）

    l= tk.Label(window, 
        text='Movies Box Office!',    # 标签的文字
        bg='white',     # 背景颜色
        font=('Arial', 34),     # 字体和字体大小
        width=15, height=2  # 标签长宽
        )
    l.pack()    # 固定窗口位置

    # Label settings
    tk.Label(window, text='Movie name: ').place(x=50, y= 150)#创建一个`label`名为`User name: `置于坐标（50,150）
    var_movie_name = tk.StringVar()#定义变量
    var_movie_name.set('')#变量赋值'example@python.com'
    entry_movie_name = tk.Entry(window, textvariable=var_movie_name)#创建一个`entry`，显示为变量`var_usr_name`即图中的`example@python.com`
    entry_movie_name.place(x=160, y=150)

    # 触发 函数
    def movie_search():
        movie_name = var_movie_name.get()  # get movie name 
        try:
            revenue = movie_search_dict[movie_name]
            tk.messagebox.showinfo('Movie Search Result', 'Movie name:' + movie_name + '\n'+ 'Box office: $' + str(revenue))
        except KeyError:
            tk.messagebox.showerror('Movie Search Result',movie_name+' Not found.\n You can try using prediction')

    def movie_prediction():
        movie_name = var_movie_name.get()  # get movie name 
        def Movie_predit_insert():

                nm = new_movie_name.get()
                nd = new_directors.get()
                if ',' in nd:
                    nd = nd.split(', ')
                nw = new_writers.get()
                if ', ' in nw:
                    nw = nw.split(', ')
                na = new_actors.get()
                if ', ' in na:
                    na = na.split(', ')
                
                nt = new_theme.get()
                nc = new_count.get()
                ny = int(new_year.get())
                nl = new_lang.get()
                nr = int(new_runtime.get())
                # ng = int(new_runtime.get())
                #year=2018, directors=['David Yates'], writers=['J.K. Rowling'], actors = ['Eddie Redmayne', 'Katherine Waterston'], genre='Action', language='English', country='UK', runtime=134
                prediction = GUI(ny, nd, nw, na, nt, nl, nc, nr)
                tk.messagebox.showinfo('Movie Prediction Result', nm+ '\n'+ 'Box office: '+str(prediction))
                # print('1,',prediction)
                # window_prediction.destroy()


        try:
            revenue = movie_search_dict[movie_name]
            tk.messagebox.showinfo('Movie Prediction Result', movie_name+' has existed in our dataset' + '\n'+ 'Box office: '+str(revenue))
        except KeyError:
            window_prediction = tk.Toplevel(window)
            window_prediction.geometry('500x450')
            window_prediction.title('Movie Box office prediction')
            new_movie_name = tk.StringVar()#将输入的注册名赋值给变量
            new_movie_name.set('Avengers: Endgame')#将最初显示定为'example@python.com'
            tk.Label(window_prediction, text='Movie title: ').place(x=10, y= 10)#将`User name:`放置在坐标（10,10）。
            entry_new_name = tk.Entry(window_prediction, textvariable=new_movie_name)#创建一个注册名的`entry`，变量为`new_name`
            entry_new_name.place(x=150, y=10)#`entry`放置在坐标（150,10）.

            # director text and label
            new_directors = tk.StringVar()
            new_directors.set('Anthony Russo, Joe Russo')
            tk.Label(window_prediction, text='Directors: ').place(x=10, y=50)
            entry_usr_pwd = tk.Entry(window_prediction, textvariable=new_directors, show=None)
            entry_usr_pwd.place(x=150, y=50)
            # writes text and label
            new_writers = tk.StringVar()
            new_writers.set('Christopher Markus, Stephen McFeely')
            tk.Label(window_prediction, text='Writers').place(x=10, y= 90)
            entry_usr_pwd_confirm = tk.Entry(window_prediction, textvariable=new_writers, show=None)
            entry_usr_pwd_confirm.place(x=150, y=90)
            # actors text and label
            new_actors = tk.StringVar()
            new_actors.set('Brie Larson, Scarlett Johansson')
            tk.Label(window_prediction, text='Actors').place(x=10, y= 130)
            entry_usr_pwd_confirm = tk.Entry(window_prediction, textvariable=new_actors, show=None)
            entry_usr_pwd_confirm.place(x=150, y=130)
            # other information text and label

            
            new_theme = tk.StringVar()
            new_theme.set('Action')
            tk.Label(window_prediction, text='themes').place(x=10, y= 170)
            entry_usr_pwd_confirm = tk.Entry(window_prediction, textvariable=new_theme, show=None)
            entry_usr_pwd_confirm.place(x=150, y=170)

            new_lang = tk.StringVar()
            new_lang.set('English')
            tk.Label(window_prediction, text='Language').place(x=10, y= 210)
            entry_usr_pwd_confirm = tk.Entry(window_prediction, textvariable=new_lang, show=None)
            entry_usr_pwd_confirm.place(x=150, y=210)

            new_count = tk.StringVar()
            new_count.set('USA')
            tk.Label(window_prediction, text='Country').place(x=10, y= 250)
            entry_usr_pwd_confirm = tk.Entry(window_prediction, textvariable=new_count, show=None)
            entry_usr_pwd_confirm.place(x=150, y=250)

            new_year = tk.StringVar()
            new_year.set('2019')
            tk.Label(window_prediction, text='Year').place(x=10, y= 290)
            entry_usr_pwd_confirm = tk.Entry(window_prediction, textvariable=new_year, show=None)
            entry_usr_pwd_confirm.place(x=150, y=290)

            new_runtime = tk.StringVar()
            new_runtime.set('134')
            tk.Label(window_prediction, text='Runtime').place(x=10, y= 330)
            entry_usr_pwd_confirm = tk.Entry(window_prediction, textvariable=new_runtime, show=None)
            entry_usr_pwd_confirm.place(x=150, y=330)
            
            # button of predict
            btn_comfirm_sign_up = tk.Button(window_prediction, text='Predict', command=Movie_predit_insert)
            btn_comfirm_sign_up.place(x=150, y=390)
            

    ### 按钮 :
    btn_search = tk.Button(window, text='Search', command=movie_search)
    btn_search.place(x = 170, y =230)
    btn_predict = tk.Button(window, text='Predict', command=movie_prediction)
    btn_predict.place(x = 270, y =230)



    window.mainloop()

data_set()
GUI_interface()
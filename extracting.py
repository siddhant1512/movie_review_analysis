import requests
from bs4 import BeautifulSoup
import cv2
import matplotlib.pyplot as plt


def extract(movie):
    try:
        url = "https://www.rottentomatoes.com/m/{}".format(movie)

        page = requests.get(url)

        html = BeautifulSoup(page.content, 'html.parser')


        ##extracting title
        title = html.findAll("h1")[0]
        print("************TITLE:************\n", title.text.strip())
        ##about movie
        print("\n\n")
        about = html.find('div',{"class":"movie_synopsis clamp clamp-6 js-clamp"})
        about_text = about.text.strip()
        print("************ABOUT:************\n",about_text)
        ## time in theatre
        time = html.find("time")
        print("\n\n")
        print("************RELEASED_ON:************\n",time.text)

        ##image_url
        try:
            img = html.find("img",{"class":"PhotosCarousel__image js-lazyLoad"})

            img_url = (img["data-src"])

            ## saving image_file
            im_file = open(movie + ".jpeg",'wb')
            im_file.write(requests.get(img_url).content)

            #showing image_throhugh cv2
            file = cv2.imread('title.text + ".jpeg"')
            plt.imshow(file)
        except:
            print("image not found")

        cast = html.findAll("div",{"class":"cast-item media inlineBlock"})
        print("\n\n")
        print("************CAST:************")
        for i in range(len(cast)):

            print(cast[i].span.text.strip())



        print("\n\n")
        print("************CRITICS____REVIEWS************")

        review = html.findAll("blockquote")
        for i in range(len(review)):

            print("review",[i+1])
            print(review[i].p.text.strip())
            print("\n")


    except:
        print("movie not found")
def rev(movie):
    r_list = []
    url = "https://www.rottentomatoes.com/m/{}".format(movie)

    page = requests.get(url)

    html = BeautifulSoup(page.content, 'html.parser')
    review = html.findAll("blockquote")
    for i in range(len(review)):
        r_list.append(review[i].p.text.strip())
    return r_list





#print("_____________MOVIE REVIEW AND ANALYSIS___________________")
#agmovie_ = input("""ENTER THE MOVIE:""")

#extract(movie_)
#list = rev(movie_)
#print(type(list[0]))
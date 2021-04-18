import json
import math
import operator
import pickle
from os.path import join
import cgi
import pprint
import pandas as pd
import numpy as np
import sys
from task3 import task3
from sklearn.decomposition import PCA


def plotTheResultInChrome(relevantImages, irrelevantImages, finalResult, iteration, typeOfAlgo):
    st = "<div style='clear:both;'></div>"
    st = st + "<h2> " + str('Iteration '+str(iteration)) + "</h2></br>"
    st = st + "<h2> " + str(len(relevantImages)) + " Relevant Images</h2>"
    for i in relevantImages:
        st = st + "<div class='images'>"
        st = st + "<div style='text-align:center;'><span style='font-weight:bold;'>" + i + "</span></div></div>"
    st = st + "</br><div style='clear:both;'></div>"
    st = st + "<h2> " + str(len(irrelevantImages)) + " Irrelevant Images</h2>"
    for i in irrelevantImages:
        st = st + "<div class='images'>"
        st = st + "<div style='text-align:center;'><span style='font-weight:bold;'>" + i + "</span></div></div>"
    st = st + "</br><div style='clear:both;'></div>"
    st = st + "<h2> Results :</h2>"
    for img in finalResult:
        news = "<div class='images'>"
        news = news + "<div style='text-align:center;'><span>" + img + "</span></div></div>"
        st = st + news

    st = st + "</div>"
    # st = st + "<form name=\"search\" action=\"task4.py\" method=\"get\">" \
    #         "Search: <input type=\"text\" name=\"searchbox\">" \
    #         "<input type=\"submit\" value=\"Submit\">" \
    #         "</form> "

    f = open("task4.html", "a")
    f.write(st)
    f.close()

    import webbrowser

    url = "task4.html"
    chrome_path = 'C:/Program Files (x86)/Google/Chrome/Application/chrome.exe %s'
    webbrowser.get(chrome_path).open(url)
    # form = cgi.FieldStorage()
    # ch = form.getvalue('searchbox')
    # return ch


def genLSH(a, imagesNames,gs,red_df,file_dct):
    for i in imagesNames:
        results = gs.query(red_df.iloc[file_dct[i], 1:])
        l = []
        for j in results:
            l.append(results[j])
        a[i] = l
    return a


if __name__ == '__main__':
    # f = open("task4.html", "w")
    # st = "<style>" \
    #     ".images { width:80px;height:60px;float:left;margin:20px;}" \
    #     "img{width:80px;height:60px;}" \
    #     "h2{ text-align: center;margin-top: 30px;}" \
    #     "</style>"
    # f.write(st)
    # f.close()
    # data = ''
    # with open('./task3/output/task3output.txt', "r") as f:
    #     data = f.read()
    # d = data.split('\n')
    # imagesNames = []
    # reducerObjectpp = []
    # for i in d:
    #     if i != '':
    #         imagesNames.append(i[i.find('<') + 1: i.rfind(',')].replace(' ', ''))
    #         reducerObjectpp.append(float(i[i.find(',') + 1: i.rfind('>')].replace(' ', '')))
    # print(imagesNames)
    # print(reducerObjectpp)
    #
    # data_df = pd.read_csv("output\\all_tfidf_gestures.csv", index_col=0)
    # pca = PCA(n_components=len(data_df) // 3, svd_solver='full')
    # red_df = pd.DataFrame(pca.fit_transform(data_df), index=data_df.index)
    # file_lst = list(red_df.index.values)
    # file_dct = {file_lst[i][:-4]: i for i in range(len(file_lst))}
    # red_df.reset_index(inplace=True)
    # print("file_dct",file_dct)
    # gs = task3.GestureSearch(11, 10, len(red_df.iloc[0]) - 1, red_df)
    # gs.train()
    # a = {}
    #
    # a = genLSH(a, imagesNames,gs,red_df,file_dct)
    # useful = []
    # for i in imagesNames:
    #     useful.append(file_dct[i])
    # #pprint.pprint(a)
    # for i in a:
    #     z = a[i]
    #     y = []
    #     for j in useful:
    #         y.append(z[j])
    #     a[i] = y
    # #pprint.pprint(a)
    #
    # # exit(1)
    #
    # # mi = min(reducerObjectpp)
    # # ma = max(reducerObjectpp)
    # # for j in range(len(reducerObjectpp)):
    # #     reducerObjectpp[j] = (reducerObjectpp[j]-mi)/(ma-mi)
    # # print(reducerObjectpp)
    #
    # # images_df = {}
    # # for i in range(len(imagesNames)):
    # #     images_df[imagesNames[i]] = reducerObjectpp[i]


    infile = open('task4pre', 'rb')
    a = pickle.load(infile)
    infile.close()
    infile1 = open('task4pre1', 'rb')
    imagesNames = pickle.load(infile1)
    infile1.close()
    relevantImages = set()
    irrelevantImages = set()
    iteration = 0
    dat = a
    ch = "n"
    relavantImagesResult = set()
    irrelavantImagesResult = set()
    while ch == "n" or ch == "N":

        iteration = iteration + 1
        numberOfRelavant = int(input("Number of relevant images "))
        numberOfIrRelavant = int(input("Number of irrelevant images "))
        for i in range(numberOfRelavant):
            relevantImages.add(input("Please enter relevant image " + str(i + 1)))
        for i in range(numberOfIrRelavant):
            irrelevantImages.add(input("Please enter irrelevant image " + str(i + 1)))

        threshold = 6
        nQuery = []
        for q in range(len(dat)):
            nq = 0
            rq = 0
            irq = 0
            for column in dat:
                if dat[column][q] >= threshold:
                    nq += 1
                    if column in relevantImages:
                        rq += 1
                    if column in irrelevantImages:
                        irq += 1

            pq = (rq + nq / len(dat)) / (len(relevantImages) + 1)
            #print('pq', pq)
            uq = (irq + nq / len(dat)) / (len(irrelevantImages) + 1)
            #print('uq', uq)
            if pq * (1 - uq) / (uq * (1 - pq) + 1) <= 0:
                nQuery.append(0)
            else:
                q = math.log((pq * (1 - uq)) / (uq * (1 - pq)), 10)
                #print('q', q)
                if q < 0:
                    nQuery.append(0)
                elif q > 1:
                    nQuery.append(1)
                else:
                    nQuery.append(q)
        # end for
        # print('nq',nq,'rq',rq,'irq',irq)
        finalResult = {}
        print(nQuery)
        for i in range(len(imagesNames)):
            product = np.dot(nQuery, dat[imagesNames[i]])
            finalResult[imagesNames[i]] = product
        print(finalResult)
        sortList = sorted(finalResult.items(), key=lambda x: x[1], reverse=True)
        finalResult = list(dict(sortList).keys())
        print(finalResult)
        # dat = genLSH(a, finalResult)
        for i in finalResult:
            dat[i] = a[i]
        lab = []
        for i in finalResult:
            if '_' in i:
                j = i.split('_')
                i = j[0]
            if 1 < int(i) < 200:
                lab.append(i + '-vattene')
            elif 200 < int(i) < 300:
                lab.append(i + '-combinato')
            elif 400 < int(i) < 600:
                lab.append(i + '-daccordo')
        relavantImagesResult.update(relevantImages)
        irrelavantImagesResult.update(irrelevantImages)
        plotTheResultInChrome(relavantImagesResult, irrelavantImagesResult, lab, iteration, "Probabilistic")

        ch = input("Are you satisfied with the output? type Y for exit N for running again ")

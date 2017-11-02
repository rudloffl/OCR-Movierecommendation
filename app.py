#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask


#Scikit packages
import numpy as np
import pandas as pd
from sklearn import metrics
#import json
from flask import jsonify
from sklearn.cluster import KMeans




def recommendation(movieid):

    labels = np.genfromtxt('labelskmean.csv', delimiter=',')
    X_projected = np.genfromtxt('pcaresult.csv', delimiter=',')
    datasettitle = pd.read_csv('movie-info.csv', sep=",")
    
    numrecommendation = 5


    #Converts the movieid in np.array code
    arrayid = datasettitle.index.get_loc(movieid)

    #Clustering Kmeans
    numcluster = 20


    kmeans = KMeans(n_clusters=numcluster)
    kmeans.fit(X_projected)

    cluster = kmeans.predict(X_projected[arrayid].reshape(1, -1))[0]

    #array ID for predictions
    candidates = [index for index, clust in enumerate(labels)if clust == cluster and index != arrayid]


    moviedist = []
    Y = X_projected[arrayid].reshape(1, -1)
    #array ID
    for candidate in candidates:
        X = X_projected[candidate].reshape(1, -1)
        distance = metrics.pairwise_distances(X, Y)[0][0]
        #movie ID calculation
        moviecode = datasettitle.index[candidate]
        moviedist.append([distance, moviecode])

    df = pd.DataFrame(moviedist)
    df.columns = ['distance', 'ID']
    df.index = df['ID']
    df = df.drop('ID', axis = 1)
    df = df.sort_values('distance')

    counter = 1
    results = []

    #print('\n##### SUGGESTIONS FOR {} #####\n'.format(datasettitle.loc[movieid]))

    #Movie ID
    for IDmovie in df.index[:numrecommendation]:
        title = datasettitle.loc[IDmovie]['movie_title']
        title = title.replace(u'\xa0', u'')
        while title[0] == ' ':
            title = title[1:]
        while title[-1] == ' ':
            title = title[:-1]
        #print('### SUGGESTION {} ###'.format(counter))
        #print('{} - {}'.format(IDmovie, title))
        distance = df['distance'].loc[IDmovie]
        #print('distance = {}\n'.format(distance))
        results.append({'id': int(IDmovie) , 'name': title})
        counter += 1
    #print(results)
    return {'_results':results}

def allmovies():
    toreturn = ''
    for IDmovie in datasettitle.index:
        title = datasettitle.loc[IDmovie]['movie_title']
        title = title.replace(u'\xa0', u'')
        while title[0] == ' ':
            title = title[1:]
        while title[-1] == ' ':
            title = title[:-1]
        year = datasettitle.loc[IDmovie]['title_year']
        newline = 'ID:{} - Year:{} - title:{} /n'.format(IDmovie, year, title)
        toreturn = toreturn + newline
    return toreturn






app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world !"

@app.route('/all')
def listmovies():
    return allmovies()

@app.route('/recommend/<movieid>')
def recommendmovie(movieid):
    return jsonify(recommendation(int(movieid)))


if __name__ == '__main__':

    app.run(debug=True)

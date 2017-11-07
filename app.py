#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask


#Scikit packages
import numpy as np
import pandas as pd
from sklearn import metrics
from flask import jsonify





def recommendation(movieid):
    
    numrecommendation = 5


    #Converts the movieid in np.array code
    try:
        arrayid = datasettitle.index.get_loc(movieid)
    except:
        return 'The ID is not existing in the dataset'




    cluster = datalabels[movieid]

    #array ID for predictions
    candidates = [index for index, clust in enumerate(datalabels.values) if clust == cluster and index != arrayid]


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

    print('\n##### SUGGESTIONS FOR {} #####\n'.format(datasettitle.loc[movieid]))

    #Movie ID
    for IDmovie in df.index[:numrecommendation]:
        title = datasettitle[IDmovie]
        title = title.replace(u'\xa0', u'')
        while title[0] == ' ':
            title = title[1:]
        while title[-1] == ' ':
            title = title[:-1]
        print('### SUGGESTION {} ###'.format(counter))
        print('{} - {}'.format(IDmovie, title))
        distance = df['distance'].loc[IDmovie]
        print('distance = {}\n'.format(distance))
        results.append({'id': int(IDmovie) , 'name': title})
        counter += 1
    return {'_results':results}

def allmovies():
    toreturn = ''
    for IDmovie in datasettitle.index:
        title = datasettitle.loc[IDmovie]
        title = title.replace(u'\xa0', u'')
        while title[0] == ' ':
            title = title[1:]
        while title[-1] == ' ':
            title = title[:-1]
        newline = 'ID : {:0004d}  <-->  title : {} <br/>'.format(IDmovie, title)
        toreturn = toreturn + newline
    return toreturn


#dataset loading
dataapi = pd.read_csv('dataapi.csv', sep=',', index_col = 0)
datasettitle = dataapi['movie_title']
datalabels = dataapi['Labels']
X_projected = dataapi.drop(['movie_title', 'Labels'],axis = 1).values



app = Flask(__name__)



@app.route('/')
def index():
    return "Welcome to the movie recommendation system please use the recommendation tool or listmovies"

@app.route('/listmovies')
def listmovies():
    return allmovies()

@app.route('/recommend/<movieid>')
def recommendmovie(movieid):
    try:
        return jsonify(recommendation(int(movieid)))
    except:
        return 'Make sure to use an integer to identify the movie'


if __name__ == '__main__':
    app.run(debug=True)
    #recommendation(1)

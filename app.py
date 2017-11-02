#! /usr/bin/python
# -*- coding:utf-8 -*-

from flask import Flask


#Scikit packages
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn import preprocessing
from sklearn import decomposition
from flask import jsonify
from sklearn.cluster import KMeans




def recommendation(movieid):
    
    numrecommendation = 5


    #Converts the movieid in np.array code
    try:
        arrayid = datasettitle.index.get_loc(movieid)
    except:
        return 'The ID is not existing in the dataset'




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

    print('\n##### SUGGESTIONS FOR {} #####\n'.format(datasettitle.loc[movieid]))

    #Movie ID
    for IDmovie in df.index[:numrecommendation]:
        title = datasettitle.loc[IDmovie]
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
dataset = pd.read_csv('movie_metadata_cleaned.csv', sep=",")
dataset.set_index('Unnamed: 0', inplace=True)
datasettitle = dataset['movie_title']
dataset = dataset.fillna(-1)
dataset = dataset.drop('movie_title', axis=1)

#PCA creation
tostudy = []
tostudy.extend([x for x in dataset.columns.values if x.endswith('code')])
tostudy.extend([x for x in dataset.columns.values if x.endswith('likes')])
tostudy.extend([x for x in dataset.columns.values if x.startswith('keyword-')])
tostudy.extend([x for x in dataset.columns.values if x.startswith('gen-')])
tostudy.extend([x for x in dataset.columns.values if x.startswith('movie-')])
tostudy.extend([x for x in dataset.columns.values if x.startswith('money-')])
tostudy.extend([x for x in dataset.columns.values if x.startswith('title-')])
tostudy.extend(['num_user_for_reviews', 'num_voted_users', 'facenumber_in_poster', 'num_critic_for_reviews', 'imdb_score'])
tostudy.extend([ 'duration', 'numrating', 'title_year', 'style-color'])
Components = 90

#Data centering
X  = dataset[tostudy]
std_scale = preprocessing.StandardScaler().fit(X)
X_scaled = std_scale.transform(X)
pca = decomposition.PCA(n_components=Components)
pca.fit(X_scaled)

# projeter X sur les composantes principales
X_projected = pca.transform(X_scaled)


#Clustering Kmeans
numcluster = 20
kmeans = KMeans(n_clusters=numcluster)
kmeans.fit(X_projected)
labels = kmeans.labels_





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

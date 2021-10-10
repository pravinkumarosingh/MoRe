# MoRe

### Movie Recommendation System Using Cosine Similarity

Based on the past user behavior, MoRe recommends the movies to users based on their similarity. It suggests movies to users with a recommendation rate that is greater than the preference rate of movie for the same user. So in core words it will give recommendations which are never liked by other, but a user might like that.

<p align="center">
    <img src="https://cdn.pixabay.com/photo/2017/06/02/22/01/dog-2367414_1280.png" width="200" height="200">
 </p>
Photo By - [mohamed_hassan](https://pixabay.com/users/mohamed_hassan-5229782/)

## Table of Content
- [Introduction to Recommendation System](#introduction-to-recommendation-system)
- [Cosine Similarity](#cosine-similarity)
- [Code](#code)

#### Introduction to Recommendation System
Recommendation systems are the systems that are designed to recommend things to user based on many different factors. These system predict things that users are more likely to purchase or interested in it. Giant companies Google, Amazon, Netflix use recommendation system to help their users to purchase products or movies for them. Recommendation system recommends you the items based on past activities this is known as __Content Based Filtering__ or the preference of the other user's that are to similar to you this is known as __Collaborative Based Filtering__ .

#### Cosine Similarity 
Cosine similarity is a metric used to measure how similar two items are. Mathematically it calculates the cosine of the angle between two vectors projected in a multidimensional space. Cosine similarity is advantageous when two similar documents are far apart by Euclidean distance(size of documents) chances are they may be oriented closed together. The smaller the angle, higher the cosine similarity.
```
1 - cosine-similarity = cosine-distance
```

![cosine-sim](https://github.com/garooda/Movie-Recommendation-Sysetm/blob/main/images/cosine%20sim%20%201.PNG)

![cos-form](https://bit.ly/33baNhZ)

#### Code
Jupyter python notebook is available at  [nbviewer](https://nbviewer.jupyter.org/github/garooda/Movie-Recommendation-Sysetm/blob/main/movie_recommendation_system.ipynb).

Download the dataset from [here](https://github.com/MahnoorJaved98/Movie-Recommendation-System/blob/main/movie_dataset.csv)

##### Importing the important libraries

```python3
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
```
##### Loading the dataset and converting it into dataframe using pandas

```python3
df = pd.read_csv("movie_dataset.csv")
```

##### Features list 
we'll choose the features that are most relevant to us and store it in the list name __features__ .

```python3
features = ['keywords', 'cast', 'genres', 'director']
```

##### Removing null values
Data preprocessing is needed before proceeding further. Hence all the null values must be removed.

```python3
for feature in features:
    df[feature] = df[feature].fillna('')
```

##### Combined features 
combining all the features in the single feature and difference column to the existing dataset.

```python3
def combined_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']

df['combined_features'] = df.apply(combined_features,axis = 1)
```

##### Extracting features

now we'll extract the features by using sklearn's __feature_extraction__ module it helps us to extract feature into format supported by machine learning algorithms. 

__CountVetcorizer()'s__  _*fit_transform*_ we'll help to count the number of the text present in the document.

```python3
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])
print("Count Matrix: ",count_matrix.toarray())
```

##### Cosine similarity 
sklearn has the module cosine_similarity which we'll use to compute the similarity between two vectors.

```python3
cosine_sim = cosine_similarity(count_matrix)
```
__cosine_sim__ is a numpy array with calculated cosine similarity between tw movies

##### Content user like as we are building content based filtering. 
Now we'll take the input movies in the __movie_user_like__ variable. Since we're building content based recommendation system we need to know the the content user like in order to predict the similar.

```python3
movie_user_like = "Dead Poets Society"

def get_index_from(title):
    return df[df.title == title]["index"].values[0]

movie_index = get_index_from(movie_user_like)
```
##### Generating similar movies matrix

```python3
similar_movies = list(enumerate(cosine_sim[movie_index]))
```

##### Sorting the similar movies in descending order

```python3
sorted_similar_movies = sorted(similar_movies, key = lambda x:x[1], reverse = True)
```

##### Printing the similar movies

```python3
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

i=0
for movies in sorted_similar_movies:
    print(get_title_from_index(movies[0]))
    i = i+1;
    if i>15:
        break
```

![Final Output](https://github.com/garooda/Movie-Recommendation-Sysetm/blob/main/images/output.PNG)

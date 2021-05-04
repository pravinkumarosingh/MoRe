# Movie-Recommendation-Sysetm

### Movie Recommendation System Using Cosine Similarity

![Logo](https://cdn.pixabay.com/photo/2017/06/02/22/01/dog-2367414_1280.png)
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
##### loading the dataset and converting it into dataframe using pandas

```python3
df = pd.read_csv("movie_dataset.csv")
```

##### features list 
we'll choose the features that are most relevant to us and store it in the list name __features__ .

```python3
features = ['keywords', 'cast', 'genres', 'director']
```

##### removing null values
Data preprocessing is needed before proceeding further. All the null values must be removed.

```python3
for feature in features:
    df[feature] = df[feature].fillna('')
```

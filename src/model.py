from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors

# Vectorize text using CountVectorizer
def vectorize(docs, max_features=500):
    vectorizer = CountVectorizer(max_features=max_features, stop_words='english')
    X = vectorizer.fit_transform(docs).toarray()
    return X, vectorizer

# Train NearestNeighbors model
def train_knn(X, n_neighbors=10, algorithm='ball_tree'):
    model = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm)
    model.fit(X)
    return model

#  Food Recommendation System

---

##  Overview

This project builds a food recommendation system that suggests products to users based on their previous reviews. If a user dislikes a product, the system suggests alternatives. If they like something, it finds similar items or variations.

The project supports two types of recommendations:

- **Product-Based:** Recommends similar products to one the user has reviewed.
- **User-Based:** Finds similar users and suggests products they've liked.

---

##  Dataset Description

The data comes from the [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews) and includes:

- 568,454 reviews  
- 256,059 users  
- 74,258 products  
- Time span: 1999–2012

Key columns used:
- `ProductId`
- `UserId`
- `Score`
- `Summary`
- `Text`
- `HelpfulnessNumerator` / `HelpfulnessDenominator`

---

##  Data Preprocessing

- Filter out products with fewer than 100 reviews.
- Create a `HelpfulnessRatio`.
- Combine `Summary` and `Text` for each product/user.
- Clean and tokenize text using `nltk`.
- Use `CountVectorizer` to convert text to feature vectors.

---

##  Model

We use `scikit-learn`’s `NearestNeighbors`:

- **Vectorization:** `CountVectorizer` with `max_features=500` and English stop words removed.
- **Model type:** K-Nearest Neighbors (KNN)
- **Algorithm:** `ball_tree`
- **Neighbors:** 10 (`n_neighbors=10`)

---

## Visualizations

![MLRepoFR](/images/BallTree.png)  
![MLRepoFR](/images/HelpfulnessRatio.png)  
![MLRepoFR](/images/ReviewLengthvs.Score.png)  
![MLRepoFR](/images/ReviewScoresDist.png)  


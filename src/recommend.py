import numpy as np

# Recommend similar products based on product ID
def recommend_products(product_id, df, X, model, k=10):
    if product_id not in df['ProductId'].values:
        print(f"Product ID {product_id} not found.")
        return []

    idx = df[df['ProductId'] == product_id].index[0]
    vect = X[idx].reshape(1, -1)
    _, indices = model.kneighbors(vect)

    recommendations = []
    for i in range(1, min(k + 1, len(indices[0]))):
        rec_idx = indices[0][i]
        pid = df['ProductId'].iloc[rec_idx]
        score = df['Score'].iloc[rec_idx]
        recommendations.append((pid, score))
    return recommendations

# Recommend products based on similar users
def recommend_users(user_id, df, X, model, k_users=3, k_products=5):
    if user_id not in df['UserId'].values:
        print(f"User ID {user_id} not found.")
        return {}

    idx = df[df['UserId'] == user_id].index[0]
    vect = X[idx].reshape(1, -1)
    _, indices = model.kneighbors(vect)

    recs = {}
    for i in range(1, min(k_users + 1, len(indices[0]))):
        uid = df['UserId'].iloc[indices[0][i]]
        top_products = df[df['UserId'] == uid].sort_values('Score', ascending=False)['ProductId'].head(k_products).tolist()
        recs[uid] = top_products
    return recs

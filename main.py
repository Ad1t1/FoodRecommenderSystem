from src import preprocess, model, recommend

# Run product recommendation
def run_product_mode():
    df = preprocess.preprocess_data('data/reviews.csv')
    X, _ = model.vectorize(df['Combined'])
    knn = model.train_knn(X)
    recs = recommend.recommend_products('B00004CXX9', df, X, knn, k=5)
    print("Recommended Products:")
    for pid, score in recs:
        print(f"{pid} (Score: {score})")

# Run user recommendation
def run_user_mode():
    df = preprocess.preprocess_data('data/reviews.csv', group_by='UserId')
    X, _ = model.vectorize(df['Combined'])
    knn = model.train_knn(X)
    recs = recommend.recommend_users('A17HMM1M7T9PJ1', df, X, knn)
    print("User-based Recommendations:")
    for user, products in recs.items():
        print(f"{user}: {products}")

if __name__ == "__main__":
    run_product_mode()
    # run_user_mode()  # Uncomment this to test user-based recommendations

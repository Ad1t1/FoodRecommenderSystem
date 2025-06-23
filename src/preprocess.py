import pandas as pd
import re
import nltk
print("NLTK version:", nltk.__version__)
nltk.download('punkt')


nltk.download('punkt')

# Clean and tokenize text
def cleanup(text):
    text = text.lower()
    text = re.sub(r'[^a-z]+', ' ', text)
    return " ".join(nltk.word_tokenize(text.strip()))

# Preprocess reviews by product or user
def preprocess_data(path, group_by='ProductId', min_reviews=100):
    df = pd.read_csv(path)
    df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
    df['Count'] = df.groupby(group_by)[group_by].transform('count')
    df = df[df['HelpfulnessRatio'].notna() & (df['Count'] >= min_reviews)]

    df_grouped = df.groupby(group_by).agg({
        'Summary': lambda x: list(x),
        'Text': lambda x: ' '.join(x),
        'Score': 'mean',
        'HelpfulnessRatio': 'mean'
    }).reset_index()

    df_grouped['Summary'] = df_grouped['Summary'].apply(lambda lst: [cleanup(s) for s in lst])
    df_grouped['Text'] = df_grouped['Text'].apply(cleanup)
    df_grouped['Combined'] = df_grouped['Text'] + ' ' + df_grouped['Summary'].apply(lambda x: ' '.join(x))
    return df_grouped

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import difflib

def load_data(file='books.csv'):
    try:
        df = pd.read_csv(file, on_bad_lines='skip')
    except Exception as e:
        print("Error:", e)
        exit()
    df['title'] = df['title'].fillna('')
    df['authors'] = df['authors'].fillna('')
    return df

def make_matrix(df):
    feats = []
    for i in range(len(df)):
        feats.append(df.loc[i, 'title'] + " " + df.loc[i, 'authors'])
    vec = TfidfVectorizer(stop_words='english')
    mat = vec.fit_transform(feats)
    sim = cosine_similarity(mat)
    return sim

def get_index(name, df):
    titles = df['title'].tolist()
    match = difflib.get_close_matches(name, titles, n=1, cutoff=0.5)
    if match:
        return df[df['title'] == match[0]].index[0]
    return None

def suggest(name, df, sim, n=5):
    idx = get_index(name, df)
    if idx is None:
        print("Book not found.")
        return
    scores = list(enumerate(sim[idx]))
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
    print("\nSimilar books to:", name)
    for i in sorted_scores:
        t = df.loc[i[0], 'title']
        a = df.loc[i[0], 'authors']
        print(f"- {t} by {a} ({round(i[1], 2)})")

def main():
    print("Book Recommender")
    name = input("Enter book name: ").strip()
    df = load_data()
    sim = make_matrix(df)
    suggest(name, df, sim)

if __name__ == '__main__':
    main()


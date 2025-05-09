import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def loadbook(filepath='books.csv'):
    try:
        data = pd.read_csv(filepath)
    except Exception as e:
        print(f"Failed to read the data file: {e}")
        exit()
    data['title'] = data['title'].fillna('')
    data['authors'] = data['authors'].fillna('')
    return data

def buildmatrix(data):
    data['features'] = data.apply(lambda row: f"{row['title']} {row['authors']}", axis=1)
    vect = TfidfVectorizer(stop_words='english')
    tfidf= vect.fit_transform(data['features'])
    sim_matrix = cosine_similarity(tfidf)
    return sim_matrix

def find_index(title, data):
    match = data[data['title'].str.lower() == title.lower()]
    return match.index[0] if not match.empty else None

def reco_books(title, data, sim_matrix, top_n=5):
    index = find_index(title, data)
    if index is None:
        print(f"\nThe book with title '{title}' was not found in the dataset.")
        return

    sim_score = list(enumerate(sim_matrix[index]))
    sort_scores = sorted(sim_score, key=lambda x: x[1], reverse=True)[1:top_n+1]

    print(f"\nBooks similar to '{title}':\n")
    for i, score in sort_scores:
        sim_title = data.iloc[i]['title']
        author = data.iloc[i]['authors']
        print(f"- {sim_title} by {author} (Similarity: {round(score, 2)})")

def main():
    print("==Book Recommendation System==")
    title_inp = input("Enter a book title:\n> ").strip()

    data = loadbook()
    sim_matrix = buildmatrix(data)
    data.reset_index(inplace=True)  
    reco_books(title_inp, data, sim_matrix)

if __name__ == "__main__":
    main()


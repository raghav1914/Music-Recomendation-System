import pandas as pd 
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')


dataset = pd.read_csv("spotify_millsongdata.csv")
#print(dataset)

#Print first 5 rows
print("First five rows", dataset.head(5))

#See last 5 rows
print("Last 5 rows", dataset.tail(5))

#See the shape 
print("Shape of the dataset", dataset.shape)

#Check if theres any null values
print("Null values", dataset.isnull().sum())       # returns the number of missing values in the dataset.

#Dropping link column
dataset = dataset.sample(5000).drop('link', axis = 1).reset_index(drop = True)
print(dataset.head(10))

'''
#Displaying the lyrics from the 'text' column 
print(dataset['text'][0])
'''



#Tokenization and Stemming function
stemmer = PorterStemmer()
def token(txt):
    token = nltk.word_tokenize(txt)
    stemmed_token = [stemmer.stem(w) for w in token]
    return " ".join(stemmed_token)

#Applying tokenization and stemming to the 'text' column
dataset['text'] = dataset['text'].apply(lambda x: token(x))
    
tfid = TfidfVectorizer(analyzer = 'word', stop_words = 'english')
matrix = tfid.fit_transform(dataset['text'])
similar = cosine_similarity(matrix)
print("Matrix", matrix)

#Function to get the index of a song
def get_song_index(song_name):
    song_index = dataset[dataset['song'] == song_name].index
    return song_index

#Recommender Function
def recommender(song_name):
    matching_songs = dataset[dataset['song'] == song_name]
    if not matching_songs.empty:
        idx = matching_songs.index[0]
        distance = sorted(list(enumerate(similar[idx])), reverse = True, key = lambda x: x[1])
        recommended_songs = [dataset.iloc[s_id[0]].song for s_id in distance[1:5]]
        return recommended_songs
    else:
        return f"Song '{song_name}' not found in the dataset"
   

recommendations = recommender('In Your Eyes')
print(recommendations)
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import os

df = pd.read_csv('./data/popcorn/labeledTrainData.tsv', delimiter='\t')


def clean_word(_words, stop=False):
    '''
    문장 전처리 함수 : soup, 특수문자 제외, 불용어 처리

    :param _words: 들어갈 문장 (dtype = str)
    :param stop: 불용어 처리 여부 (dtype = Bool)

    :return: 전처리된 문장 (dtype = str)
    '''
    soup = BeautifulSoup(_words, 'lxml').get_text()
    word = re.sub("[^a-zA-Z]", " ", soup)
    nouns = word.lower().split()

    if stop:
        stops = set(stopwords.words("english"))
        nouns = [w for w in nouns if not w in stops]

    words = " ".join(nouns)

    return words


vectorizer = CountVectorizer(analyzer="word",
                         tokenizer=None,
                         preprocessor=None,
                         stop_words=None,
                         max_features=5000)

review = []
for i in range(len(df)):
    review.append(clean_word(df['review'][i], stop=True))


train_data_features = vectorizer.fit_transform(review)
np.asarray(train_data_features)

forest = RandomForestClassifier(n_estimators=100)
forest = forest.fit(train_data_features, df["sentiment"])

df_test = pd.read_csv('./data/popcorn/testData.tsv', delimiter = '\t')

review_test = []
for i in range(len(df)):
    review_test.append(clean_word(df_test['review'][i], stop=True))

test_data_features = vectorizer.fit_transform(review)
np.asarray(test_data_features)

result = forest.predict(test_data_features)
output = pd.DataFrame(data={"id": df_test["id"], "sentiment": result})

output.to_csv(os.path.join(os.path.dirname(__file__), 'data', 'Bag_of_Words_model.csv'), index=False, quoting=3)

print("Wrote results to Bag_of_Words_model.csv")
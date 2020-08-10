import re
import nltk

import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
from nltk.corpus import stopwords

import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer()

#!/usr/bin/env python

import re
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

train = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'popcorn/labeledTrainData.tsv'),
                    header=0,
                    delimiter="\t", quoting=3)
test = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data', 'popcorn/testData.tsv'),
                   header=0, delimiter="\t",
                   quoting=3)

#train.head(5)


def preprocessing(review, remove_stopwords=False):

    review_text = BeautifulSoup(review, 'lxml').get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words


clean_train_reviews = []
for review in train['review']:
    clean_train_reviews.append(preprocessing(review, True))

clean_train_df = pd.DataFrame({'review': clean_train_reviews, 'sentiment':train['sentiment']})

tokenizer.fit_on_texts(clean_train_reviews)
text_sequences = tokenizer.texts_to_sequences(clean_train_reviews)

print(text_sequences[0])

word_vocab = tokenizer.word_index
print(word_vocab)

len(word_vocab)

data_configs = {}
data_configs['vocab'] = word_vocab
data_configs['vocab_size'] = len(word_vocab)+1

data_configs

max_len = 174
train_inputs = pad_sequences(text_sequences,maxlen=max_len,padding='post')
print(train_inputs.shape)

train_labels = np.array(train['sentiment'])
train_labels.shape


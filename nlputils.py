import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from konlpy.tag import Okt
from konlpy.tag import Mecab
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.models
from gensim.models import Word2Vec
from collections import Counter
from wordcloud import WordCloud
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

# hard-coding
tagger = Mecab(dicpath="C:\\mecab\\mecab-ko-dic")
stop_words ="은 이 것 등 더 를 좀 즉 인 옹 때 만 원 이때 개 일 기 시 럭 갤 성 삼 스 폰 트 드 기 이 리 폴 사 전 마 자 플 블 가 중 북 수 팩 년 월 저 탭"


def convert_bow(sentence, word_to_index):
    """
    sentence를 받아서, word_to_index라는 BoW에 갯수 count

    :param sentence: 문장 (str)
    :param word_to_index: dictonary of word (dict)
    :return: vector list (list)
    """

    # 벡터를 단어의 개수만큼 0으로 초기화
    vector = [0] * (len(word_to_index))

    # 문장을 토큰으로 분리
    tokenizer = Okt()
    tokens = tokenizer.morphs(sentence)

    # 단어의 인덱스 위치에 1 설정
    for token in tokens:
        if token in word_to_index.keys():
            vector[word_to_index[token]] += 1

    return vector


def get_nouns(posts):
    """
    불용어 제거 후 명사 구하기

    :param posts: str_list

    :return: noun_list
    """

    nouns = []
    for post in posts:
        for noun in tagger.nouns(post):
            if noun not in stop_words:
                nouns.append(noun)

    return nouns


def get_top_nouns(nouns, num):
    """
    자주 등장하는 dict 단어 출력
    :param nouns: noun_list
    :param num: 위에서 몇개나 뽑을 것인가

    :return: top nouns (dict)
    """
    # num : numbers for counting
    nouns_counter = Counter(nouns)
    top_nouns = dict(nouns_counter.most_common(num))

    return top_nouns


# 워드 클라우드
def word_cloud(top_nouns, save):
    wc = WordCloud(background_color="white", font_path='./font/NanumBarunGothic.ttf')
    wc.generate_from_frequencies(top_nouns)

    figure = plt.figure()
    figure.set_size_inches(10, 10)
    ax = figure.add_subplot(1, 1, 1)
    ax.axis("off")
    ax.imshow(wc)

    if save:
        # save 구현
        print("saved!")


def show_graph():
    print("yet")


# TF_IDF matrix
def tf_idf(words):
    _tf_idf = TfidfVectorizer()
    tf_idf_matrix = _tf_idf.fit_transform(words)

    return tf_idf_matrix


# word2vec
def word2vec(words):
    okt = Okt()
    tokenized_data = []
    for sentence in words:
        temp_x = okt.morphs(sentence, stem=True)  # 토큰화
        temp_x = [word for word in temp_x if word not in stop_words]  # 불용어 제거
        tokenized_data.append(temp_x)
    model = Word2Vec(sentences=tokenized_data, size=100, window=5, min_count=5, workers=4, sg=0)

    return model


# count vectorization
def count_vector():
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


def draw_histogram_by_token(review_len_by_token):
    plt.figure(figsize=(12, 5))
    plt.hist(review_len_by_token, bins=50, alpha=0.5, color='r', label='word')
    #plt.hist(review_len_by_eumjeol, bins=50, alpha=0.5, color='b', label='alphabet')
    plt.yscale('log', nonposy='clip')
    plt.title('Review Length Histogram')
    plt.xlabel('Review Length')
    plt.ylabel('Number of Reviews')


def token_info(token):
    print('문장 최대 길이: {}'.format(np.max(token)))
    print('문장 최소 길이: {}'.format(np.min(token)))
    print('문장 평균 길이: {:.2f}'.format(np.mean(token)))
    print('문장 길이 표준편차: {:.2f}'.format(np.std(token)))
    print('문장 중간 길이: {}'.format(np.median(token)))
    print('제 1사분위 길이: {}'.format(np.percentile(token, 25)))
    print('제 3사분위 길이: {}'.format(np.percentile(token, 75)))


def draw_count_plot(data):
    import seaborn as sns
    import matplotlib.pyplot as plt
    group = data.value_counts()

    fig, axe = plt.subplots(ncols=1)
    fig.set_size_inches(6, 3)
    sns.coun
    tplot(group)


def preprocessing(review, remove_stopwords=False):

    review_text = BeautifulSoup(review, 'lxml').get_text()
    review_text = re.sub("[^a-zA-Z]", " ", review_text)
    words = review_text.lower().split()
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    return words
import numpy as np
from tqdm import tqdm
from scipy.sparse import lil_matrix, csr_matrix


class BigrammModel:
    """

    Обязательный функционал:
    Предсказание вероятности входного предложения
    Предсказание наиболее вероятных пар ко входному слову
    Продолжение входной фразы словами до заданной длины
    Использование сглаживания Лапласа для поддержки невстреченных ранее слов

    """
    def __init__(self, corpus, laplace=False):

        set_of_words = set(corpus)

        index_word = {}
        word_index = {}

        for index, word in enumerate(set_of_words):
            index_word[index] = word
            word_index[word] = index

        self.index_word = index_word
        self.word_index = word_index

        arr = lil_matrix((len(set_of_words), len(set_of_words)))

        for i in tqdm(range(len(corpus) - 1)):
            first_word, second_word = corpus[i], corpus[i + 1]
            index_first_word, index_second_word = word_index[first_word], word_index[second_word]
            arr[index_first_word, index_second_word] += 1

        if laplace:
            arr += 1
            self.arr_dist = arr.multiply(csr_matrix((arr.sum(axis=1))).power(-1))
        else:
            self.arr_dist = arr.multiply(csr_matrix((arr.sum(axis=1))).power(-1))

    def get_probability(self, sentence):
        proba = 1
        sentence = ['<s>'] + sentence.split() + ['</s>']
        for i in range(len(sentence) - 1):
            first_word, second_word = sentence[i], sentence[i + 1]
            index_first_word, = self.word_index.get(first_word, np.random.randint(0, len(self.word_index)))
            index_second_word = self.word_index.get(second_word, np.random.randint(0, len(self.word_index)))
            proba *= max(self.arr_dist[index_first_word, index_second_word], 0.000001)
        return proba

    def get_nearest_word(self, word):
        return self.index_word[self.arr_dist[self.word_index[word.lower()], :].toarray().argsort()[0][-1]]

    def get_random_nearest_word(self, word):
        top_indexs = self.arr_dist[self.word_index[word.lower()], :].toarray().argsort()[0][-3:]
        random_words = [self.index_word[index] for index in top_indexs]
        return random_words[np.random.randint(0, 3)]

    def continue_sentence(self, sentence, lenght):
        sentence = sentence.split()
        start_index = len(sentence)
        sentence = ['<s>'] + sentence
        while start_index < lenght:
            sentence.append(self.get_nearest_word(sentence[-1].lower()))
            start_index += 1
        return ' '.join(sentence)

    def random_continue_sentence(self, sentence, lenght):
        sentence = sentence.split()
        start_index = len(sentence)
        sentence = ['<s>'] + sentence
        while start_index < lenght:
            sentence.append(self.get_random_nearest_word(sentence[-1].lower()))
            start_index += 1
        return ' '.join(sentence)

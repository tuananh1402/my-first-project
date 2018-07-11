from __future__ import print_function
from keras.models import Sequential
from keras import layers
import numpy as np
from six.moves import range
import re
import glob
from nltk.tokenize import sent_tokenize
import pickle
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer

SOURCE_SENTENCES_FILES = 'source_sentences.p'
SOURCE_VOCABULARY = 'source_vocabulary.p'
DICT_SENTENCES_FILES = 'dict_sentences.p'
DICT_VOCABULARY = 'dict_vocabulary.p'
DATA_PATH = 'data/'
OUTPUT_MODEL_SOCKEYE = 'model_sockeye/'

class VnConverter(object):
    def __init__(self, str):
        self.str = str
        self.maps = [
            ['k(h|H)', 'x'],
            ['K(h|H)', 'X'],
            ['c(?!(h|H))|q', 'k'],
            ['C(?!(h|H))|Q', 'K'],
            ['t(r|R)|c(h|H)', 'c'],
            ['T(r|R)|C(h|H)', 'C'],
            ['d|g(i|I)|r', 'z'],
            ['D|G(i|I)|R', 'Z'],
            ['g(i|ì|í|ỉ|ĩ|ị|I|Ì|Í|Ỉ|Ĩ|Ị)', r'z\1'],
            ['G(i|ì|í|ỉ|ĩ|ị|I|Ì|Í|Ỉ|Ĩ|Ị)', r'Z\1'],
            ['đ', 'd'],
            ['Đ', 'D'],
            ['p(h|H)', 'f'],
            ['P(h|H)', 'F'],
            ['n(g|G)(h|H)?', 'q'],
            ['N(g|G)(h|H)?', 'Q'],
            ['(g|G)(h|H)', 'g'],
            ['t(h|H)', 'w'],
            ['T(h|H)', 'W'],
            ['(n|N)(h|H)', 'n\'']
        ]

    def convert(self):
        for map in self.maps:
            self.str = re.sub(re.compile(map[0]), map[1], self.str)
        return self.str

class DataLoader(object):
    def __init__(self, path = DATA_PATH):
        self.path = path

    def load_source_data(self):
        files = glob.glob('{}*.txt'.format(self.path))
        data = [open(f, 'rb').read().decode('utf-8') for f in files]
        return data

    def load_source_sentences(self):
        source_data = self.load_source_data()
        return [sent.replace('\n', ' ') for d in tqdm(source_data) for sent in sent_tokenize(d)]

    def load_dict_sentences(self):
        source_sentences = self.load_source_sentences()
        return [VnConverter(sent).convert() for sent in source_sentences]

    def dump_vocabulary(self, sentences = None, file = None):
        if sentences and file:
            vectorizer = CountVectorizer()
            vectorizer.fit_transform(sentences)
            pickle.dump(vectorizer, open(file, 'wb'))

    def load_pickle_file(self, file = None):
        if file:
            return pickle.load(open(file, 'rb'))

    def dump_to_txt(self, sentences, filename):
        with open(file=filename, mode='a') as file:
            for sent in sentences:
                file.write(sent + '\n')

if __name__ == '__main__':
    # old_sentences = DataLoader().load_source_sentences()
    new_sentences = DataLoader().load_dict_sentences()
    print(new_sentences[:2])

    # str_convert = VnConverter(str="Xin chào các bạn. Chào mừng các bạn đến với Viblo.").convert()
    # print(str_convert)
    # with open('data/testset_old_sentences.txt', 'r') as f:
    #     lines = f.readlines()

    # print(len(lines))
    # ratio = 0.8
    # DataLoader().dump_to_txt(sentences=old_sentences[:int(len(old_sentences) * ratio)], filename=DATA_PATH + 'train_old_sentences.txt')
    # DataLoader().dump_to_txt(sentences=new_sentences[:int(len(new_sentences) * ratio)], filename=DATA_PATH + 'train_new_sentences.txt')
    # DataLoader().dump_to_txt(sentences=old_sentences[int(len(old_sentences) * ratio):], filename=DATA_PATH + 'test_old_sentences.txt')
    # DataLoader().dump_to_txt(sentences=new_sentences[int(len(new_sentences) * ratio):], filename=DATA_PATH + 'test_new_sentences.txt')


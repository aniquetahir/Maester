from FeatureGenerator import *
import pandas as pd
import numpy as np
import pickle as cPickle
import spacy
from helpers import *
import math
from pyspark import RDD, SparkContext

class NERFeatureGenerator(FeatureGenerator):
    def __init__(self, name='nerFeatureGenerator'):
        super(NERFeatureGenerator, self).__init__(name)

    def process_query(self, df:RDD, sc:SparkContext):
        CM = {
            'Headline': 0,
            'Body ID': 1,
            'articleBody': 2,
            'Headline_unigram': 3,
            'articleBody_unigram': 4,
            'Headline_bigram': 5,
            'articleBody_bigram': 6,
            'Headline_trigram': 7,
            'articleBody_trigram': 8
        }
        cm_index = 9
        print('generating NER features')
        print('for headline')
        #
        # n_train = df[~df['target'].isnull()].shape[0]
        # n_test = df[df['target'].isnull()].shape[0]

        # calculate the polarity score of each sentence then take the averagetext
        # df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x.decode('utf-8')))
        nlp = spacy.load('en_core_web_md')

        def process_ents(x):
            ents = nlp(x).ents
            rst = []
            for ent in ents:
                rst.append(ent.text.lower().replace(" ", ""))
            return rst


        # t_data = df.collect()
        # t_data = [x + [process_ents(x[CM['Headline']])] for x in t_data]
        headline_ents = process_ents(df.take(1)[0][CM['Headline']])
        # df = sc.parallelize(t_data)
        CM['headline_ents'] = cm_index
        df = df.map(lambda x: x + [headline_ents]).cache()
        # df['Headline'].apply(process_ents)
        #
        # df['headline_ents'] = df['headline_ents'].apply(lambda x: x.ents)
        #
        cm_index += 1
        CM['body_ents'] = cm_index

        t_data = df.collect()
        t_data = [x + [process_ents(x[CM['articleBody']])] for x in t_data]
        df = sc.parallelize(t_data)
        # df = df.map(lambda x: x + [process_ents(x[CM['articleBody']])])
        # df['articleBody'].apply(process_ents)
        # df['body_ents'] = df['body_ents'].apply(lambda x: x.ents)

        from sklearn.feature_extraction.text import TfidfVectorizer

        def cat_text(x):
            # res = '%s %s' % (x['Headline'], x['articleBody'])
            ent_str = ''
            for ent in x[CM['body_ents']]:
                ent_str = ent_str + ent + ' '
            return ent_str

        cm_index += 1
        CM['all_ent_text'] = cm_index


        df = df.map(lambda x: x + [cat_text(x)])

        corpus = df.map(lambda x: x[CM['all_ent_text']]).collect()

        #corpus = df["all_ent_text"] = list(df.apply(cat_text, axis=1))

        vectorizer = TfidfVectorizer(max_df=0.99, min_df=1)
        X = vectorizer.fit_transform(corpus)
        idf = vectorizer._tfidf.idf_
        idf_dict = dict(zip(vectorizer.get_feature_names(), idf))

        # tf_dict = {}
        # for line in corpus:
        #     for wd in line.split():
        #         if wd not in tf_dict:
        #             tf_dict[wd] = 0
        #         tf_dict[wd] = tf_dict[wd] + 1

        def naive_ner(row):
            headline_ents = row[CM['headline_ents']]
            body_ents = row[CM['body_ents']]

            set1 = set()
            set2 = set()
            for ent in headline_ents:
                set1.add(ent)
            for ent in body_ents:
                set2.add(ent)
            intersection = set1.intersection(set2)

            if len(intersection) == 0:
                return 0.0
            else:
                return len(intersection) / float(len(set1))

        def tfidf_ner(row):
            headline_ents = row[CM['headline_ents']]
            body_ents = row[CM['body_ents']]

            set1 = set()
            set2 = set()
            for ent in headline_ents:
                for wd in ent.split():
                    set1.add(wd)
            for ent in body_ents:
                for wd in ent.split():
                    set2.add(wd)
            intersection = set1.intersection(set2)

            tf_dict = {}
            for ent in body_ents:
                for wd in ent.split():
                    if wd not in tf_dict:
                        tf_dict[wd] = 0
                    tf_dict[wd] = tf_dict[wd] + 1

            if len(intersection) == 0:
                return 0.0
            else:
                num = 0
                den = 0
                for wd in intersection:
                    if wd not in idf_dict:
                        continue
                    num = num + tf_dict[wd] * idf_dict[wd]
                for wd in set1:
                    if wd not in idf_dict:
                        continue
                    den = den + idf_dict[wd]
                if den == 0:
                    return 0.0
                return num / float(den)

        def logtfidf_ner(row):
            headline_ents = row[CM['headline_ents']]
            body_ents = row[CM['body_ents']]

            set1 = set()
            set2 = set()
            for ent in headline_ents:
                for wd in ent.split():
                    set1.add(wd)
            for ent in body_ents:
                for wd in ent.split():
                    set2.add(wd)
            intersection = set1.intersection(set2)
            tf_dict = {}
            for ent in body_ents:
                for wd in ent.split():
                    if wd not in tf_dict:
                        tf_dict[wd] = 0
                    tf_dict[wd] = tf_dict[wd] + 1

            if len(intersection) == 0:
                return 0.0
            else:
                num = 0
                den = 0
                for wd in intersection:
                    if wd not in idf_dict:
                        continue
                    num = num + math.log(tf_dict[wd] + 1) * idf_dict[wd]
                for wd in set1:
                    if wd not in idf_dict:
                        continue
                    den = den + idf_dict[wd]
                if den == 0:
                    return 0.0
                return num / float(den)

        def idf_ner(row):
            headline_ents = row[CM['headline_ents']]
            body_ents = row[CM['body_ents']]

            set1 = set()
            set2 = set()
            for ent in headline_ents:
                for wd in ent.split():
                    set1.add(wd)
            for ent in body_ents:
                for wd in ent.split():
                    set2.add(wd)
            intersection = set1.intersection(set2)

            if len(intersection) == 0:
                return 0.0
            else:
                num = 0
                den = 0
                for wd in intersection:
                    if wd not in idf_dict:
                        continue
                    num = num + idf_dict[wd]
                for wd in set1:
                    if wd not in idf_dict:
                        continue
                    den = den + idf_dict[wd]
                if den == 0:
                    return 0.0
                return num / float(den)

        cm_index += 1
        CM['naive_ner'] = cm_index
        df = df.map(lambda x: x + [naive_ner(x)])
        # df.apply(naive_ner, axis=1)

        cm_index += 1
        CM['idf_ner'] = cm_index
        df = df.map(lambda x: x + [idf_ner(x)])
        # df.apply(idf_ner, axis=1)

        cm_index += 1
        CM['tfidf_ner'] = cm_index
        df = df.map(lambda x: x + [tfidf_ner(x)])
        # df.apply(tfidf_ner, axis=1)
        cm_index += 1
        CM['logtfidf_ner'] = cm_index
        df = df.map(lambda x: x + [logtfidf_ner(x)])
        # df.apply(logtfidf_ner, axis=1)

        # df = pd.concat([df, df['headline_sents'].apply(lambda x: compute_sentiment(x))], axis=1)

        # print 'df:'
        # print df
        # print df.columns
        # print df.shape
        headlineNer = \
            np.array(
                df.map(lambda x: [x[CM['naive_ner']], x[CM['idf_ner']], x[CM['tfidf_ner']], x[CM['logtfidf_ner']]])
                    .collect())

        print('headlineNer.shape:')
        print(headlineNer.shape)

        # headlineNerTrain = headlineNer[:n_train, :]
        # outfilename_ner_train = "train.headline.ner.pkl"
        # with open(outfilename_ner_train, "wb") as outfile:
        #     cPickle.dump(headlineNerTrain, outfile, -1)
        # print('headline NER features of training set saved in %s' % outfilename_ner_train)
        #
        # if n_test > 0:
        #     # test set is available
        #     headlineNerTest = headlineNer[n_train:, :]
        #     outfilename_ner_test = "test.headline.ner.pkl"
        #     with open(outfilename_ner_test, "wb") as outfile:
        #         cPickle.dump(headlineNerTest, outfile, -1)
        #     print('headline NER features of test set saved in %s' % outfilename_ner_test)

        print('headine ner done')
        return [headlineNer]


    def process(self, df):

        print('generating NER features')
        print('for headline')

        n_train = df[~df['target'].isnull()].shape[0]
        n_test = df[df['target'].isnull()].shape[0]

        # calculate the polarity score of each sentence then take the averagetext
        # df['headline_sents'] = df['Headline'].apply(lambda x: sent_tokenize(x.decode('utf-8')))
        nlp = spacy.load('en_core_web_md')

        def process_ents(x):
            ents = nlp(x).ents
            rst = []
            for ent in ents:
                rst.append(ent.text.lower().replace(" ", ""))
            return rst
        df['headline_ents'] = df['Headline'].apply(process_ents)
        #
        # df['headline_ents'] = df['headline_ents'].apply(lambda x: x.ents)
        #
        df['body_ents'] = df['articleBody'].apply(process_ents)
        # df['body_ents'] = df['body_ents'].apply(lambda x: x.ents)

        from sklearn.feature_extraction.text import TfidfVectorizer

        def cat_text(x):
            # res = '%s %s' % (x['Headline'], x['articleBody'])
            ent_str = ''
            for ent in x['body_ents']:
                ent_str = ent_str + ent + ' '
            return ent_str

        corpus = df["all_ent_text"] = list(df.apply(cat_text, axis=1))

        vectorizer = TfidfVectorizer(max_df=0.99, min_df=1)
        X = vectorizer.fit_transform(corpus)
        idf = vectorizer._tfidf.idf_
        idf_dict = dict(zip(vectorizer.get_feature_names(), idf))

        # tf_dict = {}
        # for line in corpus:
        #     for wd in line.split():
        #         if wd not in tf_dict:
        #             tf_dict[wd] = 0
        #         tf_dict[wd] = tf_dict[wd] + 1

        def naive_ner(row):
            headline_ents = row['headline_ents']
            body_ents = row['body_ents']

            set1 = set()
            set2 = set()
            for ent in headline_ents:
                set1.add(ent)
            for ent in body_ents:
                set2.add(ent)
            intersection = set1.intersection(set2)

            if len(intersection) == 0:
                return 0.0
            else:
                return len(intersection) / float(len(set1))

        def tfidf_ner(row):
            headline_ents = row['headline_ents']
            body_ents = row['body_ents']

            set1 = set()
            set2 = set()
            for ent in headline_ents:
                for wd in ent.split():
                    set1.add(wd)
            for ent in body_ents:
                for wd in ent.split():
                    set2.add(wd)
            intersection = set1.intersection(set2)

            tf_dict = {}
            for ent in body_ents:
                for wd in ent.split():
                    if wd not in tf_dict:
                        tf_dict[wd] = 0
                    tf_dict[wd] = tf_dict[wd] + 1

            if len(intersection) == 0:
                return 0.0
            else:
                num = 0
                den = 0
                for wd in intersection:
                    if wd not in idf_dict:
                        continue
                    num = num + tf_dict[wd] * idf_dict[wd]
                for wd in set1:
                    if wd not in idf_dict:
                        continue
                    den = den + idf_dict[wd]
                if den == 0:
                    return 0.0
                return num / float(den)

        def logtfidf_ner(row):
            headline_ents = row['headline_ents']
            body_ents = row['body_ents']

            set1 = set()
            set2 = set()
            for ent in headline_ents:
                for wd in ent.split():
                    set1.add(wd)
            for ent in body_ents:
                for wd in ent.split():
                    set2.add(wd)
            intersection = set1.intersection(set2)
            tf_dict = {}
            for ent in body_ents:
                for wd in ent.split():
                    if wd not in tf_dict:
                        tf_dict[wd] = 0
                    tf_dict[wd] = tf_dict[wd] + 1

            if len(intersection) == 0:
                return 0.0
            else:
                num = 0
                den = 0
                for wd in intersection:
                    if wd not in idf_dict:
                        continue
                    num = num + math.log(tf_dict[wd] + 1) * idf_dict[wd]
                for wd in set1:
                    if wd not in idf_dict:
                        continue
                    den = den + idf_dict[wd]
                if den == 0:
                    return 0.0
                return num / float(den)

        def idf_ner(row):
            headline_ents = row['headline_ents']
            body_ents = row['body_ents']

            set1 = set()
            set2 = set()
            for ent in headline_ents:
                for wd in ent.split():
                    set1.add(wd)
            for ent in body_ents:
                for wd in ent.split():
                    set2.add(wd)
            intersection = set1.intersection(set2)

            if len(intersection) == 0:
                return 0.0
            else:
                num = 0
                den = 0
                for wd in intersection:
                    if wd not in idf_dict:
                        continue
                    num = num + idf_dict[wd]
                for wd in set1:
                    if wd not in idf_dict:
                        continue
                    den = den + idf_dict[wd]
                if den == 0:
                    return 0.0
                return num / float(den)

        df['naive_ner'] = df.apply(naive_ner, axis=1)

        df['idf_ner'] = df.apply(idf_ner, axis=1)

        df['tfidf_ner'] = df.apply(tfidf_ner, axis=1)
        df['logtfidf_ner'] = df.apply(logtfidf_ner, axis=1)

        # df = pd.concat([df, df['headline_sents'].apply(lambda x: compute_sentiment(x))], axis=1)

        # print 'df:'
        # print df
        # print df.columns
        # print df.shape
        headlineNer = df[['naive_ner', 'idf_ner', 'tfidf_ner', 'logtfidf_ner']].values


        print('headlineNer.shape:')
        print(headlineNer.shape)

        headlineNerTrain = headlineNer[:n_train, :]
        outfilename_ner_train = "train.headline.ner.pkl"
        with open(outfilename_ner_train, "wb") as outfile:
            cPickle.dump(headlineNerTrain, outfile, -1)
        print('headline NER features of training set saved in %s' % outfilename_ner_train)

        if n_test > 0:
            # test set is available
            headlineNerTest = headlineNer[n_train:, :]
            outfilename_ner_test = "test.headline.ner.pkl"
            with open(outfilename_ner_test, "wb") as outfile:
                cPickle.dump(headlineNerTest, outfile, -1)
            print('headline NER features of test set saved in %s' % outfilename_ner_test)

        print('headine ner done')
        return 1

    def read(self, header='train'):

        filename_ner = "%s.headline.ner.pkl" % header
        with open(filename_ner, "rb") as infile:
            headlineNer = cPickle.load(infile)

        print('headlineNer.shape:')
        print(headlineNer.shape)

        return [headlineNer]

        #   Copyright 2017 Cisco Systems, Inc.
        #
        #   Licensed under the Apache License, Version 2.0 (the "License");
        #   you may not use this file except in compliance with the License.
        #   You may obtain a copy of the License at
        #
        #     http://www.apache.org/licenses/LICENSE-2.0
        #
        #   Unless required by applicable law or agreed to in writing, software
        #   distributed under the License is distributed on an "AS IS" BASIS,
        #   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        #   See the License for the specific language governing permissions and
        #   limitations under the License.

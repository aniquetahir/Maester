from FeatureGenerator import *
import ngram
import pickle as cPickle
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from helpers import *
import hashlib
nltk.download('punkt')
from pyspark import SparkContext, RDD
from pyspark.sql import SparkSession, DataFrame





def add_columns(dictionary, cm, df:RDD):
    df.zip()

class CountFeatureGenerator(FeatureGenerator):


    def __init__(self, name='countFeatureGenerator'):
        super(CountFeatureGenerator, self).__init__(name)


    def process_query(self, df:RDD):
        grams = ["unigram", "bigram", "trigram"]
        feat_names = ["Headline", "articleBody"]
        print("generate counting features")
        features = []

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
        cm_index = 8
        for feat_name in feat_names:
            for gram in grams:
                features.append("count_of_%s_%s" % (feat_name, gram))
                cm_index += 1
                CM[features[-1]] = cm_index
                df = df.map(lambda x: x + [len(x[CM[feat_name+'_'+gram]])])
                features.append("count_of_unique_%s_%s" % (feat_name, gram))
                cm_index += 1
                CM[features[-1]] = cm_index
                df = df.map(lambda x: x+ [len(set(x[CM[feat_name+'_'+gram]]))])
                features.append("ratio_of_unique_%s_%s" % (feat_name, gram))
                cm_index += 1
                CM[features[-1]] = cm_index
                df = df.map(lambda x: x + [try_divide(x[CM["count_of_unique_%s_%s"%(feat_name,gram)]], x[CM["count_of_%s_%s"%(feat_name,gram)]])])
                df.cache()

        # overlapping n-grams count
        for gram in grams:
            cm_index += 1
            CM["count_of_Headline_%s_in_articleBody" % gram] = cm_index
            df = df.map(lambda x: x + [sum([1. for w in x[CM["Headline_" + gram]] if w in set(x[CM["articleBody_" + gram]])])])
            cm_index += 1
            CM["ratio_of_Headline_%s_in_articleBody" % gram] = cm_index
            df = df.map(lambda x: x + [try_divide(x[CM["count_of_Headline_%s_in_articleBody" % gram]], x[CM["count_of_Headline_%s" % gram]])])
            df.cache()

        # number of sentences in headline and body
        for feat_name in feat_names:
            #df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x.decode('utf-8').encode('ascii', errors='ignore'))))
            cm_index += 1
            CM['len_sent_%s' % feat_name] = cm_index
            df.map(lambda x: x + [len(sent_tokenize(x[CM[feat_name]]))])
            df.cache()
            # df[feat_name].apply(lambda x: len(sent_tokenize(x)))
            #print df['len_sent_%s' % feat_name]

        # dump the basic counting features into a file
        columns = [x[0] for x in sorted(CM.items(), key=lambda x:x[1])]
        feat_names = [ n for n in columns \
                if "count" in n \
                or "ratio" in n \
                or "len_sent" in n]

        feat_indices = [ i for i, n in enumerate(columns) \
                       if "count" in n \
                       or "ratio" in n \
                       or "len_sent" in n]


        # binary refuting features
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        _hedging_seed_words = [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed'
        ]

        #df['refuting_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #df['hedging_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #check_words = _refuting_words + _hedging_seed_words
        check_words = _refuting_words
        for rf in check_words:
            fname = '%s_exist' % rf
            feat_names.append(fname)
            cm_index += 1
            CM[fname] = cm_index
            df = df.map(lambda x: x + [ 1 if rf in x[CM['Headline']] else 0 ])
            df.cache()
            # df[fname] = df['Headline'].map(lambda x: 1 if rf in x else 0)

        include_indices = []
        return df.map(lambda x: [y for i, y in enumerate(x) if i in feat_indices]).collect()


    def process(self, df):

        grams = ["unigram", "bigram", "trigram"]
        feat_names = ["Headline", "articleBody"]
        print("generate counting features")
        for feat_name in feat_names:
            for gram in grams:
                df["count_of_%s_%s" % (feat_name, gram)] = list(df.apply(lambda x: len(x[feat_name + "_" + gram]), axis=1))
                df["count_of_unique_%s_%s" % (feat_name, gram)] = \
		            list(df.apply(lambda x: len(set(x[feat_name + "_" + gram])), axis=1))
                df["ratio_of_unique_%s_%s" % (feat_name, gram)] = \
                    list(map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)]))

        # overlapping n-grams count
        for gram in grams:
            df["count_of_Headline_%s_in_articleBody" % gram] = \
                list(df.apply(lambda x: sum([1. for w in x["Headline_" + gram] if w in set(x["articleBody_" + gram])]), axis=1))
            df["ratio_of_Headline_%s_in_articleBody" % gram] = \
                list(map(try_divide, df["count_of_Headline_%s_in_articleBody" % gram], df["count_of_Headline_%s" % gram]))
        
        # number of sentences in headline and body
        for feat_name in feat_names:
            #df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x.decode('utf-8').encode('ascii', errors='ignore'))))
            df['len_sent_%s' % feat_name] = df[feat_name].apply(lambda x: len(sent_tokenize(x)))
            #print df['len_sent_%s' % feat_name]

        # dump the basic counting features into a file
        feat_names = [ n for n in df.columns \
                if "count" in n \
                or "ratio" in n \
                or "len_sent" in n]
        
        # binary refuting features
        _refuting_words = [
            'fake',
            'fraud',
            'hoax',
            'false',
            'deny', 'denies',
            # 'refute',
            'not',
            'despite',
            'nope',
            'doubt', 'doubts',
            'bogus',
            'debunk',
            'pranks',
            'retract'
        ]

        _hedging_seed_words = [
            'alleged', 'allegedly',
            'apparently',
            'appear', 'appears',
            'claim', 'claims',
            'could',
            'evidently',
            'largely',
            'likely',
            'mainly',
            'may', 'maybe', 'might',
            'mostly',
            'perhaps',
            'presumably',
            'probably',
            'purported', 'purportedly',
            'reported', 'reportedly',
            'rumor', 'rumour', 'rumors', 'rumours', 'rumored', 'rumoured',
            'says',
            'seem',
            'somewhat',
            # 'supposedly',
            'unconfirmed'
        ]
        
        #df['refuting_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #df['hedging_words_in_headline'] = df['Headline'].map(lambda x: 1 if w in x else 0 for w in _refuting_words)
        #check_words = _refuting_words + _hedging_seed_words
        check_words = _refuting_words
        for rf in check_words:
            fname = '%s_exist' % rf
            feat_names.append(fname)
            df[fname] = df['Headline'].map(lambda x: 1 if rf in x else 0)
	    
        # number of body texts paired up with the same headline
        #df['headline_hash'] = df['Headline'].map(lambda x: hashlib.md5(x).hexdigest())
        #nb_dict = df.groupby(['headline_hash'])['Body ID'].nunique().to_dict()
        #df['n_bodies'] = df['headline_hash'].map(lambda x: nb_dict[x])
        #feat_names.append('n_bodies')
        # number of headlines paired up with the same body text
        #nh_dict = df.groupby(['Body ID'])['headline_hash'].nunique().to_dict()
        #df['n_headlines'] = df['Body ID'].map(lambda x: nh_dict[x])
        #feat_names.append('n_headlines')
        print('BasicCountFeatures:')
        print(df)
        
        # split into train, test portion and save in separate files
        train = df[~df['target'].isnull()]
        print('train:')
        print(train[['Headline_unigram','Body ID', 'count_of_Headline_unigram']])
        xBasicCountsTrain = train[feat_names].values
        outfilename_bcf_train = "train.basic.pkl"
        with open(outfilename_bcf_train, "wb") as outfile:
            cPickle.dump(feat_names, outfile, -1)
            cPickle.dump(xBasicCountsTrain, outfile, -1)
        print('basic counting features for training saved in %s' % outfilename_bcf_train)
        
        test = df[df['target'].isnull()]
        print('test:')
        print(test[['Headline_unigram','Body ID', 'count_of_Headline_unigram']])
        #return 1
        if test.shape[0] > 0:
            # test set exists
            print('saving test set')
            xBasicCountsTest = test[feat_names].values
            outfilename_bcf_test = "test.basic.pkl"
            with open(outfilename_bcf_test, 'wb') as outfile:
                cPickle.dump(feat_names, outfile, -1)
                cPickle.dump(xBasicCountsTest, outfile, -1)
                print('basic counting features for test saved in %s' % outfilename_bcf_test)

        return 1


    def read(self, header='train'):

        filename_bcf = "%s.basic.pkl" % header
        with open(filename_bcf, "rb") as infile:
            feat_names = cPickle.load(infile)
            xBasicCounts = cPickle.load(infile)
            print('feature names: ')
            print(feat_names)
            print('xBasicCounts.shape:')
            print(xBasicCounts.shape)
            #print type(xBasicCounts)

        return [xBasicCounts]


    def read_query(self):
        t = ['train', 'test']
        basic_counts = []
        for typ in t:
            basic_counts.append(self.read(typ))
        return basic_counts



if __name__ == '__main__':

    cf = CountFeatureGenerator()
    cf.read()

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

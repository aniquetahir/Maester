import sys
import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StringType, IntegerType, LongType, ArrayType


from CountFeatureGenerator import CountFeatureGenerator
from NERFeatureGenerator import NERFeatureGenerator
from SvdFeatureGenerator import SvdFeatureGenerator
from TfidfFeatureGenerator import TfidfFtXGBoost
from Word2VecFeatureGenerator import Word2VecFeatureGenerator
import numpy as np

from helpers import *
import ngram

sc:SparkContext = SparkContext.getOrCreate(SparkConf().setMaster('local[*]'))
ss:SparkSession = SparkSession.builder.getOrCreate()

COLUMN_MAP = {
    'Headline': 0,
    'Body ID': 1,
    'articleBody': 2,
    'Headline_unigram': 3,
    'articleBody_unigram': 4,
    'Headline_bigram': 5,
    'articleBody_bigram': 6,
    'Headline_trigram': 7,
    'articleBody_trigram': 8,
    'count_of_Headline_unigram': 9
}


def get_related_articles(query):
    print("generate unigram")
    query_unigram = preprocess_data(query)
    query_bigram = ngram.getBigram(query_unigram, '_')
    query_trigram = ngram.getTrigram(query_unigram, '_')


    # data["Headline_unigram"] = data["Headline"].map(lambda x: preprocess_data(x))
    # data["articleBody_unigram"] = data["articleBody"].map(lambda x: preprocess_data(x))
    #
    # print("generate bigram")
    # join_str = "_"
    # data["Headline_bigram"] = data["Headline_unigram"].map(lambda x: ngram.getBigram(x, join_str))
    # data["articleBody_bigram"] = data["articleBody_unigram"].map(lambda x: ngram.getBigram(x, join_str))
    #
    # print("generate trigram")
    # join_str = "_"
    # data["Headline_trigram"] = data["Headline_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
    # data["articleBody_trigram"] = data["articleBody_unigram"].map(lambda x: ngram.getTrigram(x, join_str))
    #
    test_bodies = pd.read_csv('test_bodies_processed.csv')
    train_bodies = pd.read_csv('train_bodies_processed.csv')

    # merge bodies
    ss.createDataFrame(test_bodies).createOrReplaceTempView('test_bods')
    ss.createDataFrame(train_bodies).createOrReplaceTempView('train_bods')

    num_test = ss.sql('select max(`Body ID`) as num_test from test_bods').collect()[0]['num_test']
    train_incremented = ss.sql(f'select `Body ID`+{num_test+1} as `Body ID`, articleBody from train_bods')
    train_incremented.createOrReplaceTempView('train_bods')

    all_bodies = ss.sql('select * from test_bods union distinct (select * from train_bods)')
    all_bodies.createOrReplaceTempView('all_bods')
    print(all_bodies.count())
    all_bodies = ss.sql('select max(`Body ID`) as `Body ID`, articleBody from all_bods group by articleBody limit 100')
    print(all_bodies.count())
    # get rdd from all_bodies
    bodies_rdd = all_bodies.rdd.map(list)
    bodies_rdd = bodies_rdd.map(lambda x: [query, x[0], x[1], query_unigram, preprocess_data(x[1])])
    bodies_rdd = bodies_rdd.map(lambda x: x +
                                          [query_bigram, ngram.getBigram(x[4],'_'),
                                          query_trigram, ngram.getTrigram(x[4],'_')])

    # cfg = CountFeatureGenerator()
    # cfg.process_query(bodies_rdd)

    features = []
    gen = CountFeatureGenerator()
    features.append(gen.process_query(bodies_rdd))
    tfid_features = TfidfFtXGBoost().process_query(bodies_rdd)
    features.append(tfid_features)
    features.append(SvdFeatureGenerator().process_query(bodies_rdd, tfid_features))
    features.append(Word2VecFeatureGenerator().process_query(bodies_rdd))
    features.append(NERFeatureGenerator().process_query(bodies_rdd, sc))

    new_feats = []
    for f in features:
        for g in f:
            new_feats.append(g)

    features = new_feats

    # generators = [
    #     CountFeatureGenerator(),
    #     TfidfFtXGBoost(),
    #     SvdFeatureGenerator(),
    #     Word2VecFeatureGenerator(),
    #     NERFeatureGenerator()
    # ]
    #
    # features = [f for g in generators for f in g.read("test")]
    print(len(features))

    np.hstack(features)

    print('test')

    pass

if __name__ == '__main__':
    #cf = CountFeatureGenerator()
    #cf.read_query()
    query = 'Obama Orders Fed To Adopt Euro Currency'# sys.argv[1]
    get_related_articles(query)

    pass
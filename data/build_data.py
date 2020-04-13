import os
from collections import Counter
import json
import re
from dateutil.parser import parse
import pickle
import heapq

from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import keras


def sigmoid(value):
    return 1/(1+np.exp(-1*value))


''' Like the word2vec, sampling words depend on its frequency
    Therefore, we need to find a method to rank:
        1. movie = f(vote/max_vote) + f(review_score/10)
        2. restaurant = f(review_count/max_count) + f(star/5)
        3. amazon product = f(review_count/max_count) + f(avg_score/5)
        4. user = f(num_review/max_review)
'''


def rank_bid(review_count, score, max_count, base):
    '''calculate business popularity

        review_count: the number of reviews
        score: average review score
        max_count: max review counts in this category, to normalize the review count
        base: the score base, 10 for movie, 5 for yelp and amazon
    '''
    return sigmoid(review_count/max_count) + sigmoid(score/base)


def preprocess(doc, min_len=10, stopwords=set()):
    '''Split, tokenize documents

        stopwords (set)
    '''
    # lowercase
    doc = doc.lower()
    # replace url
    doc = re.sub(r"https?:\S+", "url", doc)

    doc = doc.replace('\n', ' ')
    doc = doc.replace('\t', ' ')

    # elepsis normalization
    doc = re.sub(r'\.+', '.', doc)
    doc = re.sub(r'!+', '!', doc)
    doc = re.sub(r'\*+', ' ', doc)
    doc = re.sub(r'_+', ' ', doc)
    doc = re.sub(r',+', ',', doc)

    doc = [item.strip() for item in word_tokenize(doc) 
        if len(item.strip())>0 and item not in stopwords
    ] # tokenize

    # examine if the document contains English words
    flag = False
    for word in doc:
        if word.isalpha():
            flag = True
            break

    if flag is False or len(doc) <= min_len:
        return None
    else:
        return ' '.join(doc)


def format_time(time):
    tdate = parse(time)
    return tdate.strftime('%m-%d-%Y')


def extract_yelp(data_dir, save_dir='./raw/yelp/'):
    '''According to the yelp, top categories ranked by counts are
            Restaurant, 59371; Home Services, 19729
            Beauty & Spas', 19370; Health & Medical, 17171
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    products_info = {
        'Restaurants': dict(),
        'Home Services': dict(),
        'Beauty & Spas': dict(),
        'Health & Medical': dict()
    }

    print('Collecting product information in each genre...')
    with open(data_dir + 'business.json') as dfile:
        max_count = dict()
        for genre in products_info:
            max_count[genre] = 0

        for line in dfile:
            entity = json.loads(line)
            if not entity['categories']:
                continue

            # filter out any businesses has less than 20 reviews
            if entity['review_count'] < 20:
                continue

            for genre in products_info:
                if genre in entity['categories']:
                    # define the structure of the business information
                    products_info[genre][entity['business_id']] = dict()
                    products_info[genre][entity['business_id']]['star'] = entity['stars']
                    products_info[genre][entity['business_id']][
                        'review_count'] = entity['review_count']
                    products_info[genre][entity['business_id']]['words'] = set()
                    products_info[genre][entity['business_id']]['uids'] = set()
                    if entity['review_count'] > max_count[genre]:
                        max_count[genre] = entity['review_count']
                    break

    # find sample size, if the size > 3000, take 3000
    sample_size = float('inf')
    for genre in products_info:
        if len(products_info) < sample_size:
            sample_size = len(products_info)
        print(genre, len(products_info[genre]))
    if sample_size > 3000:
        sample_size = 3000

    # normalize count and calculate popularity
    products_probs = dict() # for later sampling
    for genre in products_info:
        products_probs[genre] = []
        for bid in products_info[genre]:
            products_info[genre][bid]['popularity'] = rank_bid(
                products_info[genre][bid]['review_count'], 
                products_info[genre][bid]['star'], 
                max_count[genre], 5.0
            )
            products_probs[genre].append(
                products_info[genre][bid]['review_count']/max_count[genre])
        # normalize probs
        prob_sum = sum(products_probs[genre])
        products_probs[genre] = [item/prob_sum for item in products_probs[genre]]

    # collect users and find frequent tokens
    # we will later remove those tokens in the user and product words.
    print('Collecting user information...')
    users_info = dict()
    top_tokens = Counter()
    with open(data_dir + 'review.json') as dfile:
        for line in dfile:
            entity = json.loads(line)

            # filter out the categories
            flag = True
            for genre in products_info:
                if entity['business_id'] in products_info[genre]:
                    flag = False
                    break
            if flag:
                continue

            # filter out text less than 10 tokens
            entity['text'] = preprocess(entity['text'])
            if not entity['text']:
                continue

            if entity['user_id'] not in users_info:
                users_info[entity['user_id']] = dict()
                users_info[entity['user_id']]['review_count'] = 0
                users_info[entity['user_id']]['words'] = set()
                users_info[entity['user_id']]['bids'] = set()
            users_info[entity['user_id']]['review_count'] += 1

            # count tokens
            for token in entity['text'].split():
                if not token.isalpha():
                    continue
                top_tokens[token] += 1
    top_tokens = dict(
        top_tokens.most_common(10) # top 10 frequentist tokens (letter only)
    )

    # sample some business
    sample_size = float('inf')
    for genre in products_info:
        tmp_size = len(list(products_info[genre].keys()))
        if sample_size > tmp_size:
            sample_size = tmp_size
    if sample_size > 3000:
        sample_size = 3000
    for genre in products_info:
        np.random.seed(33)
        products_keys = np.random.choice(
            list(products_info[genre].keys()), 
            size=sample_size, replace=False,
            p=products_probs[genre]
        )
        products_info[genre] = dict(
            [(key, products_info[genre][key]) for key in products_keys])

    # review file
    rfile = open(save_dir+'yelp.tsv', 'w')
    columns = ['rid', 'bid', 'uid', 'text', 'date', 'genre', 'label']
    rfile.write('\t'.join(columns)+'\n')
    
    # build the review
    print('Building review data...')
    with open(data_dir + 'review.json') as dfile:
        for line in dfile:
            entity = json.loads(line)

            '''Filters'''
            # user control
            user_info = users_info.get(entity['user_id'], None)
            if not user_info:
                continue
            # filter out the user less than 10 reviews
            if user_info['review_count'] < 10:
                del users_info[entity['user_id']]
                continue
            
            # filter out the categories
            flag = True
            for genre in products_info:
                if entity['business_id'] in products_info[genre]:
                    flag = False
                    break
            if flag:
                continue

            # filter out text less than 10 tokens
            entity['text'] = preprocess(
                entity['text'], min_len=10)
            if not entity['text']:
                continue

            '''Data collection'''
            # encode labels
            if entity['stars'] > 3:
                entity['stars'] = 2
            elif entity['stars'] < 3:
                entity['stars'] = 0
            else:
                entity['stars'] = 1

            # collect review data
            line = '\t'.join([
                entity['review_id'], entity['business_id'],
                entity['user_id'], entity['text'], entity['date'].split()[0], 
                genre, str(entity['stars'])
            ])
            rfile.write(line + '\n')

            # collect words for both products and users
            for token in entity['text'].split():
                if not token.isalpha():
                    continue

                if token in top_tokens:
                    continue

                products_info[genre][entity['business_id']]['words'].add(token)
                users_info[entity['user_id']]['words'].add(token)

            # collect purchasing behaviors
            products_info[genre][entity['business_id']]['uids'].add(entity['user_id'])
            users_info[entity['user_id']]['bids'].add(entity['business_id'])

    rfile.flush()
    rfile.close()

    '''save user and product information'''
    print('Saving user information...')
    user_idx = list() 
    product_idx = list()
    with open(save_dir + 'users.json', 'w') as wfile:
        for uid in users_info:
            if len(users_info[uid]['words']) == 0:
                continue

            users_info[uid]['uid'] = uid
            users_info[uid]['words'] = list(users_info[uid]['words'])
            users_info[uid]['bids'] = list(users_info[uid]['bids'])

            wfile.write(json.dumps(users_info[uid]) + '\n')
            heapq.heappush(user_idx, (users_info[uid]['review_count'], uid))
    user_idx_encoder = dict() # a dictionary for user idx mapping 
    init_idx = len(user_idx) # 0 is the reserved idx for unknown
    while init_idx > 0:
        item = heapq.heappop(user_idx)
        user_idx_encoder[item[1]] = init_idx
        init_idx -= 1
    with open(save_dir + 'user_idx.json', 'w') as wfile:
        wfile.write(json.dumps(user_idx_encoder))

    print('Saving product information...')
    with open(save_dir + 'products.json', 'w') as wfile:
        for genre in products_info:
            for bid in products_info[genre]:
                if len(products_info[genre][bid]['words']) == 0:
                    continue

                products_info[genre][bid]['bid'] = bid
                products_info[genre][bid]['genre'] = genre
                products_info[genre][bid]['words'] = list(products_info[genre][bid]['words'])
                products_info[genre][bid]['uids'] = list(products_info[genre][bid]['uids'])
                wfile.write(json.dumps(products_info[genre][bid]) + '\n')
                heapq.heappush(product_idx, (products_info[genre][bid]['popularity'], bid))
    product_idx_encoder = dict() # a dictionary for product idx mapping 
    init_idx = len(product_idx) # 0 is the reserved idx for unknown
    while init_idx > 0:
        item = heapq.heappop(product_idx)
        product_idx_encoder[item[1]] = init_idx
        init_idx -= 1
    with open(save_dir + 'product_idx.json', 'w') as wfile:
        wfile.write(json.dumps(product_idx_encoder))


def extract_amazon(data_dir, save_dir='./raw/amazon/'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    # define genre names, genre value must match file name
    products_info = {
        'Books': dict(),
        'Electronics': dict(),
        'CDs_and_Vinyl': dict(),
        'Home_and_Kitchen': dict()
    }
    users_info = dict()
    top_tokens = Counter()

    '''Collect Product and User information'''
    print('Collecting product and user information in each genre...')
    for genre in products_info:
        print('Working on: ', data_dir+genre+'.json')
        with open(data_dir+genre+'.json') as dfile:
            for line in dfile:
                entity = json.loads(line.strip())

                # check uid, bid, review text
                if len(entity['reviewText']) < 30:
                    continue
                if len(entity['reviewerID']) < 3:
                    continue
                if len(entity['asin']) < 3:
                    continue

                entity['reviewText'] = preprocess(entity['reviewText'])
                if not entity['reviewText']:
                    continue

                # count tokens
                for token in entity['reviewText'].split():
                    if not token.isalpha():
                        continue
                    top_tokens[token] += 1

                # User info
                if entity['reviewerID'] not in users_info:
                    users_info[entity['reviewerID']] = dict()
                    users_info[entity['reviewerID']]['review_count'] = 0
                    users_info[entity['reviewerID']]['words'] = set()
                    users_info[entity['reviewerID']]['bids'] = set()
                users_info[entity['reviewerID']]['review_count'] += 1

                # Product info
                if entity['asin'] not in products_info[genre]:
                    products_info[genre][entity['asin']] = dict()
                    products_info[genre][entity['asin']]['star'] = entity['overall']
                    products_info[genre][entity['asin']]['review_count'] = 1
                    products_info[genre][entity['asin']]['words'] = set()
                    products_info[genre][entity['asin']]['uids'] = set()
                else:
                    products_info[genre][entity['asin']]['star'] = \
                        products_info[genre][entity['asin']]['star'] * \
                        products_info[genre][entity['asin']]['review_count'] + \
                        entity['overall']
                    products_info[genre][entity['asin']]['review_count'] += 1
                    products_info[genre][entity['asin']]['star'] /= \
                        products_info[genre][entity['asin']]['review_count']

    top_tokens = dict(
        top_tokens.most_common(10) # top 10 frequentist tokens (letter only)
    )

    '''Filter out the product and user less than the require number'''
    print('Filter user and sample products')
    for uid in list(users_info):
        if users_info[entity['reviewerID']]['review_count'] < 10:
            del users_info[uid]

    max_count = dict()
    products_probs = dict()
    for genre in products_info:
        max_count[genre] = 0
        products_probs[genre] = []
        for bid in list(products_info[genre]):
            if products_info[genre][bid]['review_count'] < 20:
                del products_info[genre][bid]
                continue

            products_probs[genre].append(products_info[genre][bid]['review_count'])
            if products_info[genre][bid]['review_count'] > max_count[genre]:
                max_count[genre] = products_info[genre][bid]['review_count']

    # calculate sampling probability and popularity
    sample_size = float('inf')
    for genre in products_info:
        sum_count = sum(products_probs[genre])
        products_probs[genre] = [item/sum_count for item in products_probs[genre]]

        for bid in products_info[genre]:
            products_info[genre][bid]['popularity'] = rank_bid(
                products_info[genre][bid]['review_count'], 
                products_info[genre][bid]['star'], 
                max_count[genre], 5.0
            )
        if len(products_info[genre]) < sample_size:
            sample_size = len(products_info[genre])
    if sample_size > 3000:
        sample_size = 3000    

    # sample data
    for genre in products_info:
        np.random.seed(33) # for reproduction
        products_keys = np.random.choice(
            list(products_info[genre].keys()), 
            size=sample_size, replace=False,
            p=products_probs[genre]
        )
        products_info[genre] = dict(
            [(key, products_info[genre][key]) for key in products_keys])

    '''Build the review'''
    print('Building review data...')
    # review file
    rfile = open(save_dir+'amazon.tsv', 'w')
    columns = ['rid', 'bid', 'uid', 'text', 'date', 'genre', 'label']
    rfile.write('\t'.join(columns)+'\n')
    
    for genre in products_info:
        with open(data_dir+genre+'.json') as dfile:
            for line in dfile:
                entity = json.loads(line)

                '''Filters'''
                # user control
                if entity['reviewerID'] not in users_info:
                    continue
                
                # filter out the categories
                if entity['asin'] not in products_info[genre]:
                    continue

                # filter out text less than 10 tokens
                entity['reviewText'] = preprocess(
                    entity['reviewText'], min_len=10)
                if not entity['reviewText']:
                    continue

                '''Data collection'''
                # encode labels
                if entity['overall'] > 3:
                    entity['overall'] = 2
                elif entity['overall'] < 3:
                    entity['overall'] = 0
                else:
                    entity['overall'] = 1

                # collect review data
                line = '\t'.join([
                    entity['reviewerID']+'#'+str(entity['unixReviewTime']), entity['asin'],
                    entity['reviewerID'], entity['reviewText'], format_time(entity['reviewTime']), 
                    genre, str(entity['overall'])
                ])
                rfile.write(line + '\n')

                # collect words for both products and users
                for token in entity['reviewText'].split():
                    if not token.isalpha():
                        continue

                    if token in top_tokens:
                        continue

                    products_info[genre][entity['asin']]['words'].add(token)
                    users_info[entity['reviewerID']]['words'].add(token)

                # collect purchasing behaviors
                products_info[genre][entity['asin']]['uids'].add(entity['reviewerID'])
                users_info[entity['reviewerID']]['bids'].add(entity['asin'])

    rfile.flush()
    rfile.close()

    '''save user and product information'''
    print('Saving user information...')
    user_idx = list() 
    product_idx = list()
    with open(save_dir + 'users.json', 'w') as wfile:
        for uid in users_info:
            if len(users_info[uid]['words']) == 0:
                continue

            users_info[uid]['uid'] = uid
            users_info[uid]['words'] = list(users_info[uid]['words'])
            users_info[uid]['bids'] = list(users_info[uid]['bids'])

            wfile.write(json.dumps(users_info[uid]) + '\n')
            heapq.heappush(user_idx, (users_info[uid]['review_count'], uid))
    user_idx_encoder = dict() # a dictionary for user idx mapping 
    init_idx = len(user_idx) # 0 is the reserved idx for unknown
    while init_idx > 0:
        item = heapq.heappop(user_idx)
        user_idx_encoder[item[1]] = init_idx
        init_idx -= 1
    with open(save_dir + 'user_idx.json', 'w') as wfile:
        wfile.write(json.dumps(user_idx_encoder))

    print('Saving product information...')
    with open(save_dir + 'products.json', 'w') as wfile:
        for genre in products_info:
            for bid in products_info[genre]:
                if len(products_info[genre][bid]['words']) == 0:
                    continue

                products_info[genre][bid]['bid'] = bid
                products_info[genre][bid]['genre'] = genre
                products_info[genre][bid]['words'] = list(products_info[genre][bid]['words'])
                products_info[genre][bid]['uids'] = list(products_info[genre][bid]['uids'])
                wfile.write(json.dumps(products_info[genre][bid]) + '\n')
                heapq.heappush(product_idx, (products_info[genre][bid]['popularity'], bid))
    product_idx_encoder = dict() # a dictionary for product idx mapping 
    init_idx = len(product_idx) # 0 is the reserved idx for unknown
    while init_idx > 0:
        item = heapq.heappop(product_idx)
        product_idx_encoder[item[1]] = init_idx
        init_idx -= 1
    with open(save_dir + 'product_idx.json', 'w') as wfile:
        wfile.write(json.dumps(product_idx_encoder))


def extract_imdb(data_dir, save_dir='./raw/imdb/'):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('Collecting product information in each genre...')
    # get the top categories of movies
    products_info = dict()
    users_info = dict()
    top_genres = Counter()
    max_votes = dict() # to normalize max votes
    with open(data_dir+'movies.tsv') as dfile:
        genre_idx = dfile.readline().strip().split('\t').index('genres')
        for line in dfile:
            line = line.strip().split('\t')
            if line[0] not in products_info:
                products_info[line[0]] = dict()
            products_info[line[0]]['star'] = float(line[-2])
            votes = int(line[-1])
            products_info[line[0]]['review_count'] = votes
            products_info[line[0]]['words'] = set()
            products_info[line[0]]['uids'] = set()
            products_info[line[0]]['genre'] = [
                item.strip() for item in line[genre_idx].split(',')]

            for genre in products_info[line[0]]['genre']:
                top_genres[genre] += 1

                if genre not in max_votes:
                    max_votes[genre] = 0
                if votes > max_votes[genre]:
                    max_votes[genre] = votes

    print('Top movie categories: ', top_genres.most_common(4))
    top_genres = dict([(item[0], 0) for item in top_genres.most_common(4)])

    print('Sample movies in top categories...')
    # remove the movies that are not in top categories
    for mid in list(products_info):
        products_info[mid]['genre'] = [
            item for item in products_info[mid]['genre'] if item in top_genres]

        if len(products_info[mid]['genre']) == 0:
            del products_info[mid]
            continue

        # normalize the review_count and calculate popularity
        max_val = max([max_votes[item] for item in products_info[mid]['genre']])
        products_info[mid]['popularity'] = rank_bid(
            products_info[mid]['review_count'], products_info[mid]['star'], 
            max_val, 10.0)

        for genre in products_info[mid]['genre']:
            top_genres[genre] += 1

    # sample movies in each genre.
    # decide not to sample, because of two issues
    # multilabels, some of movies have overlapping genres
    # the current labels are quite low.

    print('Collect data now...')
    files = os.listdir(data_dir + 'reviews/')
    top_tokens = Counter()

    # review file
    rfile = open(save_dir+'imdb.tsv', 'w')
    columns = ['rid', 'bid', 'uid', 'text', 'date', 'genre', 'label']
    rfile.write('\t'.join(columns)+'\n')

    for filep in files:
        mid = filep.split('.')[0]

        # only keep movies in the selected categories
        if mid not in products_info:
            continue

        with open(data_dir + 'reviews/'+filep) as dfile:
            for line in dfile:
                entity = json.loads(line)

                # filter out empty content
                if len(entity['content']) < 3:
                    continue

                if entity['rating'] == 'x':
                    continue

                entity['content'] = preprocess(entity['content'], min_len=10)
                if not entity['content']:
                    continue

               # encode rating
                entity['rating'] = int(entity['rating'])
                if entity['rating'] < 5:
                    entity['rating'] = 0
                elif entity['rating'] > 6:
                    entity['rating'] = 2
                else:
                    entity['rating'] = 1

                # save review data
                line = [
                    entity['rid'], entity['mid'], entity['uid'],
                    entity['content'], format_time(entity['date']),
                    ','.join(products_info[mid]['genre']), str(entity['rating'])
                ]
                line = '\t'.join(line)+'\n'
                rfile.write(line)
                
                # save user and product information
                if entity['uid'] not in users_info:
                    users_info[entity['uid']] = dict()
                    users_info[entity['uid']]['review_count'] = 0
                    users_info[entity['uid']]['words'] = set()
                    users_info[entity['uid']]['bids'] = set()
                users_info[entity['uid']]['review_count'] += 1
                users_info[entity['uid']]['bids'].add(mid)
                products_info[mid]['uids'].add(entity['uid'])

                for token in entity['content'].split():
                    if token.isalpha():
                        top_tokens[token] += 1
                        users_info[entity['uid']]['words'].add(token)
                        products_info[mid]['words'].add(token)

    print('Filter out top 10 tokens...')
    top_tokens = dict(
        top_tokens.most_common(10) # top 10 frequentist tokens (letter only)
    )
    print(top_tokens)
    # filter out top tokens from the user and product information
    for uid in users_info:
        users_info[uid]['words'] = [
            word for word in users_info[uid]['words'] if word not in top_tokens]
    for mid in products_info:
        products_info[mid]['words'] = [
            word for word in products_info[mid]['words'] if word not in top_tokens]

    '''save user and product information'''
    print('Saving user information...')
    user_idx = list() 
    product_idx = list()
    with open(save_dir + 'users.json', 'w') as wfile:
        for uid in users_info:
            if len(users_info[uid]['words']) == 0:
                continue

            users_info[uid]['uid'] = uid
            users_info[uid]['bids'] = list(users_info[uid]['bids'])
            wfile.write(json.dumps(users_info[uid]) + '\n')
            heapq.heappush(user_idx, (users_info[uid]['review_count'], uid))
    user_idx_encoder = dict() # a dictionary for user idx mapping 
    init_idx = len(user_idx) # 0 is the reserved idx for unknown
    while init_idx > 0:
        item = heapq.heappop(user_idx)
        user_idx_encoder[item[1]] = init_idx
        init_idx -= 1
    with open(save_dir + 'user_idx.json', 'w') as wfile:
        wfile.write(json.dumps(user_idx_encoder))

    print('Saving product information...')
    with open(save_dir + 'products.json', 'w') as wfile:
        for bid in products_info:
            if len(products_info[bid]['words']) == 0:
                continue

            products_info[bid]['bid'] = bid
            products_info[bid]['uids'] = list(products_info[bid]['uids'])
            wfile.write(json.dumps(products_info[bid]) + '\n')
            heapq.heappush(product_idx, (products_info[bid]['popularity'], bid))
    product_idx_encoder = dict() # a dictionary for product idx mapping 
    init_idx = len(product_idx) # 0 is the reserved idx for unknown
    while init_idx > 0:
        item = heapq.heappop(product_idx)
        product_idx_encoder[item[1]] = init_idx
        init_idx -= 1
    with open(save_dir + 'product_idx.json', 'w') as wfile:
        wfile.write(json.dumps(product_idx_encoder))


def multi_genre_encode(genres, genre_encoder):
    '''Encode products that have multiple labels
    '''
    encode_genres = ['0'] * len(genre_encoder)
    genres = genres.split(',')
    for genre in genres:
        encode_genres[int(genre_encoder[genre])] = '1'
    
    return ','.join(encode_genres)


def data_splits(corpus, odir, train_ratio=.8, valid_rati=.1, test_ratio=.1):
    print('Split datasets...')
    train_ratio = .8
    valid_ratio = .1
    test_ratio = .1

    train_corpus, tmp_corpus = train_test_split(
        corpus, train_size=train_ratio, 
        test_size=1-train_ratio, random_state=33,
    )
    train_corpus.to_csv(odir+'train.tsv', sep='\t', index=None)
    
    valid_corpus, test_corpus = train_test_split(
        tmp_corpus, train_size=valid_ratio/(valid_ratio+test_ratio),
        test_size=test_ratio/(valid_ratio+test_ratio),
        random_state=33,
    )
    valid_corpus.to_csv(odir+'valid.tsv', sep='\t', index=None)
    test_corpus.to_csv(odir+'test.tsv', sep='\t', index=None)


def encode_data(data_dir, dname, multi_genre=False, save_dir='./encode/'):
    '''Encode 

        dname: the name of data source
    '''
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = save_dir + dname + '/' # create a data dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    '''build tokenizer'''
    print('Build tokenizer...')
    tok = Tokenizer(num_words=20001) # 20000 known + 1 unkown tokens
    # fit on the corpus
    corpus = pd.read_csv(data_dir+dname+'.tsv', sep='\t')
    corpus = corpus.sample(frac=1) # shuffle
    tok.fit_on_texts(corpus.text)

    # save the tokenizer
    with open(save_dir + dname + '.tkn', 'wb') as wfile:
        pickle.dump(tok, wfile)

    '''genre encoder'''
    print('Encode genre in corpus...')
    genre_encoder = dict()
    genres = set()
    for genre in corpus.genre.unique():
        genre = genre.split(',')
        genres.update(genre)
    for idx, genre in enumerate(genres):
        genre_encoder[genre] = str(idx)

    with open(save_dir + 'genre_encoder.json', 'w') as wfile:
        wfile.write(json.dumps(genre_encoder))

    # encode the genre
    if multi_genre:
        corpus.genre = corpus.genre.apply(lambda x: multi_genre_encode(x, genre_encoder))
    else:
        corpus.genre = corpus.genre.apply(lambda x: genre_encoder[x])
        
    '''encode user and product information'''
    user_idx = json.load(open(data_dir+'user_idx.json'))
    product_idx = json.load(open(data_dir+'product_idx.json'))

    # Encode user and product ids in the corpus
    print('Encode user and product ids in corpus')
    corpus.bid = corpus.bid.apply(lambda x: product_idx.get(x, 0))
    corpus.uid = corpus.uid.apply(lambda x: user_idx.get(x, 0))

    # save the split raw corpus
    data_splits(corpus, data_dir)

    '''encode and pad documents'''
    print('Encode and pad documents...')
    median_len = []
    for doc in corpus.text:
        median_len.append(len(doc.split()))
    median_len = int(np.median(median_len)) + 5
    corpus.text = corpus.text.apply(lambda x: ' '.join(map(str,
        pad_sequences(tok.texts_to_sequences([x]), maxlen=median_len)[0]
    )))
    # record the max length for future usage
    with open(save_dir+'params.json', 'w') as wfile:
        wfile.write(json.dumps({'max_len': median_len}))

    # save the corpus
    corpus.to_csv(save_dir+dname+'.tsv', sep='\t', index=None)

    print('Encode users...')
    with open(save_dir+'users.json', 'w') as wfile:
        with open(data_dir+'users.json') as dfile:
            for line in dfile:
                entity = json.loads(line)
                # encode words
                entity['words'] = tok.texts_to_sequences([entity['words']])[0]
                # encode product ids
                entity['bids'] = [product_idx.get(item, 0) for item in entity['bids']]
                # encode the uid
                entity['uid_encode'] = user_idx[entity['uid']]
                wfile.write(json.dumps(entity)+'\n')

    print('Encode products...')
    with open(save_dir+'products.json', 'w') as wfile:
        with open(data_dir+'products.json') as dfile:
            for line in dfile:
                entity = json.loads(line)
                # encode words
                entity['words'] = tok.texts_to_sequences([entity['words']])[0]
                # encode user ids
                entity['uids_encode'] = [user_idx.get(item, 0) for item in entity['uids']]
                # encode genre
                if multi_genre:
                    genre_encode = [0] * len(genre_encoder)
                    for genre in entity['genre']:
                        genre_encode[int(genre_encoder[genre])] = 1
                    entity['genre'] = genre_encode
                else:
                    entity['genre'] = genre_encoder[entity['genre']]
                # encode the uid
                entity['bid_encode'] = product_idx[entity['bid']]
                wfile.write(json.dumps(entity)+'\n')
    # save the split raw corpus
    data_splits(corpus, save_dir)


if __name__ == '__main__':
#    extract_yelp('/home/xiaolei/Documents/dataset/yelp/', save_dir='./raw/yelp/')
    encode_data('./raw/yelp/', 'yelp', save_dir='./encode/')
#    extract_amazon('/home/xiaolei/Documents/dataset/amazon/', save_dir='./raw/amazon/')
    encode_data('./raw/amazon/', 'amazon', save_dir='./encode/')
#    extract_imdb('/home/xiaolei/Documents/dataset/imdb/crawl/', save_dir='./raw/imdb/')
    encode_data('./raw/imdb/', 'imdb', multi_genre=True, save_dir='./encode/')


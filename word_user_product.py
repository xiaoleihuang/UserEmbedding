import json
import os
import pickle
import sys

import keras
from keras.preprocessing.sequence import make_sampling_table
from keras.preprocessing.sequence import skipgrams

import numpy as np
import pandas as pd

import utils

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # for Tensorflow cpu usage
os.environ["CUDA_VISIBLE_DEVICES"] = ""


# design model
def build_model(params=None):
    '''
        params (dict): a dictionary of parameter settings
    '''
    if not params: # default params
        params = {
            'window': 5,
            'vocab_size': 20000,
            'user_size': 8000,
            'product_size': 5000,
            'emb_dim': 300,
            'word_emb_path': './resources/word_emb.npy',
            'user_emb_path': './resources/user_emb.npy',
            'product_emb_path': './resources/product_emb.npy',
            'word_emb_train': True,
            'user_emb_train': True,
            'product_emb_train': True,
            'user_task_weight': 1,
            'word_task_weight': 1,
            'product_task_weight': 1,
            'epochs': 5,
            'optimizer': 'adam',
            'lr': 0.00001,
        }
        
    # Word Word Input
    word_input = keras.layers.Input((1,), name='word_input')
    word_context_input = keras.layers.Input((1,), name='word_context_input')
    
    # User Word Input
    user_input = keras.layers.Input((1,), name='user_input')
    user_word_input = keras.layers.Input((1,), name='user_word_input')
    
    # Product User Input, users who purchased the product
    product_input = keras.layers.Input((1,), name='product_input')
    product_user_input = keras.layers.Input((1,), name='product_user_input')
    
    # Product Word Input, product description
    product_word_input = keras.layers.Input((1,), name='product_word_input')
    product_context_input = keras.layers.Input((1,), name='product_context_input')
    
    # load weights if word embedding path is given
    if os.path.exists(params['word_emb_path']):
        word_emb = keras.layers.Embedding(
            params['vocab_size'], params['emb_dim'],
            weights=[np.load(open(params['word_emb_path']))],
            trainable=params['word_emb_train'], name='word_emb'
        )
    else:
        word_emb = keras.layers.Embedding(
            params['vocab_size'], params['emb_dim'],
            trainable=params['word_emb_train'], name='word_emb'
        )
    
    # load weights if user embedding path is given
    if os.path.exists(params['user_emb_path']):
        user_emb = keras.layers.Embedding(
            params['user_size'], params['emb_dim'],
            weights=[np.load(open(params['user_emb_path']))],
            trainable=params['user_emb_train'], name='user_emb'
        )
    else:
        user_emb = keras.layers.Embedding(
            params['user_size'], params['emb_dim'],
            trainable=params['user_emb_train'], name='user_emb'
        )
        
    # load weights if product embedding path is given
    if os.path.exists(params['product_emb_path']):
        product_emb = keras.layers.Embedding(
            params['product_size'], params['emb_dim'],
            weights=[np.load(open(params['product_emb_path']))],
            trainable=params['product_emb_train'], name='product_emb'
        )
    else:
        product_emb = keras.layers.Embedding(
            params['product_size'], params['emb_dim'],
            trainable=params['product_emb_train'], name='product_emb'
        )
    
    '''Word Word Dot Production'''
    word_target = word_emb(word_input)
    word_context = word_emb(word_context_input)

    word_dot = keras.layers.dot(
        [word_target, word_context], axes=-1)
    word_dot = keras.layers.Reshape((1,))(word_dot)
    word_pred = keras.layers.Dense(
        1, activation='sigmoid', name='word_pred'
    )(word_dot)
    
    '''User Word Dot Production'''
    user_target = user_emb(user_input)
    user_word = word_emb(user_word_input)
    
    user_word_dot = keras.layers.dot(
        [user_target, user_word], axes=-1
    )
    user_word_dot = keras.layers.Reshape((1,))(user_word_dot)
    user_pred = keras.layers.Dense(
        1, activation='sigmoid', name='user_pred'
    )(user_word_dot)
    
    '''Product User Dot Production'''
    product_target = product_emb(product_input)
    product_user = user_emb(product_user_input)
    
    product_user_dot = keras.layers.dot(
        [product_target, product_user], axes=-1
    )
    product_user_dot = keras.layers.Reshape((1,))(product_user_dot)
    product_pred = keras.layers.Dense(
        1, activation='sigmoid', name='user_pred'
    )(product_user_dot)
    
    '''Product Word Dot Production'''
    product_word = product_emb(product_word_input)
    product_context = word_emb(product_context_input)
    product_word_dot = keras.layers.dot(
        [product_word, product_context], axes=-1
    )
    product_word_dot = keras.layers.Reshape((1,))(product_word_dot)
    product_word_pred = keras.layers.Dense(
        1, activation='sigmoid', name='user_pred'
    )(product_word_dot)
    
    '''Compose model'''
    if params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(lr=params['lr'])
    else:
        optimizer = keras.optimizers.SGD(
            lr=params['lr'], decay=1e-6, momentum=0.9, nesterov=True)
    
    # word model
    ww_model = keras.models.Model(
        inputs=[word_input, word_context_input],
        outputs=word_pred
    )
    ww_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # user word model
    uw_model = keras.models.Model(
        inputs=[user_input, user_word_input],
        outputs=user_pred
    )
    uw_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # product word model
    pw_model = keras.models.Model(
        inputs=[product_word_input, product_context_input],
        outputs=product_word_pred,
    )
    pw_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # product user model
    pu_model = keras.models.Model(
        inputs=[product_input, product_user_input],
        outputs=product_pred,
    )
    pu_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return ww_model, uw_model, pu_model, pw_model


def main(dname, encode_dir, raw_dir, odir='./resources/skipgrams/'):
    # load corpus data
    raw_corpus = pd.read_csv(raw_dir+dname+'.tsv', sep='\t')

    # load user data
    user_idx = json.load(open(raw_dir+'user_idx.json'))
    user_info = dict()
    with open(encode_dir+'users.json') as dfile:
        for line in dfile:
            line = json.loads(line)
            user_info[line['uid']] = line


    # load product data
    product_idx = json.load(open(raw_dir+'product_idx.json'))
    product_info = dict()
    with open(encode_dir+'products.json') as dfile:
        for line in dfile:
            line = json.loads(line)
            product_info[line['bid']] = line


    # load tokenizer
    tok = pickle.load(open(encode_dir+dname+'.tkn', 'rb'))
    params = {
        'window': 5,
        'vocab_size': tok.num_words,
        'user_size': len(user_info)+1, # +1 for unknown
        'product_size': len(product_info)+1,  # +1 for unknown
        'emb_dim': 300,
        'word_emb_path': './resources/word_emb.npy',
        'user_emb_path': './resources/user_emb.npy',
        'product_emb_path': './resources/product_emb.npy',
        'word_emb_train': True,
        'user_emb_train': True,
        'product_emb_train': True,
        'user_task_weight': 1,
        'word_task_weight': 1,
        'product_task_weight': 1,
        'product_user_task_weight': 1,
        'epochs': 5,
        'optimizer': 'adam',
        'lr': 1e-5,
    }
    word_sampling_table = make_sampling_table(size=params['vocab_size'])
    ww_model, uw_model, pu_model, pw_model = build_model(params)
    print()
    print(params)

    for epoch in range(params['epochs']):
        loss = 0
        # shuffle the data
        raw_corpus = raw_corpus.sample(frac=1).reset_index(drop=True)
        for step, entry in raw_corpus.iterrows():
            '''word info, ww: word-word'''
            encode_doc = tok.texts_to_sequences([entry.text])
            ww_pairs, ww_labels = skipgrams(
                sequence=encode_doc[0], vocabulary_size=params['vocab_size'],
                window_size=params['window']
            )
            
            word_pairs = [np.array(x) for x in zip(*ww_pairs)]
            ww_labels = np.array(ww_labels, dtype=np.int32)
            
            '''user info, uw: user-word'''
            cur_user = user_info[entry.uid]
            uw_pairs, uw_labels = utils.user_word_sampler(
                cur_user['uid_encode'], cur_user['words'],
                params['vocab_size'], negative_samples=1
            )
            uw_pairs = [np.array(x) for x in zip(*uw_pairs)]
            uw_labels = np.array(uw_labels, dtype=np.int32)
            
            '''product info, pu: product-user'''
            cur_prod = product_info[entry.bid]
            pu_pairs, pu_labels = utils.user_word_sampler(
                cur_prod['bid_encode'], cur_prod['uids_encode'],
                params['user_size'], negative_samples=1
            )
            pu_pairs = [np.array(x) for x in zip(*pu_pairs)]
            pu_labels = np.array(pu_labels, dtype=np.int32)
            
            '''product info, pw: product-word'''
            pw_pairs, pw_labels = utils.user_word_sampler(
                cur_prod['bid_encode'], cur_prod['words'],
                params['vocab_size'], negative_samples=1
            )
            pw_pairs = [np.array(x) for x in zip(*pw_pairs)]
            pw_labels = np.array(pw_labels, dtype=np.int32)
            
            if word_pairs and uw_pairs and pu_pairs and pw_pairs:
                loss += ww_model.train_on_batch(word_pairs, ww_labels)
                loss += uw_model.train_on_batch(uw_pairs, uw_labels)
                loss += pu_model.train_on_batch(pu_pairs, pu_labels)
                loss += pw_model.train_on_batch(pw_pairs, pw_labels)
                
                loss_avg = loss / step
            if step % 100 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')


    if not os.path.exists(odir):
        os.mkdir(odir)
    # save the model
    emb_dir = odir + dname + '/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    emb_dir = emb_dir + 'word_user_product/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)

    # save the model
    ww_model.save(emb_dir+'ww_model.h5')
    uw_model.save(emb_dir+'uw_model.h5')
    pu_model.save(emb_dir+'pu_model.h5')
    pw_model.save(emb_dir+'pw_model.h5')
    # save the word embedding
    np.save(emb_dir+'word.npy', ww_model.get_layer(name='word_emb').get_weights()[0])
    # save the user embedding
    np.save(emb_dir+'user.npy', uw_model.get_layer(name='user_emb').get_weights()[0])
    # save the product embedding
    np.save(emb_dir+'product.npy', pu_model.get_layer(name='product_emb').get_weights()[0])


if __name__ == '__main__':
    encode_dir = './data/encode/'
    raw_dir = './data/raw/'
    odir = './resources/skipgrams/'

#    for dname in ['amazon', 'yelp', 'imdb']:
#        raw_dir = raw_dir + dname + '/'
#        encode_dir = encode_dir + dname + '/'
#        main(dname, encode_dir, raw_dir, odir=odir)
    dname = sys.argv[1]
    raw_dir = raw_dir + dname + '/'
    encode_dir = encode_dir + dname + '/'
    main(dname, encode_dir, raw_dir, odir=odir)

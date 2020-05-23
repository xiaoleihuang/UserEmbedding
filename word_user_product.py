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


def main(dname, encode_dir, raw_dir, odir='./resources/skipgrams/', mode='local'):
    # load corpus data
    raw_corpus = pd.read_csv(raw_dir+dname+'.tsv', sep='\t')

    # load user data
    user_idx = json.load(open(raw_dir+'user_idx.json'))
    user_info = dict()
    user_control = set() # control if renew user_info sample method
    with open(encode_dir+'users.json') as dfile:
        for line in dfile:
            line = json.loads(line)
            user_info[line['uid']] = line
            user_info[line['uid']]['count'] = 0

    # load product data
    product_idx = json.load(open(raw_dir+'product_idx.json'))
    product_info = dict()
    product_control = set() # control if renew product_info sample method
    with open(encode_dir+'products.json') as dfile:
        for line in dfile:
            line = json.loads(line)
            product_info[line['bid']] = line
            product_info[line['bid']]['count'] = 0

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
            
            '''user info, uw: user-word; product info, pu: product-user'''
            cur_user = user_info[entry.uid]
            cur_prod = product_info[entry.bid]

            if mode == 'local':
                # user
                uw_pairs, uw_labels = utils.user_word_sampler(
                    cur_user['uid_encode'], encode_doc[0], 
                    params['vocab_size'], set(cur_user['words']), 
                    negative_samples=1
                )
                uw_pairs = [np.array(x) for x in zip(*uw_pairs)]
                uw_labels = np.array(uw_labels, dtype=np.int32)

                # product
                pu_pairs, pu_labels = utils.user_word_sampler(
                    cur_prod['bid_encode'], [user_idx[entry.uid]],
                    params['user_size'], set(cur_prod['uids_encode']),
                    negative_samples=1
                )
                pu_pairs = [np.array(x) for x in zip(*pu_pairs)]
                pu_labels = np.array(pu_labels, dtype=np.int32)

                # user-product
                pw_pairs, pw_labels = utils.user_word_sampler(
                    cur_prod['bid_encode'], encode_doc[0],
                    params['vocab_size'], set(cur_prod['words']),
                    negative_samples=1
                )
                pw_pairs = [np.array(x) for x in zip(*pw_pairs)]
                pw_labels = np.array(pw_labels, dtype=np.int32)
            elif mode == 'decay':
                decay_num = utils.sample_decay(cur_user['count'])
                if decay_num > np.random.random():
                    uw_pairs, uw_labels = utils.user_word_sampler(
                        cur_user['uid_encode'], cur_user['words'],
                        params['vocab_size'], negative_samples=1
                    )
                    uw_pairs = [np.array(x) for x in zip(*uw_pairs)]
                    uw_labels = np.array(uw_labels, dtype=np.int32)

                    user_info[entry.uid]['count'] += 1
                    user_control.add(entry.uid)
                else:
                    uw_pairs = None
                    uw_labels = None

                if len(user_control) >= len(user_info) - 1:
                    # restart the control for sampling
                    for uid in user_info:
                        user_info[uid]['count'] = 0
                    user_control.clear()


                decay_num = utils.sample_decay(cur_prod['count'])
                if decay_num > np.random.random():
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
                    
                    product_info[entry.bid]['count'] += 1
                    product_control.add(entry.bid)
                else:
                    pw_pairs = None
                    pu_pairs = None
                    pw_labels = None
                    pu_labels = None

                if len(product_control) >= len(product_info) - 10:
                    # restart the control for sampling
                    for bid in product_info:
                        product_info[bid]['count'] = 0
                    product_control.clear()
            elif mode == 'global':
                # user
                uw_pairs, uw_labels = utils.user_word_sampler(
                    cur_user['uid_encode'], set(cur_user['words']),
                    params['vocab_size'], None,
                    negative_samples=1
                )
                uw_pairs = [np.array(x) for x in zip(*uw_pairs)]
                uw_labels = np.array(uw_labels, dtype=np.int32)

                # product
                pu_pairs, pu_labels = utils.user_word_sampler(
                    cur_prod['bid_encode'], [user_idx[entry.uid]],
                    params['user_size'], set(cur_prod['uids_encode']),
                    negative_samples=1
                )
                pu_pairs = [np.array(x) for x in zip(*pu_pairs)]
                pu_labels = np.array(pu_labels, dtype=np.int32)

                # user-product
                pw_pairs, pw_labels = utils.user_word_sampler(
                    cur_prod['bid_encode'], set(cur_prod['words']),
                    params['vocab_size'], None,
                    negative_samples=1
                )
                pw_pairs = [np.array(x) for x in zip(*pw_pairs)]
                pw_labels = np.array(pw_labels, dtype=np.int32)
            
            if word_pairs:
                loss += ww_model.train_on_batch(word_pairs, ww_labels)
            if uw_pairs:
                loss += uw_model.train_on_batch(uw_pairs, uw_labels)
            if pu_pairs:
                loss += pu_model.train_on_batch(pu_pairs, pu_labels)
            if pw_pairs:
                loss += pw_model.train_on_batch(pw_pairs, pw_labels)

            loss_avg = loss / step
            if step % 100 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')


    # save the model
    ww_model.save(odir+'ww_model.h5')
    uw_model.save(odir+'uw_model.h5')
    pu_model.save(odir+'pu_model.h5')
    pw_model.save(odir+'pw_model.h5')
    # save the word embedding
    np.save(odir+'word.npy', ww_model.get_layer(name='word_emb').get_weights()[0])
    # save the user embedding
    np.save(odir+'user.npy', uw_model.get_layer(name='user_emb').get_weights()[0])
    # save the product embedding
    np.save(odir+'product.npy', pu_model.get_layer(name='product_emb').get_weights()[0])


if __name__ == '__main__':
    dname = sys.argv[1]
    # three sampling strategies: local, decay, global
    if len(sys.argv) > 1:
        mode = sys.argv[2]
    else:
        mode = 'local'

    encode_dir = './data/encode/'
    raw_dir = './data/raw/'
    raw_dir = raw_dir + dname + '/'
    encode_dir = encode_dir + dname + '/'

    odir = './resources/skipgrams/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    odir = odir + dname + '/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    odir = odir + 'word_user_product_{}/'.format(mode)
    if not os.path.exists(odir):
        os.mkdir(odir)

    main(dname, encode_dir, raw_dir, odir=odir, mode=mode)

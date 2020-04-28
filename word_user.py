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
            'emb_dim': 300,
            'word_emb_path': './resources/word_emb.npy',
            'user_emb_path': './resources/user_emb.npy',
            'word_emb_train': True,
            'user_emb_train': True,
            'user_task_weight': 1,
            'word_task_weight': 1,
            'epochs': 5,
            'optimizer': 'adam',
            'lr': 0.00001,
        }
        
    # Word Input
    word_target_input = keras.layers.Input((1,), name='word_target')
    word_context_input = keras.layers.Input((1,), name='word_context')
    
    # User Input
    user_word_input = keras.layers.Input((1,), name='user_word_input')
    user_context_input = keras.layers.Input((1,), name='user_context_input')
    
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
    
    '''Word Dot Production'''
    word_target = word_emb(word_target_input)
    word_context = word_emb(word_context_input)

    word_dot = keras.layers.dot(
        [word_target, word_context], axes=-1)
    word_dot = keras.layers.Reshape((1,))(word_dot)
    word_pred = keras.layers.Dense(
        1, activation='sigmoid', name='word_pred'
    )(word_dot)
    
    '''User Word Dot Production'''
    user_word = user_emb(user_word_input)
    user_context = word_emb(user_context_input)
    
    user_word_dot = keras.layers.dot(
        [user_word, user_context], axes=-1
    )
    user_word_dot = keras.layers.Reshape((1,))(user_word_dot)
    user_pred = keras.layers.Dense(
        1, activation='sigmoid', name='user_pred'
    )(user_word_dot)
    
    '''Compose model'''
    if params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(lr=params['lr'])
    else:
        optimizer = keras.optimizers.SGD(
            lr=params['lr'], decay=1e-6, momentum=0.9, nesterov=True)

    # word model
    ww_model = keras.models.Model(
        inputs=[word_target_input, word_context_input],
        outputs=word_pred
    )
    ww_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    # user word model
    uw_model = keras.models.Model(
        inputs=[user_word_input, user_context_input],
        outputs=user_pred
    )
    uw_model.compile(loss='binary_crossentropy', optimizer=optimizer)
    
    return ww_model, uw_model


def main(dname, encode_dir, raw_dir, odir='./resources/skipgrams/'):
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

    # load tokenizer
    tok = pickle.load(open(encode_dir+dname+'.tkn', 'rb'))
    params = {
        'window': 5,
        'vocab_size': tok.num_words,
        'user_size': len(user_info)+1, # +1 for unknown
        'emb_dim': 300,
        'word_emb_path': './resources/word_emb.npy',
        'user_emb_path': './resources/user_emb.npy',
        'word_emb_train': True,
        'user_emb_train': True,
        'user_task_weight': 1,
        'word_task_weight': 1,
        'epochs': 5,
        'optimizer': 'adam',
        'lr': 1e-5,
    }
    word_sampling_table = make_sampling_table(size=params['vocab_size'])

    ww_model, uw_model = build_model(params)
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
            decay_num = utils.sample_decay(cur_user['count'])

            if decay_num > np.ramdom.random():
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
            
            if word_pairs:
                loss += ww_model.train_on_batch(word_pairs, ww_labels)
            if uw_pairs:
                loss += uw_model.train_on_batch(uw_pairs, uw_labels)

            loss_avg = loss / step
            if step % 100 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')

    emb_dir = './resources/skipgrams/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    # save the model
    emb_dir = emb_dir + dname + '/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    emb_dir = emb_dir + 'word_user/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)


    # save the model
    ww_model.save(emb_dir+'ww_model.h5')
    uw_model.save(emb_dir+'uw_model.h5')
    # save the word embedding
    np.save(emb_dir+'word.npy', ww_model.get_layer(name='word_emb').get_weights()[0])
    # save the user embedding
    np.save(emb_dir+'user.npy', uw_model.get_layer(name='user_emb').get_weights()[0])

if __name__ == '__main__':
    encode_dir = './data/encode/'
    raw_dir = './data/raw/'
    odir='./resources/skipgrams/'

#    for dname in ['amazon', 'yelp', 'imdb']:
#        raw_dir = raw_dir + dname + '/'
#        encode_dir = encode_dir + dname + '/'
#        main(dname, encode_dir, raw_dir, odir=odir)
    dname = sys.argv[1]
    raw_dir = raw_dir + dname + '/'
    encode_dir = encode_dir + dname + '/'
    main(dname, encode_dir, raw_dir, odir=odir)

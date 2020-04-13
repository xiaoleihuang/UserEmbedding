import json
import os
os.environ["CUDA_VISIBLE_DEVICES"]="3"
import sys
import pickle

import keras
from keras.preprocessing.sequence import make_sampling_table
from keras.preprocessing.sequence import skipgrams

import numpy as np
import pandas as pd

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
            'emb_dim': 300,
            'word_emb_path': './resources/word_emb.npy',
            'epochs': 5,
            'optimizer': 'adam',
            'lr': 0.00001,
        }
    word_target_input = keras.layers.Input((1,), name='word_target')
    word_context_input = keras.layers.Input((1,), name='word_context')
    
    # load weights if word embedding path is given
    if os.path.exists(params['word_emb_path']):
        word_emb = keras.layers.Embedding(
            params['vocab_size'], params['emb_dim'],
            weights=[np.load(open(params['word_emb_path']))],
            name='word_emb'
        )
    else:
        word_emb = keras.layers.Embedding(
            params['vocab_size'], params['emb_dim'],
            name='word_emb'
        )
    
    word_target = word_emb(word_target_input)
#     word_target = keras.layers.Reshape((params['emb_dim'],1))(word_target)
    word_context = word_emb(word_context_input)
#     word_context = keras.layers.Reshape((params['emb_dim'],1))(word_context)

    word_dot = keras.layers.dot(
        [word_target, word_context], axes=-1)
    word_dot = keras.layers.Reshape((1,))(word_dot)
    word_pred = keras.layers.Dense(
        1, activation='sigmoid'
    )(word_dot)
    
    if params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(lr=params['lr'])
    
    model = keras.models.Model(
        inputs=[word_target_input, word_context_input],
        outputs=word_pred
    )    
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


def main(dname, encode_dir, raw_dir, odir='./resources/skipgrams/'):
    # load tokenizer
    tok = pickle.load(open(encode_dir+dname+'.tkn', 'rb'))
    params = {
        'window': 5,
        'vocab_size': tok.num_words,
        'emb_dim': 300,
        'word_emb_path': './resources/word_emb.npy',
        'epochs': 5,
        'optimizer': 'adam',
        'lr': 1e-5,
    }
    word_sampling_table = make_sampling_table(size=params['vocab_size'])

    # load the data
    raw_corpus = pd.read_csv(raw_dir+dname+'.tsv', sep='\t')

    # build and train model
    print(params)
    print()
    model = build_model(params)
    print(model.summary())
    for epoch in range(params['epochs']):
        loss = 0
        # shuffle the data
        raw_corpus = raw_corpus.sample(frac=1).reset_index(drop=True)
        for step, doc in enumerate(raw_corpus.text):
            encode_doc = tok.texts_to_sequences([doc])
            word_pairs, word_labels = skipgrams(
            sequence=encode_doc[0], vocabulary_size=params['vocab_size'],
            window_size=params['window'])
            
            x = [np.array(x) for x in zip(*word_pairs)]
            y = np.array(word_labels, dtype=np.int32)
            
            if word_pairs:
                loss += model.train_on_batch(x,y)
                loss_avg = loss / step
            if step % 100 == 0:
                print('Epoch: {}, Step: {}'.format(epoch, step))
                print('\tLoss: {}.'.format(loss_avg))
                print('-------------------------------------------------')

    if not os.path.exists(odir):
        os.mkdir(odir)
    emb_dir = odir + dname + '/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)
    emb_dir = emb_dir + 'word/'
    if not os.path.exists(emb_dir):
        os.mkdir(emb_dir)

    # save the model
    model.save(emb_dir+'ww_model.h5')
    # save the word embedding
    np.save(emb_dir+'word.npy', model.get_layer(name='word_emb').get_weights()[0])


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

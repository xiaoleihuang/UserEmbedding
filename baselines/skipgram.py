'''This model implements the paper of 
        Modeling Context with User Embeddings for Sarcasm Detection in Social Media

    This model trains user and word embedding jointly.
'''

import keras
import numpy as np

import utils
import config


def init_w2v_gauss(E, n_users):
    ''' adapt and revise 
            from https://github.com/samiroid/usr2vec/blob/master/code/usr2vec.py
    '''
    mu  = np.mean(E,axis=0)
    mu  = np.squeeze(np.asarray(mu))
    cov = np.cov(E,rowvar=0)
    return np.random.multivariate_normal(mu, cov,size=n_users)


def init_w2v_mean(E, n_users):
    ''' adapt and revise from 
            https://github.com/samiroid/usr2vec/blob/master/code/usr2vec.py
    '''
    rng = np.random.RandomState(1234)
    mu  = np.mean(E.T,axis=1)    
    U   = np.asarray(rng.normal(0,0.01, size=(E.shape[1],n_users)))    
    return U.T + mu[:,None].T


def data_iter(datap):
    pass

def train_model(config):
    '''define the model'''
    # input
    user_input = keras.layers.Input(shape=(None,), name='uid')
    word_input = keras.layers.Input(shape=(None,), name='wid')

    # embedding
    user_emb_wt = None
    if config['init'] and config['word_emb_path']:
        word_emb_wt = np.load(config['word_emb_path'])
        word_emb = keras.layers.Embedding(
            config['vocab_size'], config['emb_dim'], 
            weights=[word_emb_wt],
            name='word_emb'
        )(word_input)

        if config['init'] == 'gauss':
            user_emb_wt = init_w2v_gauss(word_emb_wt, config['user_size'])
        elif config['init'] == 'mean':
            user_emb_wt = init_w2v_mean(word_emb_wt, config['user_size'])
    else:
        word_emb = keras.layers.Embedding(
            config['vocab_size'], config['emb_dim'], name='word_emb'
        )(word_input)

    if not user_emb_wt:
        user_emb = keras.layers.Embedding(
            config['user_size'], config['emb_dim'], name='user_emb'
        )(user_input)
    else:
        user_emb = keras.layers.Embedding(
            config['user_size'], config['emb_dim'],
            weight=[user_emb_wt], name='user_emb'
        )(user_input)

    # dot production
    dot_prod = keras.layers.Dot(axes=-1, normalize=)([user_emb, word_emb])
    dot_prod = keras.layers.Reshape((1,))(dot_prod)

    # prediction
    predict = keras.layers.Activation('softmax', name='prediction')(dot_prod)

    # single output model
    skipgram = keras.models.Model(inputs=[user_input, word_input], outputs=predict)
    print(skipgram.summary())
    skipgram.compile(
        loss='binary_crossentropy', 
        optimizer=keras.optimizers.SGD(learning_rate=config['lr'])
    )


    '''Train the user embedding model'''
    for epoch in config['epochs']:
        train_iter = data_iter(config['train_path'])
        
        pass

    # save the model
    
    # save the word and user embeddings
   

if __name__ == '__main__':
    train_model(config.config)


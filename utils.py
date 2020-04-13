''' user word generator by negative sampling (only 1 negative sample)'''

import numpy as np
import random

def user_word_sampler(uid, sequence, vocab_size, negative_samples=1):
    '''This function was adopted from https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/sequence.py#L151

        uid (int): a user id index
        sequence (list): a sequence of word indices
        vocab_size (int): word vocabulary size
    '''
    couples = []
    labels = []
    
    for wid in sequence:
        couples.append([uid, wid])
        labels.append(1)

    if negative_samples > 0:
        num_negative_samples = int(len(labels) * negative_samples)
        words = set(sequence)

        for idx in range(num_negative_samples):
            wid = random.randint(1, vocab_size-1)
            if wid not in words: # ensure user did not use the word
                couples.append([uid, wid])
                labels.append(0)
        
    # shuffle
    seed = random.randint(0, 10e6)
    random.seed(seed)
    random.shuffle(couples)
    random.seed(seed)
    random.shuffle(labels)
    return couples, labels

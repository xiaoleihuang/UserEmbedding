'''This script is to train various embeddings
    (1) word embeddingSkip-gram mode
    (2) LDA
    (3) BERT: fine tune on the Training dataset
'''

import os
import pickle
import json
import sys
from collections import Counter

from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from gensim.models.word2vec import Word2Vec
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim.models.ldamulticore import LdaMulticore

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertConfig
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
import torch.nn.functional as F
from tqdm import tqdm, trange
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np


class RawCorpus(object):
    def __init__(self, filep, doc2id=False, dictionary=None):
        ''' Load TSV file
        '''
        self.filep = filep
        self.dictionary = dictionary
        self.doc2id = doc2id

    def __iter__(self):
        with open(self.filep) as dfile:
            columns = dfile.readline().strip().split('\t')
            text_idx = columns.index('text')
            for line in dfile:
                line = line.split('\t')[text_idx].strip().split()
                if self.doc2id and self.dictionary:
                    yield self.dictionary.doc2bow(line)
                else:
                    yield line


def train_w2v(dname, raw_dir='./data/raw/', odir='./resources/embedding/'):
    dpath = raw_dir + dname + '/' + dname + '.tsv'
    corpus = RawCorpus(dpath)
    model = Word2Vec(
        corpus, min_count=2, window=5, 
        size=300, iter=10, sg=1, workers=8,
        max_vocab_size=20000,
    )

    odir = odir + dname + '/'
    if not os.path.exists(odir):
        os.mkdir(odir)
    odir += 'w2v.txt'
    model.wv.save_word2vec_format(odir, binary=False)


def train_lda(dname, raw_dir='./data/raw/', odir='./resources/embedding/'):
    ''' 
        The number of topics should be aligned with the dimensions of the user embedding.
    '''
    odir = odir + dname + '/'
    if dname not in odir and not os.path.exists(odir):
        os.mkdir(odir)

    # load data and build dictionary
    dpath = raw_dir + dname + '/' + dname + '.tsv'
    if os.path.exists(odir + 'lda_dict.pkl'):
        dictionary = pickle.load(open(odir + 'lda_dict.pkl', 'rb'))
    else:
        corpus = RawCorpus(dpath)
        dictionary = Dictionary(corpus, prune_at=20000)
        dictionary.save(odir + 'lda_dict.pkl')

    doc_matrix = RawCorpus(dpath, True, dictionary)

    if dname == 'amazon1':
#        path_to_mallet_binary = "/export/b10/xhuang/xiaolei_data/UserEmbedding/baselines/Mallet/bin/mallet"
#        model = LdaMallet(
#            path_to_mallet_binary, corpus=doc_matrix, 
#            num_topics=300, id2word=dictionary
#        )
#        model = malletmodel2ldamodel(model)
        model = LdaModel(
            doc_matrix, id2word=dictionary, num_topics=300,
            passes=5, alpha='symmetric'
        )
    else:
        model = LdaMulticore(
            doc_matrix, id2word=dictionary, num_topics=300,
            passes=5, alpha='symmetric', workers=4
        )
    model.save(odir + 'lda.model')


class FineTuneBert:
    '''Fine tune English BERT on the training data
    '''
    def __init__(self, dname, raw_dir='./data/raw/', odir='./resources/embedding/', params=None):
        self.dname = dname
        self.raw_dir= raw_dir
        self.odir = odir

        if not os.path.exists(self.odir):
            os.mkdir(self.odir)

        if not params:
            self.params = dict()
            self.params['decay_rate'] = .001
            self.params['lr'] = 2e-5
            self.params['warm_steps'] = 100
            self.params['train_steps'] = 1000
            self.params['batch_size'] = 32
            self.params['balance'] = True
            self.params['max_len'] = 80
        else:
            self.params = params
    

    def get_free_gpu(self):
        os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
        memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
        os.remove('tmp')
        return np.argmax(memory_available)

    # Function to calculate the accuracy of our predictions vs labels
    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    def flat_f1(self, preds, labels):
        macro_score = f1_score(
            y_true=labels, y_pred=preds,
            average='macro',
        )
        weighted_score = f1_score(
            y_true=labels, y_pred=preds,
            average='weighted',
        )
        print('Weighted F1-score: ', weighted_score)
        print('Macro F1-score: ', macro_score)
        return macro_score, weighted_score


    def tune_bert(self):
        if torch.cuda.is_available():
            device = self.get_free_gpu()
            #torch.cuda.set_device(device)

            print('Tuning BERT via device: ', device)
            print('Device Name: ', torch.cuda.get_device_name(device))
            device = torch.device('cuda:{}'.format(device))
        else:
            device = torch.device('cpu')
            print('Tuning BERT via device: CPU')

        print('Loading datasets and downsample the training data')
        train_df = pd.read_csv(self.raw_dir + self.dname + '/train.tsv', sep='\t')
        valid_df = pd.read_csv(self.raw_dir + self.dname + '/valid.tsv', sep='\t')

        if self.params['balance']:
            label_count = Counter(train_df.label)
            for label_tmp in label_count:
                # downsampling
                sample_num = label_count.most_common()[-1][1]
                if sample_num - label_count[label_tmp] == 0:
                    continue

                train_df = pd.concat([train_df,
                    train_df[train_df.label==label_tmp].sample(
                        int(sample_num), replace=False
                    )])

            train_df = train_df.reset_index() # to prevent index key error

        data_df = [train_df, valid_df]

        # add special tokens at the beginning and end of each sentence
        for doc_df in data_df:
            doc_df.text = doc_df.text.apply(lambda x: '[CLS] '+ x +' [SEP]')

        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', do_lower_case=True
        )

        print('Padding Datasets...')
        for doc_df in data_df:
            doc_df.text = doc_df.text.apply(lambda x: tokenizer.tokenize(x))

        # convert to indices and pad the sequences
        for doc_df in data_df:
            doc_df.text = doc_df.text.apply(
                lambda x: pad_sequences(
                    [tokenizer.convert_tokens_to_ids(x)],
                    maxlen=self.params['max_len'], dtype="long"
                    )[0])

        # create attention masks
        for doc_df in data_df:
            attention_masks = []
            for seq in doc_df.text:
                seq_mask = [float(idx>0) for idx in seq]
                attention_masks.append(seq_mask)
            doc_df['masks'] = attention_masks


        # format train, valid
        train_inputs = torch.tensor(data_df[0].text)
        train_labels = torch.tensor(data_df[0].label)
        train_masks = torch.tensor(data_df[0].masks)
        valid_inputs = torch.tensor(data_df[1].text)
        valid_labels = torch.tensor(data_df[1].label)
        valid_masks = torch.tensor(data_df[1].masks)

        train_data = TensorDataset(
            train_inputs, train_masks, train_labels)

        if len(train_df) > 150000:
            data_size = 150000
        else:
            data_size = len(train_df)
        train_sampler = RandomSampler(train_data, replacement=True, num_samples=data_size)

        train_dataloader = DataLoader(
            train_data, sampler=train_sampler, batch_size=self.params['batch_size'])
        valid_data = TensorDataset(
            valid_inputs, valid_masks, valid_labels)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=self.params['batch_size'])

        # load the pretrained model
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=len(train_df.label.unique())
        )
        model.to(device)

        # organize parameters
        param_optimizer = list(model.named_parameters())
        no_decay = [] # , 'bert' 'bias' freeze all bert parameters
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': self.params['decay_rate']},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.params['lr'])
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.params['warm_steps'], 
            num_training_steps=self.params['train_steps']
        )

        epochs=4
        # start to fine tune
        print('Fine Tuning the model...')
        for _ in trange(epochs, desc='Epoch'):
            model.train()
            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # train batch
            for step, batch in enumerate(train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Clear out the gradients (by default they accumulate)
                optimizer.zero_grad()
                # Forward pass
                outputs = model(
                    b_input_ids, token_type_ids=None, 
                    attention_mask=b_input_mask, labels=b_labels
                )
                
                # outputs[0].backward()
                outputs[0].backward() # backward pass
                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update tracking variables
                tr_loss += outputs[0].item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

            print("Train loss: {}".format(tr_loss/nb_tr_steps))

            '''Validation'''
            best_valid_f1 = 0.0
            # Put model in evaluation mode to evaluate loss on the validation set
            model.eval()
            # tracking variables
            eval_loss, eval_accuracy = 0, 0
            nb_eval_steps, nb_eval_examples = 0, 0

            # batch eval
            y_preds = []
            for batch in valid_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    outputs = model(
                        b_input_ids, 
                        token_type_ids=None, 
                        attention_mask=b_input_mask)
                # Move logits and labels to CPU
                logits = outputs[0].detach().cpu().numpy()
                # record the prediction
                pred_flat = np.argmax(logits, axis=1).flatten()
                y_preds.extend(pred_flat)

                label_ids = b_labels.to('cpu').numpy()
                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
            
                eval_accuracy += tmp_eval_accuracy
                nb_eval_steps += 1

            print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))

            # evaluate the validation f1 score
            f1_m_valid, f1_w_valid = self.flat_f1(y_preds, valid_df.label)
            if f1_m_valid > best_valid_f1:
                best_valid_f1 = f1_m_valid
                print('Save the tuned model....')
                model.save_pretrained(odir)


def train_doc2v(dname, raw_dir='./data/raw/', odir='./resources/embedding/'):
    ''' Build paragraph2vec model
    '''
    def read_corpus(dpath):
        with open(dpath) as dfile:
            columns = dfile.readline().strip().split('\t')
            text_idx = columns.index('text')
            for idx, line in enumerate(dfile):
                line = line.split('\t')[text_idx].strip().split()        
                yield TaggedDocument(line, [idx])

    odir = odir + dname + '/'
    if not os.path.exists(odir):
        os.mkdir(odir)

    # load the corpus
    dpath = raw_dir + dname + '/' + dname + '.tsv'
    corpus = read_corpus(dpath)

    # init, train and save the model
    model = Doc2Vec(
        vector_size=300, min_count=2, epochs=40, 
        workers=8, max_vocab_size=20000
    )
    model.build_vocab(corpus)

    model.train(
        corpus, total_examples=model.corpus_count, 
        epochs=model.epochs
    )

    model.save(odir + 'doc2v.model')


if __name__ == '__main__':
    odir = './resources/embedding_vocab/'
    raw_dir = './data/raw/'
    encode_dir = './data/encode/'

    dname = sys.argv[1] # ['amazon', 'yelp', 'imdb']
    mname = sys.argv[2] # lda, word2vec, doc2vec and bert
    if dname not in ['amazon', 'yelp', 'imdb']:
        raise ValueError('Data {} is not supported currently...'.format(dname))

    if mname == 'word2vec':
        print('Training Word Embeddings: ', dname)
        train_w2v(dname, raw_dir=raw_dir, odir=odir)
    elif mname == 'lda':
        print('Training LDA: ', dname)
        train_lda(dname, raw_dir=raw_dir, odir=odir)
    elif mname == 'doc2vec':
        print('Training Doc2vec: ', dname)
        train_doc2v(dname, raw_dir=raw_dir, odir=odir)

    elif mname == 'bert':
        # load the params (max_len) and fine tune BERT
        print('Fine Tuning Google BERT: ', dname)
        if not os.path.exists(odir):
            os.mkdir(odir)
        odir = odir + dname + '/'
        if not os.path.exists(odir):
            os.mkdir(odir)
        odir = odir + 'bert/'
        if not os.path.exists(odir):
            os.mkdir(odir)

        params = json.load(open(encode_dir+dname+'/params.json'))
        params['decay_rate'] = .001
        params['lr'] = 1e-5
        params['warm_steps'] = 100
        params['train_steps'] = 1000
        params['batch_size'] = 32
        params['balance'] = True
        bmodel = FineTuneBert(dname, raw_dir=raw_dir, odir=odir, params=params)
        bmodel.tune_bert()
    else:
        raise ValueError('Model name, {}, is not in supported now...'.format(mname))


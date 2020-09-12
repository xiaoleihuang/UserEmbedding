import os
import json
import pickle

from keras.layers import GRU, Bidirectional
from keras.layers import Input, Embedding, Dense
from keras.layers import Dropout
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras

from sklearn.metrics import f1_score, classification_report
from sklearn.utils import class_weight
import numpy as np
import gensim
# from imblearn.over_sampling import RandomOverSampler
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for CPU-use only
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def data_iter(docs, labels, batch_size=32, sample_weight=None):
    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for idx in range(steps):
        batch_data = np.asarray(docs[idx * batch_size: (idx + 1) * batch_size])
        batch_label = np.asarray(labels[idx * batch_size: (idx + 1) * batch_size])
        if sample_weight is not None:
            batch_weight = np.asarray(sample_weight[idx * batch_size: (idx + 1) * batch_size])
            yield batch_data, batch_label, batch_weight
        else:
            yield batch_data, batch_label


def load_data(filename, max_len, tkn_path, train=True, do_balance=False):
    labels = []
    docs = []

    with open(filename) as dfile:
        cols = dfile.readline()  # skip the 1st column names
        cols = cols.strip().split('\t')
        doc_idx = cols.index('text')  # document index
        label_idx = cols.index('label')  # label index

        for line in dfile:
            infos = line.strip().split('\t')
            labels.append(int(infos[label_idx]))
            docs.append(infos[doc_idx].split())

    # tokenize and padding the documents
    tkn = pickle.load(open(tkn_path, 'rb'))
    docs = pad_sequences(tkn.texts_to_sequences(docs), maxlen=max_len)

    if train:
        # convert label to one hot labels
        num_label = len(np.unique(labels))
        if num_label > 2:
            tmp_labels = [[0] * num_label for _ in range(len(labels))]
            for idx, item in enumerate(labels):
                tmp_labels[idx][item] = 1

            labels = tmp_labels

        if do_balance:
            sample_weight = None
            # over sampling TODO

        else:
            sample_weight = class_weight.compute_sample_weight(
                'balanced', labels
            )

        return docs, labels, sample_weight
    else:
        return docs, labels


def build_wt(filep='', opt='embd.npy', tkn_path=''):
    """Build embedding weights by the tokenizer
        filep: the embedding file path
    """
    size = 300  # embedding size, in this study, we use 300

    if os.path.exists(opt):
        return np.load(opt)
    else:
        tkn = pickle.load(open(tkn_path, 'rb'))
        embed_len = len(tkn.word_index)
        if embed_len > tkn.num_words:
            embed_len = tkn.num_words

        # load embedding
        emb_matrix = np.zeros((embed_len + 1, size))

        if filep.endswith('.bin'):
            embeds = gensim.models.KeyedVectors.load_word2vec_format(
                filep, binary=True
            )

            for pair in zip(embeds.wv.index2word, embeds.wv.syn0):
                if pair[0] in tkn.word_index and \
                        tkn.word_index[pair[0]] < tkn.num_words:
                    emb_matrix[tkn.word_index[pair[0]]] = [
                        float(item) for item in pair[1]
                    ]
        else:
            with open(filep) as dfile:
                for line in dfile:
                    line = line.strip().split()

                    if line[0] in tkn.word_index and \
                            tkn.word_index[line[0]] < tkn.num_words:
                        emb_matrix[tkn.word_index[line[0]]] = [
                            float(item) for item in line[1:]
                        ]
        np.save(opt, emb_matrix)
        return np.asarray(emb_matrix)


def run_bilstm(data_name, params):
    """ input > embedding > Bi-LSTM > dense > dropout > sigmoid
    """
    print('Working on: ' + data_name)
    # load w2v weights for the Embedding
    weights = build_wt(
        params['emb_dir'] + data_name + '/w2v.txt',
        params['weight_dir'] + data_name + '.npy',
        params['encode_dir'] + data_name + '/{}.tkn'.format(data_name)
    )

    text_input = Input(shape=(params['max_len'],), dtype='int32')
    embed = Embedding(
        weights.shape[0], weights.shape[1],  # size of data embedding
        weights=[weights], input_length=params['max_len'],
        trainable=True,
        name='embedding')(text_input)

    bilstm = Bidirectional(GRU(
        weights.shape[1],
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=keras.regularizers.l1_l2(0, 0.0001),
        dropout=params['dp_rate'],
        # recurrent_activation='tanh',
    ))(embed)

    # dense
    dense_l = Dense(params['hidden_num'], activation='relu')(bilstm)
    dp_l = Dropout(params['dp_rate'])(dense_l)

    # output
    pred_l = Dense(params['num_class'], activation='softmax')(dp_l)
    model = Model(inputs=text_input, outputs=pred_l)

    # compile model
    if params['optimizer'] == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=0.0001)
    elif params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    else:
        optimizer = keras.optimizers.SGD(learning_rate=0.0001)

    if params['num_class'] > 2:
        model.compile(
            optimizer=optimizer, loss='categorical_crossentropy',
            metrics=['accuracy'])
    else:
        model.compile(
            optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    print(model.summary())
    best_valid_f1 = 0.0

    # fit the model
    train_path = params['data_dir'] + data_name + '/train.tsv'
    train_docs, train_labels, sample_weight = load_data(
        train_path, params['max_len'],
        params['encode_dir'] + data_name + '/{}.tkn'.format(data_name),
        train=True, do_balance=params['balance_data']
    )
    valid_path = params['data_dir'] + data_name + '/valid.tsv'
    valid_docs, valid_labels = load_data(
        valid_path, params['max_len'],
        params['encode_dir'] + data_name + '/{}.tkn'.format(data_name),
        train=False
    )
    test_path = params['data_dir'] + data_name + '/test.tsv'

    for e in range(params['epochs']):
        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))
        train_iter = data_iter(train_docs, train_labels, params['batch_size'], sample_weight)

        # train on batches
        for train_batch in train_iter:
            if sample_weight:
                x_train, y_train, batch_weight = train_batch
            else:
                x_train, y_train = train_batch
                batch_weight = None
            # skip only 1 class in the training data
            if len(np.unique(y_train)) == 1:
                continue

            # train sentiment model
            if batch_weight:
                tmp_senti = model.train_on_batch(
                    x_train, y_train,
                    sample_weight=batch_weight
                )
            else:
                tmp_senti = model.train_on_batch(
                    x_train, y_train
                )
            # calculate loss and accuracy
            loss += tmp_senti[0]
            loss_avg = loss / step
            accuracy += tmp_senti[1]
            accuracy_avg = accuracy / step

            if step % 40 == 0:
                print('Step: {}'.format(step))
                print('\tLoss: {}.'.format(loss_avg))
                print('\tAccuracy: {}.'.format(accuracy_avg))
                print('-------------------------------------------------')
            step += 1

        # each epoch try the valid data, get the best valid-weighted-f1 score
        print('Validating....................................................')
        valid_iter = data_iter(valid_docs, valid_labels, params['batch_size'])

        y_preds_valids = []
        y_valids = []
        for x_valid, y_valid in valid_iter:
            x_valid = np.asarray(x_valid)
            tmp_preds_valid = model.predict(x_valid)
            for item_tmp in tmp_preds_valid:
                y_preds_valids.append(item_tmp)
            for item_tmp in y_valid:
                y_valids.append(int(item_tmp))

        if len(y_preds_valids[0]) > 2:
            y_preds_valids = np.argmax(y_preds_valids, axis=1)
        else:
            y_preds_valids = [np.round(item[0]) for item in y_preds_valids]
        f1_valid = f1_score(y_true=y_valids, y_pred=y_preds_valids, average='weighted')
        print('Validating f1-weighted score: ' + str(f1_valid))

        # if the validation f1 score is good, then test
        if f1_valid > best_valid_f1:
            best_valid_f1 = f1_valid
            test_docs, test_labels = load_data(
                test_path, params['max_len'],
                params['encode_dir'] + data_name + '/{}.tkn'.format(data_name),
                train=False
            )
            test_iter = data_iter(test_docs, test_labels, params['batch_size'])

            y_preds = []
            y_tests = []
            for x_test, y_test in test_iter:
                x_test = np.asarray(x_test)
                tmp_preds = model.predict(x_test)
                for item_tmp in tmp_preds:
                    y_preds.append(item_tmp)
                for item_tmp in y_test:
                    y_tests.append(int(item_tmp))

            if len(y_preds[0]) > 2:
                y_preds = np.argmax(y_preds, axis=1)
            else:
                y_preds = [np.round(item[0]) for item in y_preds]

            test_result = open('./results/bilstm_results.txt', 'a')
            test_result.write(data_name + '\n')
            test_result.write('Epoch ' + str(e) + '..................................................\n')
            test_result.write(str(f1_score(y_true=y_tests, y_pred=y_preds, average='weighted')) + '\n')
            test_result.write('#####\n\n')
            test_result.write(classification_report(y_true=y_tests, y_pred=y_preds, digits=3))
            test_result.write('...............................................................\n\n')
            test_result.flush()


if __name__ == '__main__':
    if not os.path.exists('./vects/'):
        os.mkdir('./vects/')
    if not os.path.exists('./clfs/'):
        os.mkdir('./clfs/')
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    data_list = [
        'amazon_health',
        'imdb',
        'yelp'
    ]
    parameters = {
        'epochs': 15,
        'num_class': 3,
        'optimizer': 'rmsprop',
        'hidden_num': 200,
        'dp_rate': 0.2,
        'batch_size': 32,
        'encode_dir': '../data/encode/',
        'data_dir': '../data/raw/',
        'emb_dir': '../resources/embedding/',
        'weight_dir': './vects/',
        'balance_data': False,
    }

    # load data stats to determine the max length
    try:
        stats = json.load(open('../analysis/stats.json'))
    except FileNotFoundError:
        stats = None

    for dname in data_list:
        if stats and dname in stats:
            parameters['max_len'] = int(stats[dname].get('75_percent_word_per_doc', 200))
        else:
            parameters['max_len'] = 200

        run_bilstm(dname, parameters)

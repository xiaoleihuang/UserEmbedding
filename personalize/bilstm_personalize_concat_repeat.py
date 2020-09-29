import os
import json
import pickle
import sys

from keras.layers import GRU, Bidirectional
from keras.layers import Input, Embedding, Dense
from keras.layers import Dropout, Concatenate, Add, RepeatVector
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
import keras

from sklearn.metrics import f1_score, classification_report
from sklearn.utils import class_weight
import numpy as np
import gensim
# from imblearn.over_sampling import RandomOverSampler
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # for CPU-use only
# os.environ["CUDA_VISIBLE_DEVICES"] = ""


def data_iter(**kwargs):
    docs = kwargs['docs']
    labels = kwargs['labels']
    batch_size = kwargs.get('batch_size', 32)
    sample_weight = kwargs.get('sample_weight', None)
    products = kwargs.get('products', None)
    users = kwargs.get('users', None)

    # shuffle the data
    data_indices = list(range(len(docs)))
    np.random.shuffle(data_indices)
    labels = [labels[item] for item in data_indices]
    docs = [docs[item] for item in data_indices]
    if sample_weight is not None:
        sample_weight = [sample_weight[item] for item in data_indices]
    if users is not None:
        users = [users[item] for item in data_indices]
    if products is not None:
        products = [products[item] for item in data_indices]

    steps = int(len(docs) / batch_size)
    if len(docs) % batch_size != 0:
        steps += 1

    for idx in range(steps):
        batch_data = np.asarray(docs[idx * batch_size: (idx + 1) * batch_size])
        batch_label = np.asarray(labels[idx * batch_size: (idx + 1) * batch_size])
        if sample_weight is not None:
            batch_weight = np.asarray(sample_weight[idx * batch_size: (idx + 1) * batch_size])
        else:
            batch_weight = np.asarray([])
        if users is not None:
            batch_user = np.asarray(users[idx * batch_size: (idx + 1) * batch_size])
        else:
            batch_user = np.asarray([])
        if products is not None:
            batch_product = np.asarray(products[idx * batch_size: (idx + 1) * batch_size])
        else:
            batch_product = np.asarray([])

        yield batch_data, batch_label, batch_weight, batch_user, batch_product


def load_data(filename, max_len, tkn_path, train=True, do_balance=False):
    labels = []
    docs = []
    users = []
    products = []

    with open(filename) as dfile:
        cols = dfile.readline()  # skip the 1st column names
        cols = cols.strip().split('\t')
        doc_idx = cols.index('text')  # document index
        label_idx = cols.index('label')  # label index
        user_idx = cols.index('uid')
        product_idx = cols.index('bid')

        for line in dfile:
            infos = line.strip().split('\t')
            labels.append(int(infos[label_idx]))
            docs.append(infos[doc_idx].split())
            users.append(int(infos[user_idx]))
            products.append(int(infos[product_idx]))

    # tokenize and padding the documents
    tkn = pickle.load(open(tkn_path, 'rb'))
    docs = pad_sequences(tkn.texts_to_sequences(docs), maxlen=max_len)

    if train:
        if do_balance:
            sample_weight = None
            # over sampling
            sample_indices = dict()
            for idx, item in enumerate(labels):
                if item not in sample_indices:
                    sample_indices[item] = []
                sample_indices[item].append(idx)
            sample_max = max([len(sample_indices[item]) for item in sample_indices])
            tmp_indices = []
            for item in sample_indices:
                tmp_indices.extend(sample_indices[item])
                if len(sample_indices[item]) < sample_max:
                    tmp_indices.extend(
                        np.random.choice(
                            sample_indices[item],
                            size=sample_max - len(sample_indices[item]),
                            replace=True
                        )
                    )
            sample_indices = tmp_indices
            np.random.shuffle(sample_indices)

            users = np.asarray([users[item] for item in sample_indices])
            products = np.asarray([products[item] for item in sample_indices])
            docs = np.asarray([docs[item] for item in sample_indices])
            labels = np.asarray([labels[item] for item in sample_indices])
        else:
            sample_weight = class_weight.compute_sample_weight(
                'balanced', labels
            )

        # convert label to one hot labels
        num_label = len(np.unique(labels))
        if num_label > 2:
            tmp_labels = [[0] * num_label for _ in range(len(labels))]
            for idx, item in enumerate(labels):
                tmp_labels[idx][item] = 1

            labels = tmp_labels

        return docs, labels, sample_weight, users, products
    else:
        return docs, labels, users, products


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
    # suffix of output files
    output_suffix = '_'

    # load the user and product embeddings
    uemb_path = os.path.join(params['up_dir'], 'user.npy')
    if params['use_uemb'] and os.path.exists(uemb_path):
        print('Loading user embedding...: ', uemb_path)
        output_suffix += 'u'
        user_emb = np.load(uemb_path)
    else:
        user_emb = None

    pemb_path = os.path.join(params['up_dir'], 'product.npy')
    if params['use_pemb'] and os.path.exists(pemb_path):
        print('Loading product embedding...: ', pemb_path)
        output_suffix += 'p'
        product_emb = np.load(pemb_path)
    else:
        product_emb = None

    # load w2v weights for the Embedding
    weights = build_wt(
        params['emb_dir'] + data_name + '/w2v.txt',
        params['weight_dir'] + data_name + '.npy',
        params['encode_dir'] + data_name + '/{}.tkn'.format(data_name)
    )

    text_input = Input(shape=(params['max_len'],), dtype='int32', name='text_input')
    embed = Embedding(
        weights.shape[0], weights.shape[1],  # size of data embedding
        weights=[weights], input_length=params['max_len'],
        trainable=True,
        name='embedding')(text_input)

    # concatenate user, product and the text
    # user and product embeddings
    if user_emb is not None:
        user_input = Input(shape=(1,), dtype='int32', name='user_input')
        user_emb_layer = Embedding(
            user_emb.shape[0], user_emb.shape[1], weights=[user_emb],
            input_length=1, trainable=True, name='user_emb'
        )(user_input)
        user_emb_layer = RepeatVector(parameters['max_len'])(
            keras.layers.Lambda(lambda element: element[:, 0, :])(user_emb_layer))
        embed = Concatenate()([embed, user_emb_layer])
    else:
        user_input = None

    if product_emb is not None:
        product_input = Input(shape=(1,), dtype='int32', name='product_input')
        product_emb_layer = Embedding(
            product_emb.shape[0], product_emb.shape[1], weights=[product_emb],
            input_length=1, trainable=True, name='product_emb'
        )(product_input)
        product_emb_layer = RepeatVector(parameters['max_len'])(
            keras.layers.Lambda(lambda element: element[:, 0, :])(product_emb_layer))
        embed = Concatenate()([embed, product_emb_layer])
    else:
        product_input = None

    bilstm = Bidirectional(GRU(
        weights.shape[1],
        kernel_initializer="glorot_uniform",
        # kernel_regularizer=keras.regularizers.l1_l2(0, 0.0001),
        dropout=params['dp_rate'],
        # recurrent_activation='tanh',
    ))(embed)

    # dense
    dense_l = Dense(params['hidden_num'], activation='relu', name='dense')(bilstm)
    dp_l = Dropout(params['dp_rate'])(dense_l)

    # output
    pred_l = Dense(params['num_class'], activation='softmax', name='prediction')(dp_l)

    if user_input is None and product_input is not None:
        model = Model(inputs=[text_input, product_input], outputs=pred_l)
    elif user_input is not None and product_input is None:
        model = Model(inputs=[text_input, user_input], outputs=pred_l)
    elif user_input is not None and product_input is not None:
        model = Model(inputs=[text_input, user_input, product_input], outputs=pred_l)
    else:
        model = Model(inputs=text_input, outputs=pred_l)

    # compile model
    if params['optimizer'] == 'rmsprop':
        optimizer = keras.optimizers.RMSprop(learning_rate=params['lr_rate'])
    elif params['optimizer'] == 'adam':
        optimizer = keras.optimizers.Adam(learning_rate=params['lr_rate'])
    else:
        optimizer = keras.optimizers.SGD(learning_rate=params['lr_rate'])

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
    train_data = load_data(
        train_path, params['max_len'],
        params['encode_dir'] + data_name + '/{}.tkn'.format(data_name),
        train=True, do_balance=params['balance_data']
    )
    valid_path = params['data_dir'] + data_name + '/valid.tsv'
    valid_data = load_data(
        valid_path, params['max_len'],
        params['encode_dir'] + data_name + '/{}.tkn'.format(data_name),
        train=False
    )
    test_path = params['data_dir'] + data_name + '/test.tsv'
    test_data = load_data(
        test_path, params['max_len'],
        params['encode_dir'] + data_name + '/{}.tkn'.format(data_name),
        train=False
    )
    test_docs, test_labels, test_users, test_products = test_data

    for e in range(params['epochs']):
        accuracy = 0.0
        loss = 0.0
        step = 1

        print('--------------Epoch: {}--------------'.format(e))
        train_docs, train_labels, sample_weight, train_users, train_products = train_data
        train_iter = data_iter(
            docs=train_docs, labels=train_labels, sample_weight=sample_weight,
            batch_size=params['batch_size'], users=train_users, products=train_products
        )

        # train on batches
        for train_batch in train_iter:
            batch_data, batch_label, batch_weight, batch_user, batch_product = train_batch
            # skip only 1 class in the training data
            if len(np.unique(batch_label)) == 1:
                continue

            x = dict()
            x['text_input'] = batch_data
            if user_emb is not None:
                x['user_input'] = batch_user
            if product_emb is not None:
                x['product_input'] = batch_product

            # train sentiment model
            if batch_weight is not None and len(batch_weight) > 0:
                tmp_senti = model.train_on_batch(
                    x=x if len(x) > 1 else batch_data,
                    y=batch_label,
                    sample_weight=batch_weight
                )
            else:
                tmp_senti = model.train_on_batch(
                    x=x if len(x) > 1 else batch_data,
                    y=batch_label,
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
        valid_docs, valid_labels, valid_users, valid_products = valid_data
        valid_iter = data_iter(
            docs=valid_docs, labels=valid_labels, batch_size=params['batch_size'],
            users=valid_users, products=valid_products
        )

        y_preds_valids = []
        y_valids = []
        for valid_batch in valid_iter:
            batch_data, batch_label, batch_weight, batch_user, batch_product = valid_batch
            x_valid = dict()
            x_valid['text_input'] = batch_data
            if user_emb is not None:
                x_valid['user_input'] = batch_user
            if product_emb is not None:
                x_valid['product_input'] = batch_product

            tmp_preds_valid = model.predict(x_valid if len(x_valid) > 1 else batch_data)
            for item_tmp in tmp_preds_valid:
                y_preds_valids.append(item_tmp)
            for item_tmp in batch_label:
                y_valids.append(int(item_tmp))

        if len(y_preds_valids[0]) > 2:
            y_preds_valids = np.argmax(y_preds_valids, axis=1)
        else:
            y_preds_valids = [np.round(item[0]) for item in y_preds_valids]
        f1_valid = f1_score(y_true=y_valids, y_pred=y_preds_valids, average='weighted')
        print('Validating f1-weighted score: ' + str(f1_valid))

        # if the validation f1 score is good, then test
        # if f1_valid > best_valid_f1:
        best_valid_f1 = f1_valid
        test_iter = data_iter(
            docs=test_docs, labels=test_labels, batch_size=params['batch_size'],
            users=test_users, products=test_products
        )

        y_preds = []
        y_tests = []
        for test_batch in test_iter:
            batch_data, batch_label, batch_weight, batch_user, batch_product = test_batch
            x_test = dict()
            x_test['text_input'] = batch_data
            if user_emb is not None:
                x_test['user_input'] = batch_user
            if product_emb is not None:
                x_test['product_input'] = batch_product

            tmp_preds = model.predict(x_test if len(x_test) > 1 else batch_data)
            for item_tmp in tmp_preds:
                y_preds.append(item_tmp)
            for item_tmp in batch_label:
                y_tests.append(int(item_tmp))

        if len(y_preds[0]) > 2:
            y_preds = np.argmax(y_preds, axis=1)
        else:
            y_preds = [np.round(item[0]) for item in y_preds]

        test_result = open('./results/{0}_{1}_results.txt'.format(
            sys.argv[0].split('.')[0],
            output_suffix), 'a'
        )
        test_result.write(data_name + '\n')
        test_result.write(json.dumps(params) + '\n')
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
#        'amazon_health',
        'imdb',
#        'yelp'
    ]
    parameters = {
        'epochs': 20,
        'num_class': 3,
        'optimizer': 'rmsprop',
        'hidden_num': 200,
        'dp_rate': 0.2,
        'batch_size': 64,
        'encode_dir': '../data/encode/',
        'data_dir': '../data/raw/',
        'emb_dir': '../resources/embedding/',
        'weight_dir': './vects/',
        'balance_data': True,
        'lr_rate': 0.0001,
        'use_uemb': True,
        'use_pemb': True,
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
        parameters['up_dir'] = '../resources/skipgrams/' + dname + '/word_user_product/'

        run_bilstm(dname, parameters)

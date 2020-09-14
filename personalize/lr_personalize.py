from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
# from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import scipy.sparse as sp

import os
import pickle


def run_lr_3gram(data_name, train_path, test_path, per_dir=None):
    """
    :param str data_name: the name of input data
    :param str train_path: training path
    :param str test_path: testing file path
    :param str per_dir: directory path of user and product embedding text files
    :return:
    """
    print('Working on: ' + data_name)
    # suffix of output files
    output_suffix = '_'

    # load the user and product embeddings
    uemb_path = os.path.join(per_dir, 'user.npy')
    pemb_path = os.path.join(per_dir, 'product.npy')
    if os.path.exists(uemb_path):
        print('Loading user embedding...: ', uemb_path)
        output_suffix += 'u'
        user_emb = np.load(uemb_path)
    else:
        user_emb = None

    if os.path.exists(pemb_path):
        print('Loading product embedding...: ', pemb_path)
        output_suffix += 'p'
        product_emb = np.load(pemb_path)
    else:
        product_emb = None

    # check if the vectorizer and exists
    # build the vectorizer
    if not os.path.exists('./clfs/lr_p_' + data_name + '{}.pkl'.format(output_suffix)):

        print('Loading Training data........')
        # load the training data
        train_docs = []
        train_labels = []
        train_user = []
        train_product = []

        with open(train_path) as train_file:
            cols = train_file.readline()  # skip the 1st column names
            cols = cols.strip().split('\t')
            doc_idx = cols.index('text')  # document index
            label_idx = cols.index('label')  # label index
            user_idx = cols.index('uid')
            product_idx = cols.index('bid')

            for line in train_file:
                if len(line.strip()) < 5:
                    continue

                infos = line.strip().split('\t')
                train_labels.append(int(infos[label_idx]))
                train_docs.append(infos[doc_idx].strip())
                if user_emb is not None:
                    train_user.append(int(infos[user_idx].strip()))
                if product_emb is not None:
                    train_product.append(int(infos[product_idx].strip()))
        print(np.unique(train_labels))

        print('Fitting Vectorizer.......')
        if os.path.exists('./vects/lr_p_' + data_name + '{}.pkl'.format(output_suffix)):
            vect = pickle.load(
                open('./vects/lr_p_' + data_name + '{}.pkl'.format(output_suffix), 'rb')
            )
        else:
            vect = TfidfVectorizer(ngram_range=(1, 3), max_features=15000, min_df=2)
            vect.fit(train_docs)
            # save the vectorizer
            pickle.dump(vect, open('./vects/lr_p_' + data_name + '{}.pkl'.format(output_suffix), 'wb'))

        print('Transforming Training data........')
        train_docs = vect.transform(train_docs)
        print(train_docs.shape)
        if user_emb is not None:
            train_user = sp.csr_matrix([user_emb[item] for item in train_user], dtype=np.float64)
            train_docs = sp.hstack([train_docs, train_user], dtype=np.float64)
        if product_emb is not None:
            train_product = sp.csr_matrix([product_emb[item] for item in train_product], dtype=np.float64)
            train_docs = sp.hstack([train_docs, train_product], dtype=np.float64)
        print(train_docs.shape)

        # fit the model
        if len(np.unique(train_labels)) > 2:
            # clf = SGDClassifier(loss='log', class_weight='balanced', multi_class='multinomial')
            clf = LogisticRegression(class_weight='balanced', multi_class='multinomial', max_iter=2000)
        else:
            # clf = SGDClassifier(loss='log', class_weight='balanced')
            clf = LogisticRegression(class_weight='balanced', max_iter=2000)
        clf.fit(train_docs, train_labels)
        # save the classifier
        pickle.dump(clf, open('./clfs/lr_p_' + data_name + '{}.pkl'.format(output_suffix), 'wb'))
    else:
        vect = pickle.load(open('./vects/lr_p_' + data_name + '{}.pkl'.format(output_suffix), 'rb'))
        clf = pickle.load(open('./clfs/lr_p_' + data_name + '{}.pkl'.format(output_suffix), 'rb'))

    # load the test data
    test_docs = []
    test_labels = []
    test_user = []
    test_product = []

    with open(test_path) as test_file:
        cols = test_file.readline()  # skip the 1st column names
        cols = cols.strip().split('\t')
        doc_idx = cols.index('text')  # document index
        label_idx = cols.index('label')  # label index
        user_idx = cols.index('uid')
        product_idx = cols.index('bid')

        for line in test_file:
            if len(line.strip()) < 5:
                continue
            infos = line.strip().split('\t')
            test_labels.append(int(infos[label_idx]))
            test_docs.append(infos[doc_idx].strip())
            if user_emb is not None:
                test_user.append(int(infos[user_idx].strip()))
            if product_emb is not None:
                test_product.append(int(infos[product_idx].strip()))

    # transform the test data
    print('Testing.........')
    test_docs = vect.transform(test_docs)
    if user_emb is not None:
        test_user = sp.csr_matrix([user_emb[item] for item in test_user], dtype=np.float64)
        test_docs = sp.hstack([test_docs, test_user], dtype=np.float64)
    if product_emb is not None:
        test_product = sp.csr_matrix([product_emb[item] for item in test_product], dtype=np.float64)
        test_docs = sp.hstack([test_docs, test_product], dtype=np.float64)

    y_preds = clf.predict(test_docs)
    with open('./results/lr_personalize{}_results.txt'.format(output_suffix), 'a') as writefile:
        writefile.write(data_name + '_________________\n')
        writefile.write(str(f1_score(y_pred=y_preds, y_true=test_labels, average='weighted')) + '\n')
        writefile.write(classification_report(y_pred=y_preds, y_true=test_labels, digits=3) + '\n')
        writefile.write('.........................\n')


if __name__ == '__main__':
    # create directories for saving models and tokenizers
    if not os.path.exists('./vects/'):
        os.mkdir('./vects/')
    if not os.path.exists('./clfs/'):
        os.mkdir('./clfs/')
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    data_list = [
        # 'imdb',
        'yelp',
        # 'amazon_health',
    ]

    for dname in data_list:
        train_raw_path = '../data/raw/' + dname + '/train.tsv'
        test_raw_path = '../data/raw/' + dname + '/test.tsv'
        up_dir = '../resources/skipgrams/' + dname + '/word_user_product/'
        run_lr_3gram(dname, train_raw_path, test_raw_path, up_dir)

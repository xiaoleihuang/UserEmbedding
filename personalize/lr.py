from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
# from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import os
import pickle


def run_lr_3gram(data_name, train_path, test_path):
    """

    :param data_name:
    :type data_name: str
    :param train_path: training path
    :type train_path: str
    :param test_path: testing file path
    :type test_path: str
    :return:
    """
    print('Working on: '+data_name)
    # check if the vectorizer and exists
    # build the vectorizer
    if not (os.path.exists('./vects/lr_' + data_name + '.pkl') and
            os.path.exists('./clfs/lr_' + data_name + '.pkl')):

        print('Loading Training data........')
        # load the training data
        train_docs = []
        train_labels = []
        with open(train_path) as train_file:
            cols = train_file.readline()  # skip the 1st column names
            cols = cols.strip().split('\t')
            doc_idx = cols.index('text')  # document index
            label_idx = cols.index('label')  # label index

            for line in train_file:
                if len(line.strip()) < 5:
                    continue
                
                infos = line.strip().split('\t')
                train_labels.append(int(infos[label_idx]))
                train_docs.append(infos[doc_idx].strip())
        print(np.unique(train_labels))
        
        print('Fitting Vectorizer.......')
        vect = CountVectorizer(ngram_range=(1, 3), max_features=15000, min_df=2)
        vect.fit(train_docs)
        pickle.dump(vect, open('./vects/lr_'+data_name+'.pkl', 'wb'))  # save the vectorizer

        print('Transforming Training data........')
        train_docs = vect.transform(train_docs)

        # fit the model
        print('Building model............')
        if len(np.unique(train_labels)) > 2:
            # clf = SGDClassifier(loss='log', class_weight='balanced')
            clf = LogisticRegression(class_weight='balanced', multi_class='auto')
        else:
            # clf = SGDClassifier(loss='log', class_weight='balanced')
            clf = LogisticRegression(class_weight='balanced')
        clf.fit(train_docs, train_labels)
        pickle.dump(clf, open('./clfs/lr_' + data_name + '.pkl', 'wb'))  # save the classifier
    else:
        vect = pickle.load(open('./vects/lr_'+data_name+'.pkl', 'rb'))
        clf = pickle.load(open('./clfs/lr_'+data_name+'.pkl', 'rb'))

    # load the test data
    test_docs = []
    test_labels = []
    with open(test_path) as test_file:
        cols = test_file.readline()  # skip the 1st column names
        cols = cols.strip().split('\t')
        doc_idx = cols.index('text')  # document index
        label_idx = cols.index('label')  # label index
        for line in test_file:
            if len(line.strip()) < 5:
                continue
            infos = line.strip().split('\t')
            test_labels.append(int(infos[label_idx]))
            test_docs.append(infos[doc_idx].strip())

    # transform the test data
    print('Testing.........')
    test_docs = vect.transform(test_docs)
    y_preds = clf.predict(test_docs)

    with open('./results/lr_results.txt', 'a') as writefile:
        writefile.write(data_name + '_________________\n')
        writefile.write(str(f1_score(y_pred=y_preds, y_true=test_labels, average='weighted'))+'\n')
        report = classification_report(y_pred=y_preds, y_true=test_labels, digits=3)
        print(report)
        writefile.write(report + '\n')
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
        'imdb',
        'yelp',
        'amazon_health',
    ]

    for dname in data_list:
        train_raw_path = '../data/raw/'+dname+'/train.tsv'
        test_raw_path = '../data/raw/'+dname+'/test.tsv'
        run_lr_3gram(dname, train_raw_path, test_raw_path)

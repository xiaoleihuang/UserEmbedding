'''This code quantitatively measure how user differs from each other.

To measure user differences, we approximate this question by meansuring differences across different user groups. We first group the users across their purchasing habbit.

'''
import os, sys
import json
import pickle
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from scipy.sparse import hstack


def build_user_data(dname, data_dir, odir):
    """
    
    Params:
        dname (str): data name
        data_dir (str): the directory path of the input tsv
        odir (str): output dir
    """
    # create user groups
    group_opath = os.path.join(odir + '/{}_group.json'.format(dname))
    if not os.path.exists(group_opath):
        group_docs = dict()
        with open(os.path.join(data_dir, '{}.tsv'.format(dname))) as dfile:
            cols = dfile.readline().strip().split('\t')
            uid_idx = cols.index('uid')
            text_idx = cols.index('text')
            genre_idx = cols.index('genre')
            label_idx = cols.index('label')

            for line in dfile:
                line = line.strip()
                if len(line) < 5:
                    continue

                line = line.split('\t')
                if np.random.random() > .2:
                    doc_class = 'train'
                else:
                    doc_class = 'test'

                for genre in line[genre_idx].split(','):
                    if genre not in group_docs:
                        group_docs[genre] = {
                            'train': [],
                            'test': [],
                        }
                    group_docs[genre][doc_class].append([line[text_idx], int(line[label_idx])])

        # save to the path
        with open(group_opath, 'w') as wfile:
            json.dump(group_docs, wfile)
    
    # create a sample from the training set that has the same size as the validation set
    sample_opath = os.path.join(odir + '/{}_sample.tsv'.format(dname))
    sample_ratio = 0.2

    if not os.path.exists(sample_opath):
        sample_docs = list()
        with open(sample_opath, 'w') as wfile:
            with open(os.path.join(data_dir, 'train.tsv')) as dfile:
                wfile.write(dfile.readline())  # write the column names
                for line in dfile:
                    if np.random.random() < sample_ratio:
                        wfile.write(line)

    return group_opath, sample_opath



def clf_analysis(dname, inpath, odir):
    """Perform cross group classification analysis and visualize them into a heatmap
    """
    if not os.path.exists(odir + '/{}_group_results.json'.format(dname)):
        group_file = json.load(open(inpath))
        clf_results = dict()
        all_keys = list(sorted(list(group_file.keys())))  # rank by alphabet

        for gkey in group_file:
            gkey_idx = all_keys.index(gkey)
            clf_results[gkey_idx] = dict()
            x_train, y_train = zip(*group_file[gkey]['train'])

            # check if the vectorizer exists
            vect_path = odir + '/{0}_vect_{1}.pkl'.format(dname, gkey_idx)
            if os.path.exists(vect_path):
                vect = pickle.load(open(vect_path, 'rb'))
            else:
                vect = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=2)
                vect.fit(x_train)
                pickle.dump(vect, open(vect_path, 'wb'))

            # check if the classifier exists
            clf_path = odir + '/{0}_clf_{1}.pkl'.format(dname, gkey_idx)
            if os.path.exists(clf_path):
                clf = pickle.load(open(clf_path, 'rb'))
            else:
                clf = LogisticRegression(class_weight='balanced', multi_class='auto')
                clf.fit(vect.transform(x_train), y_train)
                pickle.dump(clf, open(clf_path, 'wb'))

            for other_key in group_file:
                other_key_idx = all_keys.index(other_key)
                if other_key_idx in clf_results and gkey_idx in clf_results[other_key_idx]:
                    clf_results[gkey_idx][other_key_idx] = clf_results[other_key_idx][gkey_idx]
                else:
                    x_test, y_test = zip(*group_file[other_key]['test'])
                    y_pred = clf.predict(vect.transform(x_test))
                    clf_results[gkey_idx][other_key_idx] = round(f1_score(y_pred=y_pred, y_true=y_test, average='weighted'), 4) * 100

        # save the classification results
        json.dump(clf_results, open(odir + '/{}_group_results.json'.format(dname), 'w'))

    return odir + '/{}_group_results.json'.format(dname)


def viz_clf(dpath, dname, opath):

    df = OrderedDict(json.load(open(dpath)))
    keys = list(sorted(df.keys()))
    df = pd.DataFrame(df)
    df = df[keys].transpose()  # rank the output
    keys.reverse()
    df = df[keys].transpose()
    center = np.median([item for item in df.to_numpy().ravel() if item != 1])
    
    a4_dims = (12.27, 12.27)
    fig, ax = plt.subplots(figsize=a4_dims)
    sns.set(font_scale=1.2)
    cmap = plt.get_cmap("RdBu_r")

    viz_plot = sns.heatmap(
        df, annot=True, cbar=False,  
        ax=ax, annot_kws={"size": 36}, cmap=cmap, 
        vmin=df.values.min(), fmt='.2f', center=center
    )
    plt.xticks(rotation=0, fontsize=25)
    plt.yticks(rotation=0, fontsize=25)
    plt.xlabel('User Groups', fontsize=25)
    plt.ylabel('User Groups', fontsize=25)
    plt.title(dname.capitalize(), fontsize=36)
    ax.set_facecolor("white")
    viz_plot.get_figure().savefig(opath, format='pdf')
    plt.close()


def word_analysis(dname, inpath, odir, topn=1000):
    """Extract top 1000 features in each group, compare their jacard similarity
    """
    if not os.path.exists(odir + '/{}_word_results.json'.format(dname)):
        group_file = json.load(open(inpath))
        word_results = dict()
        top_feats = dict()
        all_keys = list(sorted(list(group_file.keys())))  # rank by alphabet

        for gkey in group_file:
            gkey_idx = all_keys.index(gkey)
            x_train, y_train = zip(*group_file[gkey]['train'])

            # check if the vectorizer exists
            vect_path = odir + '/{0}_vect_word_{1}.pkl'.format(dname, gkey_idx)
            if os.path.exists(vect_path):
                vect = pickle.load(open(vect_path, 'rb'))
            else:
                vect = TfidfVectorizer(ngram_range=(1, 1), max_features=10000, min_df=2)
                vect.fit(x_train)
                pickle.dump(vect, open(vect_path, 'wb'))

            scores = mutual_info_classif(vect.transform(x_train), y_train)
        
            # rank and extract features
            top_indices = list(np.argsort(scores)[::-1][:topn])
            feas = vect.get_feature_names()
            top_feats[gkey_idx] = set([feas[idx] for idx in top_indices])

        for key in top_feats:
            if key not in word_results:
                word_results[key] = dict()

            for key1 in top_feats:
                if key1 == key:
                    word_results[key][key1] = 1.0
                else:
                    word_results[key][key1] = len(
                        top_feats[key].intersection(top_feats[key1]))/topn

        # save the classification results
        json.dump(word_results, open(odir + '/{}_word_results.json'.format(dname), 'w'))

    return odir + '/{}_word_results.json'.format(dname)



def context_analysis(dname, inpath, data_dir, odir):
    """combine local (build vectorizer by sample data) with global (biuld vectorizer by global data) features,
        test if the combination can improve the model performance using logistic regression
    """
    x_train = []; y_train = []
    x_test = []; y_test = []
    results = {'global': 0.0, 'local': 0.0}

    # load train dataset
    with open(inpath) as dfile:
        cols = dfile.readline().strip().split('\t')
        text_idx = cols.index('text')
        label_idx = cols.index('label')

        for line in dfile:
            line = line.strip()
            if len(line) < 5:
                continue

            line = line.split('\t')
            x_train.append(line[text_idx])
            y_train.append(int(line[label_idx]))

    # load sample vectorizer
    vect_local_path = odir + '/{}_vect_sample.pkl'.format(dname)
    if os.path.exists(vect_local_path):
        vect_sample = pickle.load(open(vect_local_path, 'rb'))
    else:
        vect_sample = TfidfVectorizer(ngram_range=(1, 3), max_features=10000, min_df=2)
        vect_sample.fit(x_train)
        pickle.dump(vect_sample, open(vect_local_path, 'wb'))

    # load global vectorizer
    vect_global_path = odir + '/{}_vect_global.pkl'.format(dname)
    if os.path.exists(vect_global_path):
        vect_global = pickle.load(open(vect_global_path, 'rb'))
    else:
        x_global = []
        vect_global = TfidfVectorizer(
            ngram_range=(1, 3), max_features=10000, min_df=2,
            stop_words=list(vect_sample.vocabulary_.keys()),
        )
        with open(os.path.join(data_dir, dname) +'/{}.tsv'.format(dname)) as dfile:
            cols = dfile.readline().strip().split('\t')
            text_idx = cols.index('text')

            for line in dfile:
                line = line.strip()
                if len(line) < 5:
                    continue

                line = line.split('\t')
                x_global.append(line[text_idx])

        vect_global.fit(x_global)
        x_global = []
        pickle.dump(vect_global, open(vect_global_path, 'wb'))

    # train classifiers
    clf_local_path = odir + '/{}_clf_local.pkl'.format(dname)
    if os.path.exists(clf_local_path):
        clf_local = pickle.load(open(clf_local_path, 'rb'))
    else:
        clf_local = LogisticRegression(class_weight='balanced', multi_class='auto', max_iter=2000)
        clf_local.fit(vect_sample.transform(x_train), y_train)
        pickle.dump(clf_local, open(clf_local_path, 'wb'))

    clf_global_path = odir + '/{}_clf_global.pkl'.format(dname)
    if os.path.exists(clf_global_path):
        clf_global = pickle.load(open(clf_global_path, 'rb'))
    else:
        clf_global = LogisticRegression(class_weight='balanced', multi_class='auto', max_iter=2000)
        clf_global.fit(
            hstack([vect_global.transform(x_train), vect_sample.transform(x_train)]), 
            y_train
        )
        pickle.dump(clf_global, open(clf_global_path, 'wb'))

    # release memory
    x_train = None
    del x_train
    y_train = None
    del y_train

    # load test set
    with open(test_path) as dfile:
        cols = dfile.readline().strip().split('\t')
        text_idx = cols.index('text')
        label_idx = cols.index('label')

        for line in dfile:
            line = line.strip()
            if len(line) < 5:
                continue

            line = line.split('\t')
            x_test.append(line[text_idx])
            y_test.append(int(line[label_idx]))

    # test classifiers
    y_pred = clf_local.predict(vect_sample.transform(x_test))
    results['local'] = round(f1_score(y_pred=y_pred, y_true=y_test, average='weighted'), 4) * 100
    y_pred = clf_global.predict(
        hstack([vect_global.transform(x_test), vect_sample.transform(x_test)])
    )
    results['global'] = round(f1_score(y_pred=y_pred, y_true=y_test, average='weighted'), 4) * 100
    return results


def viz_ctt_results(dpath, opath):
    df = pd.read_csv(dpath, sep='\t')

    a4_dims = (12.27, 12.27)
    fig, ax = plt.subplots(figsize=a4_dims)

    viz_plot = sns.barplot(
        x='Data', y='F1', data=df,
        ax=ax, hue="mode"
    )

    plt.xticks(rotation=0, fontsize=20)
    plt.yticks(rotation=0, fontsize=20)
    plt.title('Performance Comparison', fontsize=28)
    ax.set_facecolor("white")

    # add annotation to each bar
    for p in viz_plot.patches:
        viz_plot.annotate(
            format(p.get_height(), '.2f'),
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='center',
            xytext = (0, 9), fontsize=20, 
            textcoords = 'offset points'
        )

    viz_plot.set_ylabel('F1', fontsize=20)
    viz_plot.set_xlabel('Data', fontsize=20)
    plt.setp(ax.get_legend().get_texts(), fontsize=22)
    plt.setp(ax.get_legend().get_title(), fontsize=22)
    viz_plot.set(ylim=(min(df.F1) *.95, max(df.F1) *1.05))
    viz_plot.figure.savefig(opath, format='pdf')
    plt.close()


def history_analysis(dname):
    """This script read 20%, 40%, 60%, 80%, 100% user historical posts,
        and test on the validatation set. 
        Test if how much the historical posts can help classifier understands the current user posts
    """
    pass


if __name__ == '__main__':
    dlist = ['amazon_health', 'imdb', 'yelp']
    ugroup_path = './ugroup.json'  # a file to save grouped users
    
    data_dir = '../data/raw/'
    root_dir = './user_outputs/'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    ctt_path = root_dir + '/ctt_results.tsv'
    ctt_results = dict()
    if os.path.exists(ctt_path):
        ctt_flag = False
    else:
        ctt_flag = True

    for name in dlist:
        print('Working on...: ', name)
        odir = os.path.join(root_dir, name)
        if not os.path.exists(odir):
            os.mkdir(odir)

        indir = os.path.join(data_dir, name)
        group_path, sample_path = build_user_data(name, indir, odir)
        print(group_path, sample_path)

        # group classification analysis
        # df_path = clf_analysis(name, group_path, odir)
        # viz_clf(df_path, name, odir + '/{}_group.pdf'.format(name))

        # word feature overlap analysis
        # df_path = word_analysis(name, group_path, odir, topn=1000)
        # viz_clf(df_path, name, odir + '/{}_word.pdf'.format(name))

        # context analysis
        test_path = os.path.join(data_dir, name) + '/valid.tsv'
        if ctt_flag:
            ctt_results[name] = context_analysis(name, sample_path, data_dir, odir)

    if ctt_flag:
        with open(ctt_path, 'w') as wfile:
            wfile.write('Data\tF1\tmode\n')
            for name in ctt_results:
                for mode in ctt_results[name]:
                    wfile.write(name + '\t' + str(ctt_results[name][mode]) + '\t' + mode + '\n')

    viz_ctt_results(ctt_path, opath=root_dir + '/ctt_results.pdf')
    

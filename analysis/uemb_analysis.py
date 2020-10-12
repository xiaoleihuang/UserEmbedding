import os
import sys
import json
import heapq

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np


def load_categories(dpath):
    '''Path of product json (raw)
    '''
    prod_dict = dict()
    user_dict = dict()

    with open(dpath) as dfile:
        for line in dfile:
            entity = json.loads(line)

            if entity['bid'] not in prod_dict:
                prod_dict[entity['bid']] = dict()
            prod_dict[entity['bid']]['genre'] = entity['genre']
            prod_dict[entity['bid']]['uids'] = entity['uids']

            for uid in entity['uids']:
                if uid not in user_dict:
                    user_dict[uid] = dict()
                    user_dict[uid]['bids'] = dict()
                    user_dict[uid]['genres'] = dict()

                for genre in entity['genre']:
                    if genre not in user_dict[uid]['genres']:
                        user_dict[uid]['genres'][genre] = 0
                    user_dict[uid]['genres'][genre] += 1
                
                if entity['bid'] not in user_dict[uid]['bids']:
                    user_dict[uid]['bids'][entity['bid']] = 0
                user_dict[uid]['bids'][entity['bid']] += 1

    genre_dict = dict()

    for uid in list(user_dict.keys()):
        mx_val = max(user_dict[uid]['genres'].values())
#        if mx_val > .8 * sum(user_dict[uid]['genres'].values()):
#            del user_dict[uid]
#            continue

        for genre in user_dict[uid]['genres']:
            if user_dict[uid]['genres'][genre] == mx_val:
                user_dict[uid]['genre'] = genre

        if user_dict[uid]['genre'] not in genre_dict:
            genre_dict[user_dict[uid]['genre']] = list()
        heapq.heappush(genre_dict[user_dict[uid]['genre']], (mx_val, uid))

    min_val = min([len(genre_dict[item]) for item in genre_dict])
    ulist = []
    for genre in genre_dict:
        genre_dict[genre] = heapq.nlargest(min_val, genre_dict[genre])
        ulist.extend([item[1] for item in genre_dict[genre]])
    ulist = set(ulist)

    return prod_dict, user_dict


def tsne_user(uemb_path, udict, opath):
    uids = list()
    uembs = list()
    with open(uemb_path) as ufile:
        for line in ufile:
            line = line.strip()
            if len(line) < 5:
                continue

            line = line.split('\t')
            uids.append(line[0].strip())
            uembs.append([float(item) for item in line[1].split()])
    tsne = TSNE(n_components=2, n_jobs=-1)
    uembs = tsne.fit_transform(np.asarray(uembs))

    with open(opath, 'w') as wfile:
        wfile.write('uid\tx\ty\tdomain\n')
        for idx, uid in enumerate(uids):
            if uid in udict:
                wfile.write(
                    '\t'.join(map(
                        str, 
                        [uid, uembs[idx][0], uembs[idx][1], udict[uid]['genre']])
                    ) + '\n'
                )


def viz_user(inpath, opath):
    df = pd.read_csv(inpath, sep='\t')
    a4_dims = (12.27, 12.27)
    fig, ax = plt.subplots(figsize=a4_dims)

    viz_plot = sns.scatterplot(data=df, x='x', y='y', hue='domain', ax=ax)
    viz_plot.set_ylabel('X', fontsize=20)
    viz_plot.set_xlabel('Y', fontsize=20)
    plt.setp(ax.get_legend().get_texts(), fontsize=22)
    plt.setp(ax.get_legend().get_title(), fontsize=22)
    viz_plot.figure.savefig(opath, format='pdf')
    plt.close()


if __name__ == '__main__':
    odir = './uemb_outputs/'
    mode = sys.argv[1] if len(sys.argv) > 1 else 'skipgrams'  # baselines or skipgrams
    method = sys.argv[2] if len(sys.argv) > 2 else 'word_user_product'  # doc2user...

    if not os.path.exists(odir):
        os.mkdir(odir)

    for dname in ['imdb']:  # 'amazon_health', 'imdb', 'yelp'
        udict_path = '../data/raw/' + dname + '/products.json'
        _, udict = load_categories(udict_path)

        uemb_path = '../resources/{0}/{1}/{2}/user.txt'.format(mode, dname, method)
        opath = odir + '/{0}_{1}_{2}_user.tsv'.format(mode, dname, method)

        if not os.path.exists(opath):
            tsne_user(uemb_path, udict, opath)

        viz_user(opath, odir + '{0}_{1}_{2}_user.pdf'.format(mode, dname, method))

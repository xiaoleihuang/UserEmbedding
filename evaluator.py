'''Evaluate effectiveness of trained embeddings
'''
import os
import sys
import json

import numpy as np
from sklearn import metrics
from sklearn.cluster import SpectralClustering
from scipy.stats import entropy
from sklearn.preprocessing import MinMaxScaler


def get_labels(label2pairs):
    '''Generate true and predicted labels from the clustered words.
    '''
    y_true = []
    y_pred = []

    for idx in range(len(label2pairs)):
        for jdx in range(idx+1, len(label2pairs)):
            # the logic is to check within the same cluster, the section label should be the same
            # if not in the same section, their labels should be different
            if label2pairs[idx][-1] == label2pairs[jdx][-1]:
                y_true.append(1)
            else:
                y_true.append(0)

            # compare the section label,
            # if their section labels are the same, they should be in the same section
            # otherwise, they should be in the different sections
            if label2pairs[idx][1] == label2pairs[jdx][1]:
                y_pred.append(1)
            else:
                y_pred.append(0)
    return y_true, y_pred


def f1_beta(y_true, y_pred, beta=5):
    '''Implement the evaluation method discussed in the paper ""
    '''
    scores = dict()
    types = ['binary', 'micro', 'macro', 'weighted']
    for mtype in types:
        scores[mtype] = metrics.fbeta_score(
            y_true=y_true, y_pred=y_pred, 
            beta=beta, average=mtype
        )
        print(scores[mtype])
    print(scores)
    return scores


def mutual_info_1(label2pairs):
    '''normalized mutual information'''
    score = 0.0
    mi_scores = dict()
    cluster_label_counts = dict()
    section_label_counts = dict()

    # calculate the mutual information for cluster
    # then sum the values
    for pair in label2pairs:
        if pair[-1] not in cluster_label_counts:
            cluster_label_counts[pair[-1]] = 0
        cluster_label_counts[pair[-1]] += 1
        if pair[1] not in section_label_counts:
            section_label_counts[pair[1]] = 0
        section_label_counts[pair[1]] += 1

        if pair[-1] not in mi_scores:
            mi_scores[pair[-1]] = dict()
        
        if pair[1] not in mi_scores[pair[-1]]:
            mi_scores[pair[-1]][pair[1]] = 0

        mi_scores[pair[-1]][pair[1]] += 1
        
    # calculate the entropy of cluster H(C)
    count_sum_cluster = sum(cluster_label_counts.values())
    cluster_label_probs = [item/count_sum_cluster for item in cluster_label_counts.values()]
    h_c = entropy(cluster_label_probs)
    
    # calculate the entropy of labels H(L)
    count_sum_section = sum(section_label_counts.values())
    section_label_probs = [item/count_sum_section for item in section_label_counts.values()]
    h_l = entropy(section_label_probs)

    # calculate the I_l_c = H(L) - H(L|C)
    h_l_c = dict()
    for cluster_l in mi_scores:
        mi_sum = sum(mi_scores[cluster_l].values())
        h_l_c[cluster_l] = cluster_label_counts[cluster_l]/count_sum_cluster * entropy(
            [item/mi_sum for item in mi_scores[cluster_l].values()]
        )
    I_l_c = h_l - sum(h_l_c.values())
#    I_l_c = sum(h_l_c.values())

    # calculate the score according to the equation (11)
    # 2*I(L; C)/[H(L) + H(C)]
    score = I_l_c * 2 / (h_c/2 + h_l/2)

    return score


def mutual_info(label2pairs):
    clusters = dict()
    cluster_label_counts = dict()
    section_label_counts = dict()

    # calculate the mutual information for cluster
    # then sum the values
    for pair in label2pairs:
        if pair[-1] not in cluster_label_counts:
            cluster_label_counts[pair[-1]] = 0
        cluster_label_counts[pair[-1]] += 1
        if pair[1] not in section_label_counts:
            section_label_counts[pair[1]] = 0
        section_label_counts[pair[1]] += 1

        if pair[-1] not in clusters:
            clusters[pair[-1]] = []
        clusters[pair[-1]].append(pair)

    mi_vals = 0.0
    for cluster_key in clusters:
        y_true, y_pred = get_labels(clusters[cluster_key])
        mi_vals += metrics.normalized_mutual_info_score(y_true, y_pred, average_method='arithmetic')
        
    # calculate the entropy of cluster H(C)
    count_sum_cluster = sum(cluster_label_counts.values())
    cluster_label_probs = [item/count_sum_cluster for item in cluster_label_counts.values()]
    h_c = entropy(cluster_label_probs)
    
    # calculate the entropy of labels H(L)
    count_sum_section = sum(section_label_counts.values())
    section_label_probs = [item/count_sum_section for item in section_label_counts.values()]
    h_l = entropy(section_label_probs)

    score = mi_vals * 2 / (h_c + h_l)
    return score


def load_emb(dpath):
    '''load user or product embedding

    Parameters:
    -----------
    dpath: str
        A tsv file path of user or product embedding

    Return:
    
    '''
    ids_idx = dict()
    embs = list()
    with open(dpath) as dfile:
        for idx, line in enumerate(dfile):
            eid, vectors = line.strip().split('\t')
            vectors = np.asarray([float(item) for item in vectors.strip().split()])
            if np.isnan(vectors).any() or np.isinf(vectors).any():
                print(eid)
            ids_idx[idx] = eid
            embs.append(vectors)

    embs = np.asarray(embs)
    idx_ids = dict(zip(ids_idx.keys(), ids_idx.values()))
    print(embs.shape)

    return ids_idx, idx_ids, embs


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

    return prod_dict, user_dict


'''Evaluation Task 1: Product Embedding Evaluator. Evaluate the semantic similarity of products which share the same category labels.

Cluster mode -> cluster, randomly sample, the products which are in the same categories should be clustered together.

Prediction mode -> use the embedding for multi label prediction.

'''
def eval_product_cluster(emb_pairs, genre_dict, cluster_num=10, opt=None):
    '''Cluster the products using spectural clustering
    '''
    ids_idx, idx_ids, embs = emb_pairs
    scluster = SpectralClustering(
        # the original paper use the cosine
        # default is rbf
        affinity='cosine',#'cosine', 'rbf', 
        n_clusters=cluster_num,
        # assign_labels=discretize,
        # solver='lobpcg',
        n_jobs=-1
    )
    
    # cluster
    embs = np.array(embs)
    if scluster.affinity == 'cosine':
        scaler = MinMaxScaler()
        embs = scaler.fit_transform(embs)

    scluster.fit(embs)
    idx2labels = scluster.labels_ # a list

    eval_pairs = [[], []] # cluster vs. real label
    for idx in range(len(idx2labels)):
        # sample every two labels, check if they are the same categories.
        for jdx in range(idx+1, len(idx2labels)):
            # cluster labels
            if idx2labels[idx] == idx2labels[jdx]:
                eval_pairs[0].append(1)
            else:
                eval_pairs[0].append(0)

            # category labels
            if genre_dict[idx_ids[idx]]['genre'] == genre_dict[idx_ids[jdx]]['genre']:
                eval_pairs[1].append(1)
            else:
                if bool(set(genre_dict[idx_ids[idx]]['genre']) & set(genre_dict[idx_ids[jdx]]['genre'])):
                    eval_pairs[1].append(eval_pairs[0][-1])
                else:
                    eval_pairs[1].append(0)

    accuracy = metrics.accuracy_score(y_true=eval_pairs[0], y_pred=eval_pairs[1])
    f1 = metrics.f1_score(y_true=eval_pairs[0], y_pred=eval_pairs[1], average='weighted')
    print('Accuracy: ', accuracy)
    print('F1-score: ', f1)


def eval_product_predict():# TODO
    '''Multilabel prediction task --> product category
    '''
    pass



'''Evaluation Task 2: User Embedding Evaluator, Evaluate by the User Who share the similar purchasing behaviors. 

Cluster users -> users who purchase the items within the same product category should be clustered in the same group -> sample every two users, correct if they are in the same cluster and purchased the same products or they are not in the same cluster and purchsed not the same product category; 0 is the opposite creteria -> F1 measurement

Predict users -> User interests prediction

'''
def eval_user_cluster(emb_pairs, user_dict, cluster_num=10, opt=None):
    '''Cluster users by spectral clustering
    '''
    ids_idx, idx_ids, embs = emb_pairs
    scluster = SpectralClustering(
        # the original paper use the cosine
        # default is rbf
        affinity='cosine',#'cosine', 'rbf', 
        n_clusters=cluster_num,
        # assign_labels=discretize,
        # solver='lobpcg',
        n_jobs=-1
    )

    # cluster
    embs = np.array(embs)
    if scluster.affinity == 'cosine':
        scaler = MinMaxScaler()
        embs = scaler.fit_transform(embs)

    scluster.fit(embs)
    idx2labels = scluster.labels_ # a list

    eval_pairs = [[], []] # cluster vs. real label
    for idx in range(len(idx2labels)):
        # sample every two labels, check if they are the same categories.
        for jdx in range(idx+1, len(idx2labels)):
            # cluster labels
            if idx2labels[idx] == idx2labels[jdx]:
                eval_pairs[0].append(1)
            else:
                eval_pairs[0].append(0)

            # category labels
            idx_genres = list(user_dict[idx_ids[idx]]['genres'].keys())
            jdx_genres = list(user_dict[idx_ids[jdx]]['genres'].keys())
            if idx_genres == jdx_genres:
                eval_pairs[1].append(1)
            else:
                if bool(set(idx_genres) & set(jdx_genres)):
                    eval_pairs[1].append(eval_pairs[0][-1])
                else:
                    eval_pairs[1].append(0)

    accuracy = metrics.accuracy_score(y_true=eval_pairs[0], y_pred=eval_pairs[1])
    f1 = metrics.f1_score(y_true=eval_pairs[0], y_pred=eval_pairs[1], average='weighted')
    print('Accuracy: ', accuracy)
    print('F1-score: ', f1)


def eval_user_predict(): # TODO
    pass


if __name__ == '__main__':
    raw_dir = './data/raw/'
    resource_dir = './resources/'
    baseline_dir = resource_dir + 'baselines/'

    dname = sys.argv[1] # 'yelp', 'amazon', 'imdb'
    mode = sys.argv[2] # 'word2user', 'doc2user', 'lda2user', 'bert2user', 'skipgrams'
    cluster_num = int(sys.argv[3]) # 4, 8, 12

    print('System Arguments: ', ', '.join(sys.argv))

    if dname in ['yelp', 'amazon', 'imdb']:
        print('Data Name: ', dname.upper())
        prod_json_path = raw_dir + dname + '/products.json'
        prod_dict, user_dict = load_categories(prod_json_path)


        if mode == 'skipgrams':
            my_dir = resource_dir + mode + '/'
            task = 'word_user_product' # word_user, word_user_product
            my_emb_dir = my_dir + dname + '/' + task + '/'

            # evaluate product embeddings by separation
            print('-----------------------{My model}-------------------')
            print('Product Evaluation -------- Cluster')
            prod_emb_pair = load_emb(my_emb_dir + 'product.txt')
            eval_product_cluster(prod_emb_pair, prod_dict, cluster_num=cluster_num, opt=None)

            # evaluate user embeddings by MAP@K, for task 1 (word_user_product)
            print('User Evaluation -------- Cluster: ', task)
            user_emb_pair = load_emb(my_emb_dir + 'user.txt')
            eval_user_cluster(user_emb_pair, user_dict, cluster_num=cluster_num, opt=None)

            # evaluate user embeddings by MAP@K, for task 2 (word_user)
            task = 'word_user'
            my_emb_dir = my_dir + dname + '/' + task + '/'

            print('User Evaluation -------- Cluster: ', task)
            user_emb_pair = load_emb(my_emb_dir + 'user.txt')
            eval_user_cluster(user_emb_pair, user_dict, cluster_num=cluster_num, opt=None)

        else:
            dname_dir = baseline_dir + dname + '/'
            print('-----------------------{}-------------------'.format(mode))
            method_dir = dname_dir + mode + '/'

            print('Product Evaluation -------- Cluster')
            prod_emb_pair = load_emb(method_dir + 'product.txt')
            eval_product_cluster(prod_emb_pair, prod_dict, cluster_num=cluster_num, opt=None)

            print('User Evaluation -------- Cluster')
            user_emb_pair = load_emb(method_dir + 'user.txt')
            eval_user_cluster(user_emb_pair, user_dict, cluster_num=cluster_num, opt=None)

            print('------------------------------------------------------------\n\n')


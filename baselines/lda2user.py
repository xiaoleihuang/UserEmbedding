'''Extract topics from each document, average all topic vectors as user representations
This script is to implement methods from Multi-View Unsupervised User Feature Embedding for Social
Media-based Substance Use Prediction

PostLDA-Doc
'''
import gensim
from gensim.models import LdaModel
import pickle
import numpy as np
import sys
import os


class Lda2User(object):
    '''Apply LDA model on the documents to generate user and product representation.
        Outputs will be one user/product_id + vect per line.

        Parameters
        ----------
        task: str
            Task name, such as amazon, yelp and imdb
        dict_path: str
            Path of LDA dictionary file
        model_path: str
            Path of LDA model file
    '''
    def __init__(self, task, dict_path, model_path):
        self.task = task
        self.dictionary = self.__load_dict(dict_path)
        self.model = self.__load_model(model_path)

    def __load_dict(self, dict_path):
        return pickle.load(open(dict_path, 'rb'))

    def __load_model(self, model_path):
        return LdaModel.load(model_path)

    def lda2item(self, data_path, opath, id_idx=2, mode='average'):
        '''Extract user vectors from the given data path

            Parameters
            ----------
            data_path: str
                Path of data file, tsv file
            opath: str
                Path of output path for user vectors
            id_idx: int
                Index of id, 2 is for user, 3 is for product
        '''
        item_dict = dict()
        ofile = open(opath, 'w')

        with open(data_path) as dfile:
            dfile.readline() # skip the column names

            for line in dfile:
                line = line.strip().split('\t')

                tid = line[id_idx]
                text = line[3]
                if tid not in item_dict:
                    item_dict[tid] = []

                # collect data
                if mode == 'average':
                    item_dict[tid].append(text.split())
                else:
                    if len(item_dict[tid]) == 0:
                        item_dict[tid].append(text.split())
                    else:
                        item_dict[tid][0].extend(text.split())

        for tid in item_dict:
            # encode the document by lda
            item_dict[tid] = np.asarray([
                self.model[self.dictionary.doc2bow(doc)] for doc in item_dict[tid]
            ])
            # average the lda inferred documents
            item_dict[tid] = np.mean(item_dict[tid], axis=0)

            # write to file
            ofile.write(tid + '\t' + ' '.join(map(str, item_dict[tid])))
        ofile.flush()
        ofile.close()


if __name__ == '__main__':
    task = sys.argv[1]
    raw_dir = '../data/raw/'
    task_data_path = raw_dir + task + '/' + task + '.tsv'

    data_dir = raw_dir + task + '/'
    baseline_dir = '../resources/baselines/'
    task_dir = baseline_dir + task + '/'
    odir = task_dir + 'lda2user/'
    opath_user = odir + 'user.txt'
    opath_product = odir + 'product.txt'

    resource_dir = '../resources/embedding/'
    dict_path = resource_dir + task + '/lda_dict.pkl'
    model_path = resource_dir + task + '/lda.model'

    # create directories
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    if not os.path.exists(odir):
        os.mkdir(odir)

    # Lda2User
    l2u = Lda2User(task, dict_path, model_path)
    # user vectors
    l2u.lda2item(
        data_path=task_data_path, 
        opath=opath_user, 
        id_idx=2
    )
    # product vectors
    l2u.lda2item(
        data_path=task_data_path, 
        opath=opath_product, 
        id_idx=3
    )



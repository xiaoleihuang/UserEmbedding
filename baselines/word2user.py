'''Extract document representations by average word representations,
average all document vectors as user or product representations
'''
import os
import gensim
import numpy as np


class Word2User(object):
    '''Apply Word2Vec model on the documents to generate user and product representation.
        Outputs will be one user/product_id + vect per line.

        Parameters
        ----------
        task: str
            Task name, such as amazon, yelp and imdb
        tkn_path: str
            Path of LDA dictionary file
        model_path: str
            Path of LDA model file
    '''
    def __init__(self, task, tkn_path, model_path, emb_dim=300):
        self.task = task
        self.tkn = self.__load_tkn(tkn_path)
        self.emb_dim = emb_dim
        self.model = self.__load_model(model_path, emb_dim)

    def __load_tkn(self, tkn_path):
        return pickle.load(open(tkn_path, 'rb'))

    def __load_model(self, model_path, emb_dim=300):
        # support three types, bin/txt/npy
        emb_len = len(self.tkn.word_index)
        if embed_len > self.tkn.num_words:
            embed_len = self.tkn.num_words
        model = np.zeros((embed_len + 1, emb_dim))

        if model_path.endswith('.bin'):
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                model_path, binary=True
            )
            for pair in zip(w2v_model.wv.index2word, w2v_model.wv.syn0):
                if pair[0] in self.tkn.word_index and \
                    self.tkn.word_index[pair[0]] < self.tkn.num_words:
                    model[self.tkn.word_index[pair[0]]] = pair[1]

        elif model_path.endswith('.npy'):
            model = np.load(model_path)

        elif model_path.endswith('.txt'):
            with open(model_path) as dfile:
                for line in dfile:
                    line = line.strip().split()
                    word = line[0]
                    vectors = np.asarray(line[1:], dtype='float32')

                    if word in self.tkn.word_index and \
                    self.tkn.word_index[word] < self.tkn.num_words:
                        model[self.tkn.word_index[word]] = vectors

        else:
            raise ValueError('Current other formats are not supported!')
        return model

    def word2item(self, data_path, opath, id_idx=2, mode='average'):
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
                item_dict[tid].extend(text.split())

        for tid in item_dict:
            # encode the document by word2vec
            item_dict[tid] = np.asarray([
                self.model[self.tkn.word_index[word]] for word in item_dict[tid]
            ])
            # average the word2vec inferred documents
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
    odir = baseline_dir + 'word2user/'
    opath_user = odir + 'user.txt'
    opath_product = odir + 'product.txt'

    resource_dir = '../resources/embedding/'
    tkn_path = '../data/encode/' + task + '/' + task + '.tkn'
    model_path = resource_dir + task + '/w2v.txt'

    # create directories
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)
    if not os.path.exists(odir):
        os.mkdir(odir)

    # Word2User
    l2u = Word2User(task, tkn_path, model_path)
    # user vectors
    l2u.word2item(
        data_path=task_data_path, 
        opath=opath_user, 
        id_idx=2
    )
    # product vectors
    l2u.word2item(
        data_path=task_data_path, 
        opath=opath_product, 
        id_idx=3
    )

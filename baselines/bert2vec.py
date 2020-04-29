'''Average BERT outputs as user representations.
'''

import os
import sys

import numpy as np
import torch
torch.set_num_threads(15)
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import BertForSequenceClassification


class Bert2User(object):
    '''Apply Doc2Vec model on the documents to generate user and product representation.
        Outputs will be one user/product_id + vect per line.

        Parameters
        ----------
        task: str
            Task name, such as amazon, yelp and imdb
        model_dir: str
            Directory path of Bert model file
    '''
    def __init__(self, task, model_dir):
        self.task = task
        self.model = self.__load_model(model_dir)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def __load_model(self, model_dir):
        if os.path.exists(model_dir):
            model = BertForSequenceClassification.from_pretrained(
                model_dir, output_hidden_states=True
            )
        else:
            model = BertForSequenceClassification.from_pretrained(
                'bert-base-uncased', output_hidden_states=True
            )
        model.to('cpu')
        return model

    def bert2item(self, data_path, opath, id_idx=2, mode='average'):
        '''Extract user vectors from the given data path

            Parameters
            ----------
            data_path: str
                Path of data file, tsv file
            opath: str
                Path of output path for user vectors
            id_idx: int
                Index of id, 2 is for user, 1 is for product
        '''
        item_dict = dict()
        ofile = open(opath, 'w')

        print('Loading Data...')
        with open(data_path) as dfile:
            dfile.readline() # skip the column names

            for line in dfile:
                line = line.strip()
                if len(line) < 5:
                    continue
                line = line.split('\t')

                tid = line[id_idx].strip()
                text = line[3].strip()

                if len(tid) == 0 or len(text) == 0:
                    continue

                if tid not in item_dict:
                    item_dict[tid] = []

                # collect data
                if mode == 'average':
                    try:
                        item_dict[tid].append(self.tokenizer.encode_plus(
                            text,
                            add_special_tokens=True,
                            max_length=512,
                            return_tensors='pt'
                        ))
                    except:
                        continue
                else:
                    if len(item_dict[tid]) == 0:
                        item_dict[tid].append(text)
                    else:
                        item_dict[tid][0] += text

        for tid in list(item_dict.keys()):
            print('Working on item: ', tid)
            # preprocess the document
            if len(item_dict[tid]) == 1 and mode != 'average':
                item_dict[tid][0] = self.tokenizer.encode_plus(
                    item_dict[tid][0],
                    add_special_tokens=True,
                    max_length=512,
                    return_tensors='pt'
                )

            # encode the document by bert, get the last Bert layer's outputs
            item_dict[tid] = np.asarray([
                self.model(**doc)[1][-1][:,0,:].detach().numpy() for doc in item_dict[tid]
            ])
            # average the lda inferred documents
            item_dict[tid] = np.squeeze(np.mean(item_dict[tid], axis=0))

            # write to file
            ofile.write(tid + '\t' + ' '.join(map(str, item_dict[tid])) + '\n')

            # save memory
            del item_dict[tid]
        ofile.flush()
        ofile.close()


if __name__ == '__main__':
    task = sys.argv[1]
    raw_dir = '../data/raw/'
    task_data_path = raw_dir + task + '/' + task + '.tsv'

    data_dir = raw_dir + task + '/'
    baseline_dir = '../resources/baselines/'
    task_dir = baseline_dir + task + '/'
    odir = task_dir + 'bert2user/'
    opath_user = odir + 'user.txt'
    opath_product = odir + 'product.txt'

    resource_dir = '../resources/embedding/'
    model_dir = resource_dir + task + '/bert/'

    # create directories
    if not os.path.exists(baseline_dir):
        os.mkdir(baseline_dir)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    if not os.path.exists(odir):
        os.mkdir(odir)

    # Bert2User
    d2u = Bert2User(task, model_dir)
    # user vectors
    d2u.bert2item(
        data_path=task_data_path, 
        opath=opath_user, 
        id_idx=2
    )
    # product vectors
    d2u.bert2item(
        data_path=task_data_path, 
        opath=opath_product, 
        id_idx=1
    )

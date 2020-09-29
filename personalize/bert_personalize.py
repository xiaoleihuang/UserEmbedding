import os
from abc import ABC
from collections import Counter
import json

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, classification_report
from tqdm import trange
import pandas as pd

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as torch_func

from transformers import BertTokenizer, BertModel, BertConfig, BertPreTrainedModel
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
torch.set_num_threads(os.cpu_count())


def get_freer_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.remove('tmp')
    return np.argmax(memory_available)


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def flat_f1(preds, labels):
    macro_score = f1_score(
        y_true=labels, y_pred=preds,
        average='macro',
    )
    weighted_score = f1_score(
        y_true=labels, y_pred=preds,
        average='weighted',
    )
    print('Weighted F1-score: ', weighted_score)
    print('Macro F1-score: ', macro_score)
    return macro_score, weighted_score


class PersonalizeBert4SeqClassification(BertPreTrainedModel, ABC):
    """The code is adopted from BertForSequenceClassification,
        https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification
    """
    def __init__(self, params, bert_config):
        """
        Here we are taking simple concatenation method to integrate
            user, product and bert encoded representations together.
        :param params:
        """
        super().__init__(bert_config)
        self.num_labels = params['num_label']
        bert_hidden_size = 768

        uemb_path = os.path.join(params['up_dir'], 'user.npy')
        if os.path.exists(uemb_path):
            uemb = np.load(uemb_path)
            bert_hidden_size += uemb.shape[-1]
            self.uemb = nn.Embedding.from_pretrained(torch.FloatTensor(uemb))
            self.uemb.weight.requires_grad = True
        else:
            self.uemb = None

        pemb_path = os.path.join(params['up_dir'], 'product.npy')
        if os.path.exists(pemb_path):
            pemb = np.load(pemb_path)
            bert_hidden_size += pemb.shape[1]
            self.pemb = nn.Embedding.from_pretrained(torch.FloatTensor(pemb))
            self.pemb.weight.requires_grad = True
        else:
            self.pemb = None

        self.bert = BertModel.from_pretrained('bert-base-uncased', config=bert_config)
        self.dropout = nn.Dropout(params['dp_rate'])
        self.classifier = nn.Linear(bert_hidden_size, params['num_label'])

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        input_uids=None,
        input_pids=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        if input_uids is not None and self.uemb is not None:
            uemb = self.uemb(input_uids)
            pooled_output = torch.cat((pooled_output, uemb), dim=-1)
        if input_pids is not None and self.pemb is not None:
            pemb = self.pemb(input_pids)
            pooled_output = torch.cat((pooled_output, pemb), dim=-1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def run_bert(params):
    """Google Bert Classifier
    """
    print(params)
    # suffix of output files
    output_suffix = '_'

    if torch.cuda.is_available():
        device = int(get_freer_gpu())
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    print('Loading Datasets and oversample training data...')
    train_df = pd.read_csv(params['data_dir'] + 'train.tsv', sep='\t', na_values='x')

    # oversample the minority class
    if params['balance']:
        label_count = Counter(train_df.label)
        for label_tmp in label_count:
            sample_num = label_count.most_common(1)[0][1] - label_count[label_tmp]
            if sample_num == 0:
                continue
            train_df = pd.concat([train_df,
                                  train_df[train_df.label == label_tmp].sample(
                                      int(sample_num * params['balance_ratio']), replace=True
                                  )])
        train_df = train_df.reset_index()  # to prevent index key error

    valid_df = pd.read_csv(params['data_dir'] + 'valid.tsv', sep='\t', na_values='x')
    test_df = pd.read_csv(params['data_dir'] + 'test.tsv', sep='\t', na_values='x')
    data_df = [train_df, valid_df, test_df]
    # We need to add special tokens at the beginning and end of each sentence for BERT to work properly
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(lambda x: '[CLS] ' + x + ' [SEP]')

    print('Padding Datasets...')
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case=True
    )
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(lambda x: tokenizer.tokenize(x))

    # convert to indices and pad the sequences
    for doc_df in data_df:
        doc_df.text = doc_df.text.apply(
            lambda x: pad_sequences(
                [tokenizer.convert_tokens_to_ids(x)],
                maxlen=params['max_len'], dtype="long"
            )[0])

    # create attention masks
    for doc_df in data_df:
        attention_masks = []
        for seq in doc_df.text:
            seq_mask = [float(idx > 0) for idx in seq]
            attention_masks.append(seq_mask)
        doc_df['masks'] = attention_masks

    batch_size = params['batch_size']
    # format train, valid, test
    train_inputs = torch.tensor(data_df[0].text)
    train_labels = torch.tensor(data_df[0].label)
    train_masks = torch.tensor(data_df[0].masks)
    train_uids = torch.tensor(data_df[0].uid)
    train_pids = torch.tensor(data_df[0].bid)
    train_data = TensorDataset(
        train_inputs, train_masks, train_labels, train_uids, train_pids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, num_workers=1)

    valid_inputs = torch.tensor(data_df[1].text)
    valid_labels = torch.tensor(data_df[1].label)
    valid_masks = torch.tensor(data_df[1].masks)
    valid_uids = torch.tensor(data_df[1].uid)
    valid_pids = torch.tensor(data_df[1].bid)
    valid_data = TensorDataset(
        valid_inputs, valid_masks, valid_labels, valid_uids, valid_pids)
    valid_sampler = SequentialSampler(valid_data)

    test_inputs = torch.tensor(data_df[2].text)
    test_labels = torch.tensor(data_df[2].label)
    test_masks = torch.tensor(data_df[2].masks)
    test_uids = torch.tensor(data_df[2].uid)
    test_pids = torch.tensor(data_df[2].bid)
    test_data = TensorDataset(
        test_inputs, test_masks, test_labels, test_uids, test_pids)
    test_sampler = SequentialSampler(test_data)

    # load user and product embeddings
    uemb_path = os.path.join(params['up_dir'], 'user.npy')
    if os.path.exists(uemb_path):
        output_suffix += 'u'
    pemb_path = os.path.join(params['up_dir'], 'product.npy')
    if os.path.exists(pemb_path):
        print('Loading product embedding...: ', pemb_path)
        output_suffix += 'p'

    # load the pretrained model
    print('Loading Pretrained Model...')
    conf = BertConfig.from_pretrained('bert-base-uncased', num_labels=params['num_label'])
    model = PersonalizeBert4SeqClassification(params, conf)
    model.to(device)
    # only load bert to the GPU
    # model.bert.to(device=device)

    # organize parameters
    param_optimizer = list(model.named_parameters())
    if params['freeze']:
        no_decay = ['bias', 'bert']  # , 'bert' freeze all bert parameters
    else:
        no_decay = ['bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': params['decay_rate']},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=params['lr'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=params['warm_steps'],
        num_training_steps=params['train_steps']
    )
    wfile = open('./results/bert_personalize{}_results.txt'.format(output_suffix), 'a')
    wfile.write(params['data_name'] + '_________________\n')
    wfile.write(json.dumps(params) + '\n')
    wfile.write('\n')
    
    best_valid_f1 = 0.0

    # Training
    print('Training the model...')
    for epoch in trange(params['epochs'], desc='Epoch'):
        model.train()
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # train batch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_uids, b_pids = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(
                b_input_ids, token_type_ids=None,
                attention_mask=b_input_mask, labels=b_labels,
                input_uids=b_uids, input_pids=b_pids
            )
            # backward pass
            outputs[0].backward()
            # outputs.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()
            scheduler.step()

            # Update tracking variables
            tr_loss += outputs[0].item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        '''Validation'''
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        # tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # batch eval
        y_preds = []
        valid_dataloader = DataLoader(
            valid_data, sampler=valid_sampler, batch_size=batch_size)
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels, b_uids, b_pids = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                    input_uids=b_uids, input_pids=b_pids
                )
            # Move logits and labels to CPU
            logits = outputs[0].detach().cpu().numpy()
            # record the prediction
            pred_flat = np.argmax(logits, axis=1).flatten()
            y_preds.extend(pred_flat)

            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
        # evaluate the validation f1 score
        f1_m_valid, f1_w_valid = flat_f1(y_preds, valid_df.label)
        if f1_w_valid > best_valid_f1:
            print('Test....')
            best_valid_f1 = f1_w_valid
            print('Epoch {0}, valid f1 score {1}'.format(epoch, best_valid_f1))
            y_preds = []
            y_probs = []

            test_dataloader = DataLoader(
                test_data, sampler=test_sampler, batch_size=batch_size)
            # test if valid gets better results
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels, b_uids, b_pids = batch
                with torch.no_grad():
                    outputs = model(
                        b_input_ids, token_type_ids=None, attention_mask=b_input_mask,
                        input_uids=b_uids, input_pids=b_pids
                    )
                probs = torch_func.softmax(outputs[0], dim=1)
                probs = probs.detach().cpu().numpy()
                pred_flat = np.argmax(probs, axis=1).flatten()
                y_preds.extend(pred_flat)
                y_probs.extend([item[1] for item in probs])

            # save the predicted results
            wfile.write('Epoch: {}.........................\n'.format(epoch))
            wfile.write(str(f1_score(y_pred=y_preds, y_true=test_df.label, average='weighted')) + '\n')
            report = classification_report(y_pred=y_preds, y_true=test_df.label, digits=3)
            print(report)
            wfile.write(report + '\n')
            wfile.write('.........................\n')
            wfile.write('\n')

    # release cuda memory
    torch.cuda.empty_cache()


if __name__ == '__main__':
    # create directories for saving models and tokenizers
    if not os.path.exists('./vects/'):
        os.mkdir('./vects/')
    if not os.path.exists('./clfs/'):
        os.mkdir('./clfs/')
    if not os.path.exists('./results/'):
        os.mkdir('./results/')

    data_list = [
        # 'amazon_health',
        # 'imdb',
        'yelp',
    ]

    # config follows https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json
    parameters = dict()
    parameters['balance_ratio'] = 1
    parameters['freeze'] = True
    parameters['decay_rate'] = .001
    parameters['lr'] = 9e-5
    parameters['warm_steps'] = 100
    parameters['train_steps'] = 1000
    parameters['batch_size'] = 16
    parameters['balance'] = True
    parameters['num_label'] = 3
    parameters['epochs'] = 9
    parameters['dp_rate'] = 0.1

    # load data stats to determine the max length
    try:
        stats = json.load(open('../analysis/stats.json'))
    except FileNotFoundError:
        stats = None

    for dname in data_list:
        parameters['data_name'] = dname
        data_dir = '../data/raw/' + dname + '/'
        parameters['data_dir'] = data_dir
        parameters['max_len'] = int(stats[dname].get('75_percent_word_per_doc', 200))
        parameters['up_dir'] = '../resources/skipgrams/' + dname + '/word_user_product/'

        run_bert(parameters)

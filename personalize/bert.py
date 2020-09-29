import os
from collections import Counter
import json

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import f1_score, classification_report
from tqdm import trange
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as torch_func
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from transformers import AdamW, BertForSequenceClassification
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


def run_bert(params):
    """Google Bert Classifier
    """
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

    # format train, valid, test
    train_inputs = torch.tensor(data_df[0].text)
    train_labels = torch.tensor(data_df[0].label)
    train_masks = torch.tensor(data_df[0].masks)
    valid_inputs = torch.tensor(data_df[1].text)
    valid_labels = torch.tensor(data_df[1].label)
    valid_masks = torch.tensor(data_df[1].masks)
    test_inputs = torch.tensor(data_df[2].text)
    test_labels = torch.tensor(data_df[2].label)
    test_masks = torch.tensor(data_df[2].masks)

    batch_size = params['batch_size']

    train_data = TensorDataset(
        train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=batch_size, num_workers=os.cpu_count())
    valid_data = TensorDataset(
        valid_inputs, valid_masks, valid_labels)
    valid_sampler = SequentialSampler(valid_data)
    valid_dataloader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=batch_size)
    test_data = TensorDataset(
        test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(
        test_data, sampler=test_sampler, batch_size=batch_size)

    # load the pretrained model
    print('Loading Pretrained Model...')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=params['num_label'])
    model.to(device)

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
    wfile = open('./results/bert_results.txt', 'a')
    wfile.write(params['data_name'] + '_________________\n')

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
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            outputs = model(
                b_input_ids, token_type_ids=None,
                attention_mask=b_input_mask, labels=b_labels
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
        best_valid_f1 = 0.0
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()
        # tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # batch eval
        y_preds = []
        for batch in valid_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask)
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

            # test if valid gets better results
            for batch in test_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch
                with torch.no_grad():
                    outputs = model(
                        b_input_ids,
                        token_type_ids=None,
                        attention_mask=b_input_mask)
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

    parameters = dict()
    parameters['balance_ratio'] = 0.9
    parameters['freeze'] = False
    parameters['decay_rate'] = .001
    parameters['lr'] = 2e-5
    parameters['warm_steps'] = 100
    parameters['train_steps'] = 1000
    parameters['batch_size'] = 16
    parameters['balance'] = True
    parameters['num_label'] = 3
    parameters['epochs'] = 6

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

        run_bert(parameters)

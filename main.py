'''
Main script to train and test models
'''
import argparse
from collections import defaultdict
import logging
import time
import os
import pandas as pd
import csv
from upweight_dataset import upweight_dataset
import torch
import shutil
from typing import Dict, Iterable, List, Tuple
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from sklearn.metrics import classification_report
from allennlp.nn import util
import json
import numpy as np
import data
import models

def build_data_loader(data):
    data_loader = SimpleDataLoader(data, 8, shuffle=True)
    return data_loader

def build_trainer(model, serialization_dir, train_loader, dev_loader):
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer
def run_training_loop(modeldir, datadir, role, embed_dim=10, max_tokens=32, upsampled_data=False):
    dataset_reader = data.build_dataset_reader()

    train_data, dev_data = data.read_data(dataset_reader, datadir, role, upsampled_data=upsampled_data)
    vocab = data.build_vocab(train_data + dev_data)
    vocab.save_to_files(os.path.join(modeldir, 'vocab'))
    model = models.build_model(vocab, embed_dim=embed_dim, max_tokens=max_tokens)

    train_loader, dev_loader = build_data_loader(train_data), build_data_loader(dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)


    trainer = build_trainer(model, modeldir, train_loader, dev_loader)
    print("Starting training")
    trainer.train()
    print("Finished training")

    return model, dataset_reader

def run_eval_loop(modeldir, datadir, role, embed_dim=10, max_tokens=32):
    dataset_reader = data.build_dataset_reader()
    vocab = Vocabulary.from_files(os.path.join(modeldir, 'vocab'))
    model = models.build_model(vocab, embed_dim=embed_dim, max_tokens=max_tokens)
    with open(os.path.join(modeldir, 'best.th'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    predictor = models.SentenceClassifierPredictor(model, dataset_reader)
    labels = []
    preds = []
    with open(os.path.join(datadir, f'{role}-test.csv'), 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # Skip header
            for row in reader:
                item_id, graph_id, text, pred_head_idx, arg_head_idx, label = row
                pred_head_idx, arg_head_idx, label = int(pred_head_idx), int(arg_head_idx), int(label)
                output = predictor.predict(item_id, graph_id, text, pred_head_idx, arg_head_idx, label)
                preds.append(np.argmax(output['probs']))
                labels.append(label)
    report = classification_report(labels, preds, output_dict=True)
    with open(os.path.join(modeldir, 'results.json'), 'w') as fout:
        fout.write(json.dumps(report, indent=4))

def run_example(modeldir, datadir, embed_dim=10, max_tokens=32, upsampled=False):
    dataset_reader = data.build_dataset_reader()
    preds = defaultdict(list)
    for role in ['agent', 'patient', 'theme', 'experiencer', 'recipient']:
        if upsampled:
            model_path = os.path.join(modeldir, f'{role}-upsampled')
        else:
            model_path = os.path.join(modeldir, role)
        vocab = Vocabulary.from_files(os.path.join(model_path, 'vocab'))
        model = models.build_model(vocab, embed_dim=embed_dim, max_tokens=max_tokens)
        with open(os.path.join(model_path, 'best.th'), 'rb') as f:
            model.load_state_dict(torch.load(f))
        predictor = models.SentenceClassifierPredictor(model, dataset_reader)
        with open(os.path.join(datadir, f'examples.csv'), 'r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader) # Skip header
                for row in reader:
                    item_id, graph_id, text, pred_head_idx, arg_head_idx, label = row
                    pred_head_idx, arg_head_idx, = int(pred_head_idx), int(arg_head_idx)
                    output = predictor.predict(item_id, graph_id, text, pred_head_idx, arg_head_idx, 1)
                    # preds[role].append(round(output['probs'][1]*100, 2))
                    preds[role].append(np.argmax(output['probs']))
    table = pd.DataFrame.from_records(preds)
    examples = list(pd.read_csv(os.path.join(datadir, f'examples.csv'))['SENTENCE'])
    table = table[['agent', 'patient', 'theme', 'experiencer', 'recipient']]
    table.index = examples
    table = table.to_latex()
    if upsampled:
        fname = 'latex/example_predictions_upsampled.tex'
    else:
        fname = 'latex/example_predictions.tex'
    with open(fname, 'w') as fout:
        fout.write(table)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('modeldir', help='path to save/load models')
    parser.add_argument('datadir', help='path to data files')
    parser.add_argument('role', choices=['agent', 'patient', 'experiencer', 'recipient', 'theme', 'all'])
    parser.add_argument('action', choices=['train','test', 'example'])
    parser.add_argument('--embed_dim', default=10, type=int)
    parser.add_argument('--max_tokens', default=32, type=int)
    parser.add_argument('--override_modeldir', default=False, action='store_true')
    parser.add_argument('--upsampled_data', default=False, action='store_true')
    args = parser.parse_args()

    start = time.time()
    if args.action=='train':
        if os.path.exists(args.modeldir) and os.path.exists(args.modeldir):
            if len(os.listdir(args.modeldir)) != 0:
                print(f'Overriding {args.modeldir}')
                shutil.rmtree(args.modeldir)
            else:
                raise Exception('Nonempty modeldir')

        run_training_loop(args.modeldir, args.datadir, args.role, 
                            embed_dim=args.embed_dim, max_tokens=args.max_tokens, upsampled_data=args.upsampled_data)
    elif args.action=='test':
        run_eval_loop(args.modeldir, args.datadir, args.role, embed_dim=args.embed_dim, max_tokens=args.max_tokens)
    elif args.action=='example':
        run_example(args.modeldir, args.datadir, embed_dim=args.embed_dim, max_tokens=args.max_tokens, upsampled=args.upsampled_data)
    end = time.time()
    print(f'Time to run script: {end-start} secs')
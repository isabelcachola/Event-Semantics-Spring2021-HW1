import os
import csv
from typing import Dict, Iterable, List, Tuple
from allennlp.data import DatasetReader, Instance, Vocabulary
from allennlp.data.fields import TextField, LabelField, SpanField, MetadataField, Field
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

@DatasetReader.register('classification-csv')
class ClassificationCsvReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def text_to_instance(self, item_id, graph_id, text, pred_head_idx, arg_head_idx, label):
        text_field = TextField(self.tokenizer.tokenize(text),
                                       self.token_indexers)
        label_field = LabelField(label, skip_indexing=True)
        if pred_head_idx < arg_head_idx:
            start, end = pred_head_idx, arg_head_idx
        else:
            start, end = arg_head_idx, pred_head_idx
        span_field = SpanField(start, end, text_field)
        metadata_field = MetadataField(
            {
                'item_id': item_id,
                'graph_id': graph_id,
                'pred_head_idx': pred_head_idx,
                'arg_head_idx': arg_head_idx
            }
        )

        fields = {
                    'text': text_field, 
                    'label': label_field, 
                    'span': span_field,
                    'metadata': metadata_field
        }
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader) # Skip header
            for row in reader:
                # "ITEM_ID","GRAPH_ID","SENTENCE","PRED_HEAD_IDX","ARG_HEAD_IDX","LABEL"
                item_id, graph_id, text, pred_head_idx, arg_head_idx, label = row
                # Convert from string to int
                pred_head_idx, arg_head_idx, label = int(pred_head_idx), int(arg_head_idx), int(label)
                yield self.text_to_instance(item_id, graph_id, text, pred_head_idx, arg_head_idx, label)

def build_dataset_reader(max_tokens=None):
    return ClassificationCsvReader(max_tokens=max_tokens)

def read_data(reader, datadir, role, test_only=False, upsampled_data=False):
    print("Reading data")
    if test_only:
        test_data = list(reader.read(os.path.join(datadir, f'{role}-test.csv')))
        return test_data
    else:
        if upsampled_data:
            training_data = list(reader.read(os.path.join(datadir, f'{role}-train-upsampled.csv')))
        else:
            training_data = list(reader.read(os.path.join(datadir, f'{role}-train.csv')))
        validation_data = list(reader.read(os.path.join(datadir, f'{role}-dev.csv')))
        return training_data, validation_data

def build_vocab(instances):
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)
import tempfile
import json
from torch.nn import functional as F
from typing import Dict, Iterable, List
from allennlp.data.fields.span_field import SpanField

import torch
from allennlp.data import DatasetReader, Instance, Vocabulary, TextFieldTensors
from allennlp.data.fields import LabelField, TextField, Field
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.nn import util
from allennlp.models import Model
from allennlp.training.metrics import CategoricalAccuracy, F1Measure
from allennlp.training.util import evaluate
from allennlp.predictors import Predictor
from allennlp.modules.span_extractors.span_extractor import SpanExtractor

@Model.register('simple_classifier')
class SimpleClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        self.num_labels = 2 #vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), self.num_labels)
        self.span_extractor = SpanExtractor()
        self.metrics = {
            # "accuracy": CategoricalAccuracy(),
            "f1" : F1Measure(positive_label=1)
        }

    def get_span_embeddings(self, embedded_text, span):
        span_embeddings = []
        max_size = 0
        embed_size = embedded_text.shape[-1]
        for i, example in enumerate(embedded_text):
            start, end = span[i]
            span_embedded = example[start:end+1]
            if span_embedded.shape[0] > max_size:
                max_size = span_embedded.shape[0]
            span_embeddings.append(span_embedded)
        for i in range(len(span_embeddings)):
            _s = span_embeddings[i].shape[0]
            span_embeddings[i] = F.pad(span_embeddings[i], (0, 0,0, max_size-_s))
        return torch.stack(span_embeddings)

    def forward(self, text, label, span, metadata):
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        embedded_span = self.get_span_embeddings(embedded_text, span)
        embeddings = torch.cat((embedded_text, embedded_span), axis=1)

        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embeddings)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, item_id, graph_id, text, pred_head_idx, arg_head_idx, label):
        # This method is implemented in the base class.
        return self.predict_json(
            {
            'item_id': item_id, 
            'graph_id': graph_id, 
            'text': text, 
            'pred_head_idx': pred_head_idx, 
            'arg_head_idx': arg_head_idx, 
            'label': label
            }
        )

    def _json_to_instance(self, json_dict):
        return self._dataset_reader.text_to_instance(**json_dict)


def build_model(vocab: Vocabulary, embed_dim=10, max_tokens=32) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    # import ipdb;ipdb.set_trace()
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=embed_dim, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=embed_dim)
    return SimpleClassifier(vocab, embedder, encoder)

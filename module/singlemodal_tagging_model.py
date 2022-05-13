from transformers import BertModel
from module.crf import CRF
import torch.nn as nn
from ZEN import ZenModel
from .base_encoder import BaseEncoder
import numpy as np
import torch


class SinglemodalTaggingModel(BaseEncoder):
    def __init__(self, args, data):
        super(SinglemodalTaggingModel, self).__init__(args)
        if self.args.text_encoder == "ZEN":
            self.text_encoder = ZenModel.from_pretrained(args.zen_directory)
            self.name = "pure_ZEN"
        else:
            if self.args.text_encoder == "BERT":
                plm_directory = args.bert_directory
                self.name = "pure_BERT"
            elif self.args.text_encoder == "WoBERT":
                plm_directory = args.wobert_directory
                self.name = "pure_WoBERT"
            elif self.args.text_encoder == "MacBERT":
                plm_directory = args.macbert_directory
                self.name = "pure_MacBERT"
            else:
                raise Exception("Unsupport encoder: %s" % (self.args.encoder))
            self.text_encoder = BertModel.from_pretrained(plm_directory)
        ## Embedding
        if self.args.use_emb:
            self.char_embedding = nn.Embedding(data.char_alphabet.size(), data.char_emb_dim)
            if args.random_emb:
                self.char_embedding.weight.data.copy_(
                    torch.from_numpy(self.random_embedding(data.char_alphabet.size(), data.char_emb_dim)))
            else:
                self.char_embedding.weight.data.copy_(torch.from_numpy(data.char_embedding))
            self.hidden_dim = self.text_encoder.config.hidden_size + data.char_emb_dim
        else:
            self.hidden_dim = self.text_encoder.config.hidden_size

        self.dropout = nn.Dropout(args.dropout)
        if args.ner_type == "Flat_NER":
            self.tag_size = data.flat_label_alphabet.size()
        else:
            self.tag_size = data.nested_label_alphabet.size()

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tag_size + 2)
        self.crf = CRF(self.tag_size, args.use_gpu, args.crf_reduction)

        if args.fix_embeddings:
            self.text_encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.token_type_embeddings.weight.requires_grad = False

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, input):
        crf_input = self.get_crf_input(input)
        _, best_path = self.crf.viterbi_decode(crf_input, input["label_mask"])
        return best_path

    def neg_log_likelihood(self, input):
        crf_input = self.get_crf_input(input)
        total_loss = self.crf.neg_log_likelihood_loss(crf_input, input["label_mask"], input["label_ids"])
        # _, best_path = self.crf.viterbi_decode(crf_input, input["input_masks"])
        return total_loss

    def get_crf_input(self, input):
        if self.args.text_encoder == "ZEN":
            textual_repr = self.text_encoder(input_ids=input["input_ids"], input_ngram_ids=input["ngram_ids"],
                                                   ngram_position_matrix=input["ngram_positions"],
                                                   attention_mask=input["input_masks"], output_all_encoded_layers=False)[0]
        else:
            textual_repr = self.text_encoder(input["input_ids"], attention_mask=input["input_masks"])[0]
        if self.args.use_emb:
            embeds = self.char_embedding(input["char_emb_ids"])
            embeds = self.dropout(embeds)
            textual_repr = torch.cat((textual_repr, embeds), dim=-1)
        hidden_repr = self.hidden2tag(textual_repr)
        return hidden_repr


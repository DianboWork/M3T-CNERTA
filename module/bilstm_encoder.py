import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from .base_encoder import BaseEncoder
from .crf import CRF


class BiLSTM_Tagging(BaseEncoder):
    def __init__(self, args, data):
        super(BiLSTM_Tagging, self).__init__(args)
        print("building BiLSTM Encoder")
        self.name = "BLSTM_CRF"
        self.char_emb_dim = data.char_emb_dim
        self.char_embeddings = nn.Embedding(data.char_alphabet.size(), self.char_emb_dim)
        if data.char_embedding is not None:
            self.char_embeddings.weight.data.copy_(torch.from_numpy(data.char_embedding))
        else:
            self.char_embeddings.weight.data.copy_(
                torch.from_numpy(self.random_embedding(data.char_alphabet.size(), self.char_emb_dim)))

        self.hidden_dim = args.hidden_dim
        self.lstm = nn.LSTM(self.char_emb_dim, self.hidden_dim // 2 if args.bilstm_flag else self.hidden_dim, num_layers=args.lstm_layer, batch_first=True, bidirectional=args.bilstm_flag)

        if args.ner_type == "Flat_NER":
            label_dim = data.flat_label_alphabet.size()
        else:
            label_dim = data.nested_label_alphabet.size()
        self.hidden2tag = nn.Linear(self.hidden_dim, label_dim + 2)
        self.crf = CRF(label_dim, args.use_gpu, args.crf_reduction)
        self.dropout = nn.Dropout(args.dropout)

    def random_embedding(self, vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def _get_lstm_features(self, batch_char, batch_len):
        embeds = self.char_embeddings(batch_char)
        embeds = self.dropout(embeds)
        embeds_pack = pack_padded_sequence(embeds, batch_len, batch_first=True, enforce_sorted=False)
        out_packed, (_, _) = self.lstm(embeds_pack)
        lstm_feature, _ = pad_packed_sequence(out_packed, batch_first=True, total_length=self.args.max_seq_length)
        lstm_feature = self.dropout(lstm_feature)
        return lstm_feature

    def neg_log_likelihood(self, input):
        lstm_feature = self._get_lstm_features(input["char_emb_ids"], input["char_lens"])
        crf_feature = self.hidden2tag(lstm_feature)
        total_loss = self.crf.neg_log_likelihood_loss(crf_feature, input["label_mask"], input["label_ids"])
        return total_loss

    def forward(self, input):
        lstm_feature = self._get_lstm_features(input["char_emb_ids"], input["char_lens"])
        crf_feature = self.hidden2tag(lstm_feature)
        _, best_path = self.crf.viterbi_decode(crf_feature, input["label_mask"])
        return best_path

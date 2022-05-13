from transformers import BertModel
from .audio_encoder import AudioTransformerEncoder
import torch.nn.functional as F
from .crf import CRF
import torch.nn as nn
import torch
from ZEN import ZenModel
from module.attention_fusion_layer import ShortCutAttentionFusionLayer, GateAttentionFusionLayer
from .ctc_encoder import CTCModel
from .umt import CoupledCMT


class SpeechRecongizer(CTCModel
                       ):
    def __init__(self, args, data):
        super(SpeechRecongizer, self).__init__(args)
        self.audio_encoder = AudioTransformerEncoder(self.args.num_mel_bins, d_model=self.args.audio_hidden_dim, n_blocks=self.args.n_blocks)
        self.audio_embedding = nn.Embedding(data.char_alphabet.size(), self.args.audio_hidden_dim)
        # self.to_vocab = nn.Linear(self.args.audio_hidden_dim, data.char_alphabet.size())
        self.ctc = nn.CTCLoss(blank=data.char_alphabet.blank_id, zero_infinity=True)
        self.char_alphabet = data.char_alphabet

    def forward(self, input):
        audio_logit, audio_mask = self.get_audio_repr(input)
        pred_vocab = audio_logit.argmax(dim=-1).tolist()
        audio_len = audio_mask.sum(dim=1)
        char_emb_ids = input["char_emb_ids"].tolist()
        bsz = len(audio_len)
        for i in range(bsz):
            gold = [self.char_alphabet.get_instance(ele) for ele in char_emb_ids[i] if ele != self.char_alphabet.pad_id]
            pred = [self.char_alphabet.get_instance(ele) for ele in pred_vocab[i]][:audio_len[i]]
            print(pred)
            print(gold)

    def neg_log_likelihood(self, input):
        audio_logit, audio_mask = self.get_audio_repr(input)
        input_lengths = torch.sum(audio_mask, dim=-1)
        target_lengths = input["char_lens"]
        log_prob = masked_log_softmax(audio_logit, input["char_emb_mask"], dim=2)
        # log_prob = F.log_softmax(audio_logit, dim=2)
        bsz = input["char_emb_ids"].size()[0]
        ctc_loss = self.ctc(log_prob.transpose(0, 1), input["char_emb_ids"], input_lengths, target_lengths)
        return ctc_loss

    def get_audio_repr(self, input):
        audio_repr, audio_mask = self.audio_encoder(input["audio_features"], input["audio_feature_masks"])
        audio_logit = torch.matmul(audio_repr, self.audio_embedding.weight.data.transpose(1, 0))
        return audio_logit, audio_mask


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()
    return F.log_softmax(vector, dim=dim)


def masked_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        vector = vector + (1.0 - mask) * -10000.0
    return F.softmax(vector, dim=dim)
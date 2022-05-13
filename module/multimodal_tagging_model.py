from transformers import BertModel
from .audio_encoder import AudioTransformerEncoder
from .crf import CRF
import torch.nn as nn
from ZEN import ZenModel
from .attention_fusion_layer import ShortCutAttentionFusionLayer, GateAttentionFusionLayer
from .umt import CoupledCMT
from .base_encoder import BaseEncoder


class MultimodalTaggingModel(BaseEncoder):
    def __init__(self, args, data):
        super(MultimodalTaggingModel, self).__init__(args)
        self.args = args
        if self.args.text_encoder == "ZEN":
            self.text_encoder = ZenModel.from_pretrained(args.zen_directory)
            self.name = "audio_ZEN"
        else:
            if self.args.text_encoder == "BERT":
                plm_directory = args.bert_directory
                self.name = "audio_BERT"
            elif self.args.text_encoder == "WoBERT":
                plm_directory = args.wobert_directory
                self.name = "audio_WoBERT"
            elif self.args.text_encoder == "MacBERT":
                plm_directory = args.macbert_directory
                self.name = "audio_MacBERT"
            else:
                raise Exception("Unsupport encoder: %s" % (self.args.encoder))
            self.text_encoder = BertModel.from_pretrained(plm_directory)
        self.audio_encoder = AudioTransformerEncoder(self.args.num_mel_bins, d_model=self.args.audio_hidden_dim, n_blocks=self.args.n_blocks)
        self.dropout = nn.Dropout(args.dropout)

        # self.fusion_layer = ShortCutAttentionFusionLayer(self.args)
        if self.args.fusion_layer == "UMT":
            self.fusion_layer = CoupledCMT(self.args)
            output_dim = self.text_encoder.config.hidden_size * 3
        else:
            self.fusion_layer = GateAttentionFusionLayer(self.args)
            output_dim = self.text_encoder.config.hidden_size
        self.name = self.name + "_" + self.args.fusion_layer

        if args.ner_type == "Flat_NER":
            label_dim = data.flat_label_alphabet.size()
        else:
            label_dim = data.nested_label_alphabet.size()

        self.hidden2tag = nn.Linear(output_dim, label_dim + 2)
        self.crf = CRF(label_dim, args.use_gpu, args.crf_reduction)
        if args.fix_embeddings:
            self.text_encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.text_encoder.embeddings.token_type_embeddings.weight.requires_grad = False

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
        audio_repr, audio_mask = self.audio_encoder(input["audio_features"], input["audio_feature_masks"])
        multimodal_repr = self.fusion_layer(textual_repr, input["input_masks"], audio_repr, audio_mask.float())
        hidden_repr = self.hidden2tag(multimodal_repr)
        return hidden_repr

import torch.nn as nn
import torch
from transformers.configuration_bert import BertConfig
import copy, math
from .modeling_bert import BertEncoder, BertIntermediate, BertAttention, BertOutput


class CoupledCMT(nn.Module):
    """Coupled Cross-Modal Attention BERT model for token-level classification with CRF on top.
    """
    def __init__(self, args):
        super(CoupledCMT, self).__init__()
        config = BertConfig(num_hidden_layers=1)
        self.dim_match_1 = nn.Linear(args.audio_hidden_dim, config.hidden_size)
        self.dim_match_2 = nn.Linear(args.audio_hidden_dim, config.hidden_size)
        self.self_attention = BertEncoder(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.text2audio_attention = BertCrossEncoder(config)
        self.audio2text_attention = BertCrossEncoder(config)
        self.text2text_attention = BertCrossEncoder(config)
        self.gate = nn.Linear(config.hidden_size * 2, config.hidden_size)

    def forward(self, textual_hidden_repr, textual_mask, audio_hidden_repr, audio_mask):
        audio_hidden_repr_v1 = self.dim_match_1(audio_hidden_repr)
        audio_hidden_repr_v2 = self.dim_match_2(audio_hidden_repr)
        if audio_mask.dim() == 3:
            extended_audio_mask = (1.0 - audio_mask[:, None, :, :]) * -10000.0
        elif audio_mask.dim() == 2:
            extended_audio_mask = (1.0 - audio_mask[:, None, None, :]) * -10000.0
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    audio_hidden_repr.shape, audio_mask.shape))

        if textual_mask.dim() == 3:
            extended_textual_mask = (1.0 - textual_mask[:, None, :, :]) * -10000.0
        elif textual_mask.dim() == 2:
            extended_textual_mask = (1.0 - textual_mask[:, None, None, :]) * -10000.0
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    textual_hidden_repr.shape, textual_mask.shape)
            )

        cross_encoder = self.text2audio_attention(hidden_states=textual_hidden_repr, encoder_hidden_states=audio_hidden_repr_v1, encoder_attention_mask=extended_audio_mask)
        cross_output_layer = cross_encoder[-1]  # self.batch_size * text_len * hidden_dim

        # apply img2txt attention mechanism to obtain multimodal-based text representations
        cross_text_encoder = self.audio2text_attention(hidden_states=audio_hidden_repr_v2, encoder_hidden_states=textual_hidden_repr, encoder_attention_mask=extended_textual_mask)
        cross_text_output_layer = cross_text_encoder[-1]  # self.batch_size * audio_len * hidden_dim

        cross_final_txt_encoder = self.text2text_attention(hidden_states=textual_hidden_repr, encoder_hidden_states=cross_text_output_layer, encoder_attention_mask=extended_audio_mask)
        cross_final_txt_layer = cross_final_txt_encoder[-1]  # self.batch_size * text_len * hidden_dim


        # gate
        merge_representation = torch.cat((cross_final_txt_layer, cross_output_layer), dim=-1)
        gate_value = torch.sigmoid(self.gate(merge_representation))  # batch_size, text_len, hidden_dim
        gated_converted_att_embed = torch.mul(gate_value, cross_output_layer)
        # reverse_gate_value = torch.neg(gate_value).add(1)
        # gated_converted_att_vis_embed = torch.add(torch.mul(reverse_gate_value, cross_final_txt_layer),
                                        # torch.mul(gate_value, cross_output_layer))

        # direct concatenation
        # gated_converted_att_vis_embed = self.dropout(gated_converted_att_vis_embed)
        final_output = torch.cat((cross_final_txt_layer, gated_converted_att_embed, textual_hidden_repr), dim=-1)
        return final_output


class BertCrossAttentionLayer(nn.Module):
    def __init__(self, config):
        super(BertCrossAttentionLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask):
        attention_output = self.attention(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class BertCrossEncoder(nn.Module):
    def __init__(self, config):
        super(BertCrossEncoder, self).__init__()
        layer = BertCrossAttentionLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, encoder_hidden_states, encoder_attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers

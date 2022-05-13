import torch.nn as nn
import torch
from .modeling_bert import BertAttention, BertIntermediate, BertOutput
from transformers.configuration_bert import BertConfig


class ShortCutAttentionFusionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = BertConfig()
        self.args = args
        if self.args.audio_hidden_dim != self.config.hidden_size:
            self.dim_match = nn.Sequential(
                nn.Linear(self.args.audio_hidden_dim, self.config.hidden_size),
                nn.Tanh()
            )
        self.crossattention = BertAttention(self.config)
        self.intermediate = BertIntermediate(self.config)
        self.output = BertOutput(self.config)

    def forward(
        self,
        textual_hidden_repr,
        textual_mask,
        audio_hidden_repr,
        audio_attention_mask
    ):
        if self.args.audio_hidden_dim != self.config.hidden_size:
            audio_hidden_repr = self.dim_match(audio_hidden_repr)
        if audio_attention_mask.dim() == 3:
            extended_audio_attention_mask = (1.0 - audio_attention_mask[:, None, :, :]) * -10000.0
        elif audio_attention_mask.dim() == 2:
            extended_audio_attention_mask = (1.0 - audio_attention_mask[:, None, None, :]) * -10000.0
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    audio_hidden_repr.shape, audio_attention_mask.shape
                )
            )
        cross_attention_outputs = self.crossattention(
            hidden_states=textual_hidden_repr, encoder_hidden_states=audio_hidden_repr,  encoder_attention_mask=extended_audio_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output+textual_hidden_repr


class GateAttentionFusionLayer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.config = BertConfig()
        self.args = args
        if self.args.audio_hidden_dim != self.config.hidden_size:
            self.dim_match = nn.Sequential(
                nn.Linear(self.args.audio_hidden_dim, self.config.hidden_size),
                nn.Tanh()
            )
        self.hidden_dim = self.config.hidden_size
        self.crossattention = BertAttention(self.config)
        self.intermediate = BertIntermediate(self.config)
        self.output = BertOutput(self.config)
        self.text_linear = nn.Linear(self.hidden_dim, self.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.audio_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_linear = nn.Linear(self.hidden_dim * 2, 1)

    def forward(
        self,
        textual_hidden_repr,
        textual_mask,
        audio_hidden_repr,
        audio_attention_mask
    ):
        if self.args.audio_hidden_dim != self.config.hidden_size:
            audio_hidden_repr = self.dim_match(audio_hidden_repr)
        if audio_attention_mask.dim() == 3:
            audio_extended_attention_mask = audio_attention_mask[:, None, :, :]
        elif audio_attention_mask.dim() == 2:
            audio_extended_attention_mask = audio_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    audio_hidden_repr.shape, audio_attention_mask.shape
                )
            )
        audio_extended_attention_mask = (1.0 - audio_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=textual_hidden_repr, attention_mask=textual_mask, encoder_hidden_states=audio_hidden_repr,  encoder_attention_mask=audio_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        audio_repr = self.output(intermediate_output, attention_output)

        # gate_audio = self.gate_linear(
        #     torch.cat([torch.tanh(self.audio_linear(audio_repr)), torch.tanh(self.text_linear(textual_hidden_repr))], dim=-1))  # [batch_size, max_seq_len, 1]
        # gate_audio = torch.sigmoid(gate_audio)  # [batch_size, max_seq_len, 1]
        # gate_audio = gate_audio.repeat(1, 1, self.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        # multimodal_features = torch.mul(gate_audio, audio_repr) + torch.mul(1 - gate_audio,
        #                                                                         textual_hidden_repr)  # [batch_size, max_seq_len, hidden_dim]

        return audio_repr

class TripleFusionLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        config = BertConfig()
        self.dim_match = nn.Sequential(
            nn.Linear(args.audio_hidden_dim, config.hidden_size),
            nn.Tanh()
        )
        self.hidden_dim = config.hidden_size
        self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
        self.text_linear = nn.Linear(self.hidden_dim, self.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.audio_linear = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_linear = nn.Linear(self.hidden_dim * 2, 3)


    def forward(
            self,
            textual_hidden_repr,
            textual_mask,
            audio_hidden_repr,
            audio_attention_mask
    ):
        audio_hidden_repr = self.dim_match(audio_hidden_repr)
        if audio_attention_mask.dim() == 3:
            audio_extended_attention_mask = audio_attention_mask[:, None, :, :]
        elif audio_attention_mask.dim() == 2:
            audio_extended_attention_mask = audio_attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                    audio_hidden_repr.shape, audio_attention_mask.shape
                )
            )
        audio_extended_attention_mask = (1.0 - audio_extended_attention_mask) * -10000.0
        cross_attention_outputs = self.crossattention(
            hidden_states=textual_hidden_repr, attention_mask=textual_mask, encoder_hidden_states=audio_hidden_repr,
            encoder_attention_mask=audio_extended_attention_mask
        )
        attention_output = cross_attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        audio_repr = self.output(intermediate_output, attention_output)

        # gate_audio = self.gate_linear(
        #     torch.cat([torch.tanh(self.audio_linear(audio_repr)), torch.tanh(self.text_linear(textual_hidden_repr))], dim=-1))  # [batch_size, max_seq_len, 1]
        # gate_audio = torch.sigmoid(gate_audio)  # [batch_size, max_seq_len, 1]
        # gate_audio = gate_audio.repeat(1, 1, self.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        # multimodal_features = torch.mul(gate_audio, audio_repr) + torch.mul(1 - gate_audio,
        #                                                                         textual_hidden_repr)  # [batch_size, max_seq_len, hidden_dim]

        return audio_repr
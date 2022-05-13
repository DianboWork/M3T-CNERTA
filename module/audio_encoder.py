import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class PositionalEncoding(nn.Module):
    """Positional encoding."""
    def __init__(self, d_model, dropout_rate, max_len=5000):
        """Initialize class.
        :param int d_model: embedding dim
        :param float dropout_rate: dropout rate
        :param int max_len: maximum input length
        """
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        pe = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2, dtype=torch.float32) *
                             -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor):
        """Add positional encoding.
        Args:
            x (torch.Tensor): Input. Its shape is (batch, time, ...)
        Returns:
            torch.Tensor: Encoded tensor. Its shape is (batch, time, ...)
        """
        self.extend_pe(x)
        x = x * self.xscale + self.pe[:, :x.size(1)]
        return self.dropout(x)


class Conv2dSubsampling(nn.Module):
    """Convolutional 2D subsampling (to 1/4 length)
    :param int idim: input dim
    :param int odim: output dim
    :param flaot dropout_rate: dropout rate
    """
    def __init__(self, idim, odim, dropout_rate=0.0):
        super(Conv2dSubsampling, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, odim, 3, 2),
            nn.ReLU(),
            nn.Conv2d(odim, odim, 3, 2),
            nn.ReLU()
        )
        self.out = nn.Sequential(
            nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        return x, x_mask.unsqueeze(1)[:, :, :-2:2][:, :, :-2:2]


class MultiHeadedAttention(nn.Module):
    """Multi-Head Attention layer

        :param int n_head: the number of head s
        :param int d_model: the number of features
        :param float dropout_rate: dropout rate
        """

    def __init__(self, n_heads, d_model, dropout_rate=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, query, key, value, mask):
        """Compute 'Scaled Dot Product Attention'

        :param torch.Tensor query: (batch, time1, size)
        :param torch.Tensor key: (batch, time2, size)
        :param torch.Tensor value: (batch, time2, size)
        :param torch.Tensor mask: (batch, time1, time2)
        :param torch.nn.Dropout dropout:
        :return torch.Tensor: attentined and transformed `value` (batch, time1, d_model)
             weighted by the query dot key attention (batch, head, time1, time2)
        """
        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)

        context, _ = self.compute_attn_weights_and_context(q, k, v, mask)
        return context

    def compute_attn_weights_and_context(self, q, k, v, mask=None):
        n_batch = q.size(0)
        q = q.view(n_batch, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.view(n_batch, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.view(n_batch, -1, self.n_heads, self.d_k).transpose(1, 2)  # (batch, head, time2, d_k)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, head, time1, time2)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, time1, time2)
            min_value = float(np.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            attn_weights = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)  # (batch, head, time1, time2)
        else:
            attn_weights = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(attn_weights)
        x = torch.matmul(p_attn, v)  # (batch, head, time1, d_k)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.n_heads * self.d_k)  # (batch, time1, d_model)
        context = self.linear_out(x)  # (batch, time1, d_model)
        return context, attn_weights


class PositionwiseFeedForward(nn.Module):
    """Positionwise feed forward

    :param int idim: input dimenstion
    :param int hidden_units: number of hidden units
    :param float dropout_rate: dropout rate
    """

    def __init__(self, idim, hidden_units, dropout_rate, activation='relu'):
        super(PositionwiseFeedForward, self).__init__()
        self.activation = activation
        self.w_1 = nn.Linear(idim, hidden_units * 2 if activation == 'glu' else hidden_units)
        self.w_2 = nn.Linear(hidden_units, idim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.w_1(x)
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'glu':
            x = F.glu(x)
        elif self.activation == 'gelu':
            x = F.gelu(x)
        elif self.activation == 'swish':
            x = x * torch.sigmoid(x)
        else:
            raise NotImplementedError
        return self.w_2(self.dropout(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, n_heads, d_model, d_ff, slf_attn_dropout_rate,
                 ffn_dropout_rate, residual_dropout_rate, normalize_before=False,
                 concat_after=False, activation='relu'):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(
            n_heads, d_model, slf_attn_dropout_rate)
        self.feed_forward = PositionwiseFeedForward(
            d_model, d_ff, ffn_dropout_rate, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(residual_dropout_rate)
        self.dropout2 = nn.Dropout(residual_dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(d_model * 2, d_model)

    def forward(self, x, mask):
        """Compute encoded features

        :param torch.Tensor x: encoded source features (batch, max_time_in, size)
        :param torch.Tensor mask: mask for x (batch, max_time_in)
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        residual = x
        if self.normalize_before:
            x = self.norm1(x)
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x, x, x, mask)), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout1(self.self_attn(x, x, x, mask))
        if not self.normalize_before:
            x = self.norm1(x)
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        x = residual + self.dropout2(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm2(x)
        return x


class AudioTransformerEncoder(nn.Module):
    def __init__(self, input_size, d_model=256, n_heads=4, d_ff=2048, n_blocks=6,
                 pos_dropout_rate=0.0, slf_attn_dropout_rate=0.0, ffn_dropout_rate=0.0, residual_dropout_rate=0.1,
                 normalize_before=False, concat_after=False, activation='relu'):
        super(AudioTransformerEncoder, self).__init__()
        self.normalize_before = normalize_before
        self.embed = Conv2dSubsampling(input_size, d_model, pos_dropout_rate)
        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                n_heads, d_model, d_ff, slf_attn_dropout_rate, ffn_dropout_rate,
                residual_dropout_rate=residual_dropout_rate, normalize_before=normalize_before,
                concat_after=concat_after, activation=activation) for _ in range(n_blocks)
        ])
        if self.normalize_before:
            self.after_norm = nn.LayerNorm(d_model)
        self.output_size = d_model

    def forward(self, inputs, inputs_mask):
        enc_output, enc_mask = self.embed(inputs, inputs_mask)
        enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)
        for _, block in enumerate(self.blocks):
            enc_output = block(enc_output, enc_mask)
            # enc_output.masked_fill_(~enc_mask.transpose(1, 2), 0.0)
        if self.normalize_before:
            enc_output = self.after_norm(enc_output)
        return enc_output, enc_mask.squeeze(1)
import torch
import torch.nn as nn


class CoAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CoAttention, self).__init__()
        # linear for word-guided audio attention
        self.text_linear_1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.audio_linear_1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.att_linear_1 = nn.Linear(hidden_dim * 2, 1)

        # linear for audio-guided textual attention
        self.text_linear_2 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.audio_linear_2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.att_linear_2 = nn.Linear(hidden_dim * 2, 1)

    def forward(self, text_features, audio_features):
        """
        :param text_features: (batch_size, max_seq_len, hidden_dim)
        :param audio_features: (batch_size, num_audio_region, hidden_dim)
        :return att_text_features (batch_size, max_seq_len, hidden_dim)
                att_audio_features (batch_size, max_seq_len, hidden_dim)
        """
        ############### 1. Word-guided audio attention ###############
        # 1.1. Repeat the vectors -> [batch_size, max_seq_len, num_audio_region, hidden_dim]
        max_seq_len = text_features.shape[1]
        max_audio_len = audio_features.shape[1]
        text_features_repr = text_features.unsqueeze(2).repeat(1, 1, max_audio_len, 1)
        audio_features_repr = audio_features.unsqueeze(1).repeat(1, max_seq_len, 1, 1)

        # 1.2. Feed to single layer (d*k) -> [batch_size, max_seq_len, num_audio_region, hidden_dim]
        text_features_repr = self.text_linear_1(text_features_repr)
        audio_features_repr = self.audio_linear_1(audio_features_repr)

        # 1.3. Concat & tanh -> [batch_size, max_seq_len, num_audio_region, hidden_dim * 2]
        concat_features = torch.cat([text_features_repr, audio_features_repr], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 1.4. Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, num_audio_region]
        audio_att = self.att_linear_1(concat_features).squeeze(-1)
        audio_att = torch.softmax(audio_att, dim=-1)

        # 1.5 Make new image vector with att matrix -> [batch_size, max_seq_len, hidden_dim]
        att_audio_features = torch.matmul(audio_att, audio_features)  # Vt_hat

        ############### 2. audio-guided textual Attention ###############
        # 2.1 Repeat the vectors -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        audio_features_repr = att_audio_features.unsqueeze(2).repeat(1, 1, self.args.max_seq_len, 1)
        text_features_repr = text_features.unsqueeze(1).repeat(1, self.args.max_seq_len, 1, 1)

        # 2.2 Feed to single layer (d*k) -> [batch_size, max_seq_len, max_seq_len, hidden_dim]
        audio_features_repr = self.audio_linear_2(audio_features_repr)
        text_features_repr = self.text_linear_2(text_features_repr)

        # 2.3. Concat & tanh -> [batch_size, max_seq_len, max_seq_len, hidden_dim * 2]
        concat_features = torch.cat([audio_features_repr, text_features_repr], dim=-1)
        concat_features = torch.tanh(concat_features)

        # 2.4 Make attention matrix (linear -> squeeze -> softmax) -> [batch_size, max_seq_len, max_seq_len]
        textual_att = self.att_linear_2(concat_features).squeeze(-1)
        textual_att = torch.softmax(textual_att, dim=-1)

        # 2.5 Make new text vector with att_matrix -> [batch_size, max_seq_len, hidden_dim]
        att_text_features = torch.matmul(textual_att, text_features)  # Ht_hat

        return att_text_features, att_audio_features


class GMF(nn.Module):
    """GMF (Gated Multimodal Fusion)"""

    def __init__(self, args):
        super(GMF, self).__init__()
        self.args = args
        self.text_linear = nn.Linear(args.hidden_dim, args.hidden_dim)  # Inferred from code (dim isn't written on paper)
        self.audio_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

    def forward(self, att_text_features, att_audio_features):
        """
        :param att_text_features: (batch_size, max_seq_len, hidden_dim)
        :param att_audio_features: (batch_size, max_seq_len, hidden_dim)
        :return: multimodal_features
        """
        new_audio_feat = torch.tanh(self.audio_linear(att_audio_features))  # [batch_size, max_seq_len, hidden_dim]
        new_text_feat = torch.tanh(self.text_linear(att_text_features))  # [batch_size, max_seq_len, hidden_dim]

        gate_audio = self.gate_linear(torch.cat([new_audio_feat, new_text_feat], dim=-1))  # [batch_size, max_seq_len, 1]
        gate_audio = torch.sigmoid(gate_audio)  # [batch_size, max_seq_len, 1]
        gate_audio = gate_audio.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]
        multimodal_features = torch.mul(gate_audio, new_audio_feat) + torch.mul(1 - gate_audio, new_text_feat)  # [batch_size, max_seq_len, hidden_dim]

        return multimodal_features


class FiltrationGate(nn.Module):
    """
    In this part, code is implemented in other way compare to equation on paper.
    So I mixed the method between paper and code (e.g. Add `nn.Linear` after the concatenated matrix)
    """

    def __init__(self, args):
        super(FiltrationGate, self).__init__()
        self.args = args

        self.text_linear = nn.Linear(args.hidden_dim, args.hidden_dim, bias=False)
        self.multimodal_linear = nn.Linear(args.hidden_dim, args.hidden_dim, bias=True)
        self.gate_linear = nn.Linear(args.hidden_dim * 2, 1)

        self.resv_linear = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.output_linear = nn.Linear(args.hidden_dim * 2, len(TweetProcessor.get_labels()))

    def forward(self, text_features, multimodal_features):
        """
        :param text_features: Original text feature from BiLSTM [batch_size, max_seq_len, hidden_dim]
        :param multimodal_features: Feature from GMF [batch_size, max_seq_len, hidden_dim]
        :return: output: Will be the input for CRF decoder [batch_size, max_seq_len, hidden_dim]
        """
        # [batch_size, max_seq_len, 2 * hidden_dim]
        concat_feat = torch.cat([self.text_linear(text_features), self.multimodal_linear(multimodal_features)], dim=-1)
        # This part is not written on equation, but if is needed
        filtration_gate = torch.sigmoid(self.gate_linear(concat_feat))  # [batch_size, max_seq_len, 1]
        filtration_gate = filtration_gate.repeat(1, 1, self.args.hidden_dim)  # [batch_size, max_seq_len, hidden_dim]

        reserved_multimodal_feat = torch.mul(filtration_gate,
                                             torch.tanh(self.resv_linear(multimodal_features)))  # [batch_size, max_seq_len, hidden_dim]
        output = self.output_linear(torch.cat([text_features, reserved_multimodal_feat], dim=-1))  # [batch_size, max_seq_len, num_tags]

        return output


class ACN(nn.Module):
    """
    ACN (Adaptive CoAttention Network)
    CharCNN -> BiLSTM -> CoAttention -> GMF -> FiltrationGate -> CRF
    """

    def __init__(self, args, textual_hidden_dim):
        super(ACN, self).__init__()
        # Transform each audio vector as same dimensions ad the text vector
        self.dim_match = nn.Sequential(
            nn.Linear(args.audio_hidden_dim, textual_hidden_dim),
            nn.Tanh()
        )
        self.co_attention = CoAttention(args)
        self.gmf = GMF(args)
        self.filtration_gate = FiltrationGate(args)

    def forward(self, text_features, audio_features, mask):
        """
        :param text_features: (batch_size, max_seq_len)
        :param audio_feature: (batch_size, audio_seq_len, audio_feat_dim)
        :param mask: (batch_size, max_seq_len)
        :return:
        """
        assert text_features.size(-1) == audio_features.size(-1)

        att_text_features, att_audio_features = self.co_attention(text_features, audio_features)
        multimodal_features = self.gmf(att_text_features, att_audio_features)
        logits = self.filtration_gate(text_features, multimodal_features)
        return logits
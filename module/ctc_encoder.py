import torch.nn as nn
import torch
import torch.nn.functional as F


class CTCModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self):
        pass

    def batchify(self, batch):
        guids = [ele.guid for ele in batch]
        padded_audio_features = []
        padded_audio_feature_masks = []
        audio_feautre_lengths = [ele.audio_feature_length for ele in batch]
        max_feature_length = max(audio_feautre_lengths)
        # max_feature_length = self.args.max_audio_length

        for ele in batch:
            padding_feature_len = max_feature_length - ele.audio_feature_length
            padded_audio_features.append(
                F.pad(ele.audio_feature, pad=(0, 0, 0, padding_feature_len), value=0.0).unsqueeze(0))
            padded_audio_feature_masks.append([1] * ele.audio_feature_length + [0] * padding_feature_len)
        audio_features = torch.cat(padded_audio_features, dim=0)
        audio_feature_masks = torch.IntTensor(padded_audio_feature_masks) > 0
        audio_feautre_lengths = torch.IntTensor(audio_feautre_lengths)
        char_emb_ids = torch.LongTensor([ele.char_emb_ids for ele in batch])
        char_lens = torch.LongTensor([ele.char_len for ele in batch])
        char_emb_mask = torch.FloatTensor([ele.char_emb_mask for ele in batch])
        if self.args.use_gpu:
            audio_features = audio_features.cuda()
            audio_feature_masks = audio_feature_masks.cuda()
            audio_feautre_lengths = audio_feautre_lengths.cuda()
            char_emb_ids = char_emb_ids.cuda()
            char_lens = char_lens.cuda()
            char_emb_mask = char_emb_mask.cuda()
        return {"guids": guids, "audio_features": audio_features,
                "audio_feature_masks": audio_feature_masks,
                "audio_feautre_lengths": audio_feautre_lengths,
                "char_emb_ids": char_emb_ids, "char_lens": char_lens, "char_emb_mask": char_emb_mask}

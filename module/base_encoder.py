import torch.nn as nn
import torch
import torch.nn.functional as F


class BaseEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self):
        pass

    def batchify(self, batch):
        guids = [ele.guid for ele in batch]

        ## Audio Feature
        if batch[0].audio_feature is not None:
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
            if self.args.use_gpu:
                audio_features = audio_features.cuda()
                audio_feature_masks = audio_feature_masks.cuda()
                audio_feautre_lengths = audio_feautre_lengths.cuda()
        else:
            audio_features, audio_feature_masks, audio_feautre_lengths = None, None, None

        ## textual feature
        if batch[0].char_input_ids:
            char_input_ids = torch.LongTensor([ele.char_input_ids for ele in batch])
            char_input_masks = torch.FloatTensor([ele.char_input_mask for ele in batch])
            char_emb_ids = torch.LongTensor([ele.char_emb_ids for ele in batch])
            char_lens = torch.LongTensor([ele.char_len for ele in batch])
            char_emb_mask = torch.FloatTensor([ele.char_emb_mask for ele in batch])
            if self.args.ner_type == "Flat_NER":
                label_ids = torch.LongTensor([ele.char_flat_label_ids for ele in batch])
                label_mask = torch.BoolTensor([ele.char_flat_label_mask for ele in batch])
            elif self.args.ner_type == "Nested_NER":
                label_ids = torch.LongTensor([ele.char_nested_label_ids for ele in batch])
                label_mask = torch.BoolTensor([ele.char_nested_label_mask for ele in batch])
            else:
                raise Exception("Invalid NER Type.")
            if self.args.use_gpu:
                label_ids = label_ids.cuda()
                label_mask = label_mask.cuda()
                char_input_ids = char_input_ids.cuda()
                char_input_masks = char_input_masks.cuda()
                char_emb_ids = char_emb_ids.cuda()
                char_lens = char_lens.cuda()
                char_emb_mask = char_emb_mask.cuda()
            if batch[0].ngram_ids:
                ngram_ids = torch.LongTensor([ele.ngram_ids for ele in batch])
                ngram_positions = torch.LongTensor([ele.ngram_positions for ele in batch])
                # ngram_seg_ids = torch.LongTensor([ele.ngram_seg_ids for ele in batch])
                # ngram_masks = torch.LongTensor([ele.ngram_masks for ele in batch])
                # ngram_masks = torch.LongTensor([ele.ngram_masks for ele in batch])
                if self.args.use_gpu:
                    ngram_ids = ngram_ids.cuda()
                    ngram_positions = ngram_positions.cuda()
                return {"guids": guids, "input_ids": char_input_ids, "input_masks": char_input_masks,
                        "label_ids": label_ids, 'label_mask': label_mask, "ngram_ids": ngram_ids,
                        "ngram_positions": ngram_positions, "audio_features": audio_features,
                        "audio_feature_masks": audio_feature_masks, "audio_feautre_lengths": audio_feautre_lengths,
                        "char_emb_ids": char_emb_ids, "char_lens": char_lens, "char_emb_mask": char_emb_mask}
                    #
            else:
                return {"guids": guids, "input_ids": char_input_ids, "input_masks": char_input_masks,
                        "label_ids": label_ids, 'label_mask': label_mask, "audio_features": audio_features,
                        "audio_feature_masks": audio_feature_masks, "audio_feautre_lengths": audio_feautre_lengths,
                        "char_emb_ids": char_emb_ids, "char_lens": char_lens, "char_emb_mask": char_emb_mask}
        else:
            word_input_ids = torch.LongTensor([ele.word_input_ids for ele in batch])
            word_input_masks = torch.FloatTensor([ele.word_input_mask for ele in batch])
            if self.args.ner_type == "Flat_NER":
                label_ids = torch.LongTensor([ele.word_flat_label_ids for ele in batch])
                label_mask = torch.BoolTensor([ele.word_flat_label_mask for ele in batch])
            elif self.args.ner_type == "Nested_NER":
                label_ids = torch.LongTensor([ele.word_nested_label_ids for ele in batch])
                label_mask = torch.BoolTensor([ele.word_nested_label_mask for ele in batch])
            else:
                raise
            if self.args.use_gpu:
                label_ids = label_ids.cuda()
                label_mask = label_mask.cuda()
                word_input_ids = word_input_ids.cuda()
                word_input_masks = word_input_masks.cuda()
            return {"guids": guids, "input_ids": word_input_ids, "input_masks": word_input_masks,
                    "label_ids": label_ids,
                    "label_mask": label_mask, "audio_features": audio_features,
                    "audio_feature_masks": audio_feature_masks,
                    "audio_feautre_lengths": audio_feautre_lengths}
    
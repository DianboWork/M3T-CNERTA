import json, os, sys
from tqdm import tqdm
from utils.alphabet import Alphabet
from copy import deepcopy


class Data(object):
    def __init__(self, args):
        self.args = args
        self.train_examples = self._create_examples(self._read_json(args.train_file), "train")
        self.char_alphabet = Alphabet("character", blankflag=True, padflag=True, unkflag=True, path=args.vocab_path)

        self.train_features = convert_examples_to_features(self.train_examples, self.char_alphabet, args)
        self.char_alphabet.close()
        self.show_data_summary()

    def show_data_summary(self):
        print("DATA SUMMARY START:")
        print("     Char  Alphabet Size  : %s" % self.char_alphabet.size())
        print("     Train Instance Number: %s" % (len(self.train_features)))
        print("     Valid Instance Number: %s" % (len(self.valid_features)))
        print("     Test  Instance Number: %s" % (len(self.valid_features)))
        print("DATA SUMMARY END.")
        sys.stdout.flush()

    @classmethod
    def _read_json(cls, input_file):
        f = open(input_file)
        data = f.readlines()
        data = [ele.rstrip().lstrip('\ufeff') for ele in data]
        data = [json.loads(line) for line in data]
        return data

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            examples.append(InputExample(guid="%s-%s" % (set_type, i), text=line["sentence"], label=line["entity"], audio=line["audio"], speaker_info=line["speaker_info"][0]))
        return examples


class InputExample(object):
    def __init__(self, guid, text, audio, speaker_info, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            audio: string. The id of audio
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.label = label
        self.text = text
        self.audio = audio
        self.speaker_info = speaker_info


class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, guid, audio_feature, char_emb_ids, char_len):
        self.guid = guid
        self.audio_feature = audio_feature
        self.audio_feature_length = self.audio_feature.shape[0]
        self.char_emb_ids = char_emb_ids
        self.char_len = char_len
        self.char_emb_mask = []


def build_char_emb_mask(features, char_alphabet):
    for feature in features:
        char_emb_mask = [0] * char_alphabet.size()
        char_emb_ids = feature.char_emb_ids
        for id in char_emb_ids:
            if id != char_alphabet.pad_id:
                char_emb_mask[id] = 1
        char_emb_mask[char_alphabet.blank_id] = 1
        feature.char_emb_mask = char_emb_mask
    return features


def convert_examples_to_features(examples, char_alphabet, args):
    features = []
    for (_, example) in enumerate(tqdm(examples)):
        audio_feat = audio_feature(example, args)
        char_emb_ids, char_len = emb_feature(example, char_alphabet, args.max_seq_length)
        features.append(InputFeatures(example.guid, audio_feature=audio_feat, char_emb_ids=char_emb_ids, char_len=char_len))
    return features


import torchaudio as ta
import torch
import numpy as np
import random


def emb_feature(example, char_alphabet, max_seq_length):
    char_tokens = []
    for i, char in enumerate(example.text):
        char_tokens.append(char)
    if len(char_tokens) >= max_seq_length - 2:
        char_tokens = char_tokens[0:(max_seq_length - 2)]
    # char_tokens.insert(0, "[CLS]")
    # char_tokens.append("[SEP]")
    char_emb_ids = [char_alphabet.get_index(ele) for ele in char_tokens]
    char_emb_len = len(char_emb_ids)
    while len(char_emb_ids) < max_seq_length:
        char_emb_ids.append(char_alphabet.pad_id)
    return char_emb_ids, char_emb_len


def audio_feature(example, args):
    wavform, sample_frequency = ta.load_wav(args.audio_directory + example.audio + ".wav")
    feature = ta.compliance.kaldi.fbank(wavform, num_mel_bins=args.num_mel_bins, sample_frequency=sample_frequency, dither=0.0)
    if args.normalization:
        std, mean = torch.std_mean(feature)
        feature = (feature - mean) / std
    # if args.gaussian_noise > 0.0:
    #     noise = torch.normal(torch.zeros(feature.size(-1)), std=args.gaussian_noise)
    #     feature += noise
    # if args.apply_spec_augment:
    #     feature = spec_augment(feature)
    return feature


def spec_augment(mel_spectrogram, freq_mask_num=2, time_mask_num=2, freq_mask_rate=0.3, time_mask_rate=0.05, max_mask_time_len=100):
    tau = mel_spectrogram.shape[0]
    v = mel_spectrogram.shape[1]
    warped_mel_spectrogram = mel_spectrogram
    freq_masking_para = int(v * freq_mask_rate)
    time_masking_para = min(int(tau * time_mask_rate), max_mask_time_len)
    # Step 1 : Frequency masking
    if freq_mask_num > 0:
        for _ in range(freq_mask_num):
            f = np.random.uniform(low=0.0, high=freq_masking_para)
            f = int(f)
            f0 = random.randint(0, v-f)
            warped_mel_spectrogram[:, f0:f0+f] = 0
    # Step 2 : Time masking
    if time_mask_num > 0:
        for _ in range(time_mask_num):
            t = np.random.uniform(low=0.0, high=time_masking_para)
            t = int(t)
            t0 = random.randint(0, tau-t)
            warped_mel_spectrogram[t0:t0+t, :] = 0
    return warped_mel_spectrogram

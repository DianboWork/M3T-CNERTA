import jieba
import numpy as np
import math
from random import shuffle
import torchaudio as ta
import torch
import random


def emb_feature(example, char_alphabet, max_seq_length):
    char_tokens = []
    for i, char in enumerate(example.text):
        char_tokens.append(char)
    if len(char_tokens) > max_seq_length - 2:
        char_tokens = char_tokens[0:(max_seq_length - 2)]
    char_tokens.insert(0, "[CLS]")
    char_tokens.append("[SEP]")
    char_emb_ids = [char_alphabet.get_index(ele) for ele in char_tokens]
    char_emb_len = len(char_emb_ids)
    char_emb_mask = [0] * char_alphabet.size()
    for id in char_emb_ids:
        char_emb_mask[id] = 1
    char_emb_mask[char_alphabet.blank_id] = 1
    while len(char_emb_ids) < max_seq_length:
        char_emb_ids.append(char_alphabet.pad_id)
    return char_emb_ids, char_emb_len, char_emb_mask


def char_feature(example, tokenizer, max_seq_length):
    char_tokens = []
    for i, char in enumerate(example.text):
        char_token = tokenizer.tokenize(char)
        char_tokens.extend(char_token)
    if len(char_tokens) >= max_seq_length - 1:
        char_tokens = char_tokens[0:(max_seq_length - 2)]
    output_tokens = char_tokens
    char_tokens.insert(0, "[CLS]")
    char_tokens.append("[SEP]")
    char_input_ids = tokenizer.convert_tokens_to_ids(char_tokens)
    char_input_mask = [1] * len(char_input_ids)
    while len(char_input_ids) < max_seq_length:
        char_input_ids.append(0)
        char_input_mask.append(0)
    return char_input_ids, char_input_mask, output_tokens


def word_feature(example, tokenizer, max_seq_length):
    word_tokens = []
    for i, word in enumerate(jieba.cut(example.text, HMM=False)):
        if word in tokenizer.vocab:
            word_tokens.append(word)
        else:
            for j, char in enumerate(word):
                word_tokens.append(char)
    if len(word_tokens) >= max_seq_length - 1:
        word_tokens = word_tokens[0:(max_seq_length - 2)]
    word_tokens.insert(0, "[CLS]")
    word_tokens.append("[SEP]")
    word_input_ids = tokenizer.convert_tokens_to_ids(word_tokens)
    word_input_mask = [1] * len(word_input_ids)
    while len(word_input_ids) < max_seq_length:
        word_input_ids.append(0)
        word_input_mask.append(0)
    return word_input_ids, word_input_mask


def lexicon_feature(char_tokens, ngram_dict, max_seq_length):
    ngram_matches = []
    #  Filter the ngram segment from 2 to 7 to check whether there is a ngram
    for p in range(2, 8):
        for q in range(0, len(char_tokens) - p + 1):
            character_segment = char_tokens[q:q + p]
            # j is the starting position of the ngram
            # i is the length of the current ngram
            character_segment = tuple(character_segment)
            if character_segment in ngram_dict.ngram_to_id_dict:
                ngram_index = ngram_dict.ngram_to_id_dict[character_segment]
                ngram_matches.append([ngram_index, q, p, character_segment])
    shuffle(ngram_matches)
    max_ngram_in_seq_proportion = math.ceil((len(char_tokens) / max_seq_length) * ngram_dict.max_ngram_in_seq)
    if len(ngram_matches) > max_ngram_in_seq_proportion:
        ngram_matches = ngram_matches[:max_ngram_in_seq_proportion]

    ngram_ids = [ngram[0] for ngram in ngram_matches]
    ngram_positions = [ngram[1] for ngram in ngram_matches]
    ngram_lengths = [ngram[2] for ngram in ngram_matches]
    ngram_tuples = [ngram[3] for ngram in ngram_matches]
    ngram_seg_ids = [0 if position < (len(char_tokens) + 2) else 1 for position in ngram_positions]
    ngram_mask_array = np.zeros(ngram_dict.max_ngram_in_seq, dtype=np.bool)
    ngram_mask_array[:len(ngram_ids)] = 1

    # record the masked positions
    ngram_positions_matrix = np.zeros(shape=(max_seq_length, ngram_dict.max_ngram_in_seq), dtype=np.int32)
    for i in range(len(ngram_ids)):
        ngram_positions_matrix[ngram_positions[i]:ngram_positions[i] + ngram_lengths[i], i] = 1.0

    # Zero-pad up to the max ngram in seq length.
    padding = [0] * (ngram_dict.max_ngram_in_seq - len(ngram_ids))
    ngram_ids += padding
    ngram_lengths += padding
    ngram_seg_ids += padding
    return ngram_ids, ngram_positions_matrix, ngram_lengths, ngram_tuples, ngram_seg_ids, ngram_mask_array


def word2vec_feature(example, alphabet, max_seq_length):
    char_tokens = []
    for i, char in enumerate(example.text):
        char_tokens.append(char)
    if len(char_tokens) >= max_seq_length - 1:
        char_tokens = char_tokens[0: max_seq_length]
    char_input_ids = [alphabet.get_index(ele) for ele in char_tokens]
    return char_input_ids


def audio_feature(example, args):
    wavform, sample_frequency = ta.load_wav(args.audio_directory + example.audio + ".wav")
    feature = ta.compliance.kaldi.fbank(wavform, num_mel_bins=args.num_mel_bins, sample_frequency=sample_frequency, dither=0.0)
    if args.normalization:
        std, mean = torch.std_mean(feature)
        feature = (feature - mean) / std
    if args.gaussian_noise > 0.0:
        noise = torch.normal(torch.zeros(feature.size(-1)), std=args.gaussian_noise)
        feature += noise
    if args.apply_spec_augment:
        feature = spec_augment(feature)
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
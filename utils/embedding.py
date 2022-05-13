import numpy as np


def build_pretrain_embedding(embedding_path, alphabet, skip_first_row=False, separator=" ", embedd_dim=100, norm=True):
    embedd_dict = dict()
    if embedding_path != None:
        embedd_dict, embedd_dim = load_pretrain_emb(embedding_path, skip_first_row, separator)
    scale = np.sqrt(3.0 / embedd_dim)
    pretrain_emb = np.empty([alphabet.size(), embedd_dim])
    perfect_match = 0
    case_match = 0
    not_match = 0
    for alph, index in alphabet.iteritems():
        if alph in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph])
            else:
                pretrain_emb[index, :] = embedd_dict[alph]
            perfect_match += 1
        elif alph.lower() in embedd_dict:
            if norm:
                pretrain_emb[index, :] = norm2one(embedd_dict[alph.lower()])
            else:
                pretrain_emb[index, :] = embedd_dict[alph.lower()]
            case_match += 1
        else:
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedd_dim])
            not_match += 1
    pretrained_size = len(embedd_dict)
    print("Embedding: %s\n     pretrain num:%s, prefect match:%s, case_match:%s, oov:%s, oov%%:%s" % (
    embedding_path, pretrained_size, perfect_match, case_match, not_match, (not_match + 0.) / alphabet.size()))
    return pretrain_emb, embedd_dim


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec / root_sum_square


def load_pretrain_emb(embedding_path, skip_first_row=False, separator=" "):
    embedd_dim = -1
    embedd_dict = dict()
    with open(embedding_path, 'r') as file:
        i = 0
        j = 0
        for line in file:
            if i == 0:
                i = i + 1
                if skip_first_row:
                    _ = line.strip()
                    continue
            j = j+1
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(separator)
            if embedd_dim < 0:
                embedd_dim = len(tokens) - 1
            else:
                if embedd_dim + 1 == len(tokens):
                    embedd = np.empty([1, embedd_dim])
                    embedd[:] = tokens[1:]
                    embedd_dict[tokens[0]] = embedd
                else:
                    continue
    return embedd_dict, embedd_dim
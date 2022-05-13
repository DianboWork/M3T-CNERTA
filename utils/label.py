import jieba
import copy


def is_nested_entity(ent, entity_list):
    flag = False
    for e in entity_list:
        if e != ent:
            if cover(ent, e) or cover(e, ent):
                flag = True
                break
    return flag


def is_outermost_entity(ent, entity_list):
    flag = True
    for e in entity_list:
        if e != ent:
            if cover(ent, e):
                flag = False
                break
    return flag


def cover(e1, e2):
    e1_start = e1[0]
    e1_end = e1[1]
    e2_start = e2[0]
    e2_end = e2[1]
    if e1_start >= e2_start and e1_end <= e2_end and not (e1_start == e2_start and e1_end == e2_end):
        return True
    else:
        return False


def conflict(e1, e2):
    e1_start = e1[0]
    e1_end = e1[1]
    e2_start = e2[0]
    e2_end = e2[1]
    if e1_start > e2_start and e1_start < e2_end < e1_end:
        return True
    elif e2_start > e1_start and e2_start < e1_end < e2_end:
        return True
    else:
        return False


def cws_entities(example, vocab):
    word_text = []
    word_span = []
    word_entities = []
    for _, word in enumerate(jieba.cut(example.text, HMM=False)):
        if word in vocab:
            word_text.append(word)
        else:
            for _, char in enumerate(word):
                word_text.append(char)
    i = 0
    for word in word_text:
        word_span.append([i, i + len(word)])
        i = i + len(word)
    #     print(word_span)
    for ent in example.label:
        flag = True
        for span in word_span:
            if cover(ent, span):
                flag = False
                break
            if conflict(ent, span):
                flag = False
                break
        if flag:
            word_entities.append(copy.deepcopy(ent))
    for ent in word_entities:
        idx = 0
        for span in word_span:
            if span[0] == ent[0]:
                ent[0] = idx
            if span[1] == ent[1]:
                ent[1] = idx + 1
                break
            idx = idx + 1
    return word_entities, word_text


def linearize_label(example, max_seq_length, schema, vocab=None, pad=True):
    if not vocab:
        remained_ent = example.label
        label = ["O"] * len(example.text)
    else:
        remained_ent, word_text = cws_entities(example, vocab)
        label = ["O"] * len(word_text)
    while True:
        if len(remained_ent) == 0:
            break
        else:
            is_outermost = []
            for ent in remained_ent:
                is_outermost.append(is_outermost_entity(ent, remained_ent))
            tagging_ent = [remained_ent[i] for i in range(len(remained_ent)) if is_outermost[i]]
            remained_ent = [remained_ent[i] for i in range(len(remained_ent)) if not is_outermost[i]]
            if schema == "BIO":
                label = assign_label_BIO(tagging_ent, label)
            else:
                label = assign_label_BILOU(tagging_ent, label)
    # pad label
    if pad:
        label_mask = [1] + [1] * len(label) + [1]
        label = ["O"] + label + ["O"]
        while len(label) < max_seq_length:
            label.append("O")
            label_mask.append(0)
        return label, label_mask
    else:
        return label


def flat_label(example, max_seq_length, schema, vocab=None, pad=True):
    if not vocab:
        remained_ent = example.label
        label = ["O"] * len(example.text)
    else:
        remained_ent, word_text = cws_entities(example, vocab)
        label = ["O"] * len(word_text)

    if len(remained_ent) != 0:
        is_outermost = []
        for ent in remained_ent:
            is_outermost.append(is_outermost_entity(ent, remained_ent))
        tagging_ent = [remained_ent[i] for i in range(len(remained_ent)) if is_outermost[i]]
        if schema == "BIO":
            label = assign_label_BIO(tagging_ent, label)
        else:
            label = assign_label_BILOU(tagging_ent, label)
    # pad label
    if pad:
        label_mask = [1] + [1] * len(label) + [1]
        label = ["O"] + label + ["O"]
        while len(label) < max_seq_length:
            label.append("O")
            label_mask.append(0)
        return label, label_mask
    else:
        return label


def assign_label_BILOU(entity_list, label):
    if all(l == "O" for l in label):
        for ent in entity_list:
            start, end = ent[0:2]
            if end - start == 1:
                label[start] = "U-" + ent[-1]
            else:
                label[start:end] = ["I-" + ent[-1]] * (end - start)
                label[start] = "B-" + ent[-1]
                label[end - 1] = "L-" + ent[-1]
        return label
    else:
        for ent in entity_list:
            start, end = ent[0:2]
            if end - start == 1:
                label[start] = label[start] + "|" + "U-" + ent[-1]
            else:
                c_label = ["I-" + ent[-1]] * (end - start)
                c_label[0] = "B-" + ent[-1]
                c_label[-1] = "L-" + ent[-1]
                for i in range(end - start):
                    label[start + i] = label[start + i] + "|" + c_label[i]
        return label


def assign_label_BIO(entity_list, label):
    if all(l == "O" for l in label):
        for ent in entity_list:
            start, end = ent[0:2]
            if end-start == 1:
                label[start] = "B-" + ent[-1]
            else:
                label[start:end] = ["I-" + ent[-1]]*(end-start)
                label[start] = "B-" + ent[-1]
        return label
    else:
        for ent in entity_list:
            start, end = ent[0:2]
            if end-start==1:
                label[start] = label[start] + "|" + "B-" + ent[-1]
            else:
                c_label = ["I-" + ent[-1]]*(end-start)
                c_label[0] = "B-" + ent[-1]
                for i in range(end-start):
                    label[start+i] = label[start+i] + "|" + c_label[i]
        return label

# example  = {'sentence': '华商报记者根据网传的吕某经常跑步地点',
#  'audio': ['BAC009S0706W0384'],
#  'entity': [[0, 3, '华商报', 'ORG'], [10, 11, '吕', 'PER']]}

# a = linearize_label(example)
# print(a)
from transformers import BertModel
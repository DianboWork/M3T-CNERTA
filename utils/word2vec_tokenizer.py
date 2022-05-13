# def tokenize(item):
#     """
#     Args:
#         sentence: string, the input sentence
#         pos_head: [start, end], position of the head entity
#         pos_end: [start, end], position of the tail entity
#         is_token: if is_token == True, sentence becomes an array of token
#     Return:
#         Name of the relation of the sentence
#     """
#     if 'text' in item:
#         sentence = item['text']
#         is_token = False
#     else:
#         sentence = item['token']
#         is_token = True
#     pos_head = item['h']['pos']
#     pos_tail = item['t']['pos']
#
#     # Sentence -> token
#     if not is_token:
#         if pos_head[0] > pos_tail[0]:
#             pos_min, pos_max = [pos_tail, pos_head]
#             rev = True
#         else:
#             pos_min, pos_max = [pos_head, pos_tail]
#             rev = False
#         sent_0 = self.tokenizer.tokenize(sentence[:pos_min[0]])
#         sent_1 = self.tokenizer.tokenize(sentence[pos_min[1]:pos_max[0]])
#         sent_2 = self.tokenizer.tokenize(sentence[pos_max[1]:])
#         ent_0 = self.tokenizer.tokenize(sentence[pos_min[0]:pos_min[1]])
#         ent_1 = self.tokenizer.tokenize(sentence[pos_max[0]:pos_max[1]])
#         if self.mask_entity:
#             ent_0 = ['[UNK]']
#             ent_1 = ['[UNK]']
#         tokens = sent_0 + ent_0 + sent_1 + ent_1 + sent_2
#         if rev:
#             pos_tail = [len(sent_0), len(sent_0) + len(ent_0)]
#             pos_head = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
#         else:
#             pos_head = [len(sent_0), len(sent_0) + len(ent_0)]
#             pos_tail = [len(sent_0) + len(ent_0) + len(sent_1), len(sent_0) + len(ent_0) + len(sent_1) + len(ent_1)]
#     else:
#         tokens = sentence
#
#     # Token -> index
#     if self.blank_padding:
#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, self.max_length, self.token2id['[PAD]'],
#                                                               self.token2id['[UNK]'])
#     else:
#         indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokens, unk_id=self.token2id['[UNK]'])
#
#     # Position -> index
#     pos1 = []
#     pos2 = []
#     pos1_in_index = min(pos_head[0], self.max_length)
#     pos2_in_index = min(pos_tail[0], self.max_length)
#     for i in range(len(tokens)):
#         pos1.append(min(i - pos1_in_index + self.max_length, 2 * self.max_length - 1))
#         pos2.append(min(i - pos2_in_index + self.max_length, 2 * self.max_length - 1))
#
#     if self.blank_padding:
#         while len(pos1) < self.max_length:
#             pos1.append(0)
#         while len(pos2) < self.max_length:
#             pos2.append(0)
#         indexed_tokens = indexed_tokens[:self.max_length]
#         pos1 = pos1[:self.max_length]
#         pos2 = pos2[:self.max_length]
#
#     indexed_tokens = torch.tensor(indexed_tokens).long().unsqueeze(0)  # (1, L)
#     pos1 = torch.tensor(pos1).long().unsqueeze(0)  # (1, L)
#     pos2 = torch.tensor(pos2).long().unsqueeze(0)  # (1, L)
#
#     # Mask
#     mask = []
#     pos_min = min(pos1_in_index, pos2_in_index)
#     pos_max = max(pos1_in_index, pos2_in_index)
#     for i in range(len(tokens)):
#         if i <= pos_min:
#             mask.append(1)
#         elif i <= pos_max:
#             mask.append(2)
#         else:
#             mask.append(3)
#     # Padding
#     if self.blank_padding:
#         while len(mask) < self.max_length:
#             mask.append(0)
#         mask = mask[:self.max_length]
#
#     mask = torch.tensor(mask).long().unsqueeze(0)  # (1, L)
#     return indexed_tokens, pos1, pos2, mask



import torch, random, gc, copy, time, json
from torch import nn, optim
from tqdm import tqdm
try:
    from transformers import AdamW
except:
    from pytorch_transformers import AdamW
from optim.radam import RAdam
from optim.radamw import RAdamW
from optim.adabelief import AdaBelief
from utils.average_meter import AverageMeter
from utils.metric import get_flat_ner_fmeasure, get_nested_ner_fmeasure, get_flat_ner, get_nested_ner
# from utils.batchify import batchify
import nni


class Trainer(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data
        if self.args.ner_type == "Nested_NER":
            self.label_alphabet = self.data.nested_label_alphabet
        elif self.args.ner_type == "Flat_NER":
            self.label_alphabet = self.data.flat_label_alphabet
        else:
            raise Exception("Unsupport NER type: %s" % (self.args.ner_type))
        self.assign_optimizer()
        if args.use_gpu:
            self.model = self.model.to(torch.device("cuda"))

    def train_model(self):
        best_test_f1 = 0
        train_features = self.data.train_features
        train_num = len(train_features)
        batch_size = self.args.batch_size
        total_batch = train_num // batch_size + 1
        for epoch in range(self.args.max_epoch):
            # Train
            self.model.train()
            self.model.zero_grad()
            self.optimizer = self.lr_decay(self.optimizer, epoch, self.args.lr_decay)
            print("=== Epoch %d train ===" % epoch, flush=True)
            avg_loss = AverageMeter()
            random.shuffle(train_features)
            # for batch_id in range(total_batch):
            for batch_id in tqdm(range(total_batch)):
                start = batch_id * batch_size
                end = (batch_id + 1) * batch_size
                if end > train_num:
                    end = train_num
                    continue
                batch_features = train_features[start:end]
                if not batch_features:
                    continue
                batch = self.model.batchify(batch_features)
                loss = self.model.neg_log_likelihood(batch)
                avg_loss.update(loss.item(), 1)
                # Optimize
                loss.backward()
                if self.args.max_grad_norm != 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 100 == 0 and batch_id != 0:
                    print("     Instance: %d; loss: %.4f" % (start, avg_loss.avg), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            print("=== Epoch %d Test ===" % epoch, flush=True)
            speed, acc, p, r, f, pred_results = self.eval_model("valid")
            speed, acc, p, r, f, pred_results = self.eval_model("test")

            # speed, acc, p, r, f, pred_results = self.eval_model("train")

            # best_param_name = self.args.generated_param_directory + "%s_%s_epoch_%d_f1_%.4f.model" % (
            # self.model.name, self.args.ner_type, epoch, f)
            # best_param = copy.deepcopy(self.model.state_dict())
            # torch.save(best_param, best_param_name)
            """@nni.report_intermediate_result(f)"""
            if f > best_test_f1:
                print("Achieving Best Result on Test Set.", flush=True)
                best_param_name = self.args.generated_param_directory + "%s_%s_epoch_%d_f1_%.4f.model" %(self.model.name, self.args.ner_type, epoch, f)
                best_param = copy.deepcopy(self.model.state_dict())
                torch.save(best_param, best_param_name)
                best_test_f1 = f
                best_test_result_epoch = epoch
            gc.collect()
            torch.cuda.empty_cache()
        print("Best result on test set is %f achieving at epoch %d." % (best_test_f1, best_test_result_epoch),
              flush=True)
        # print("Best model param are save at %s. " % (best_param_name))
        # torch.save(best_param, best_param_name)
        """@nni.report_final_result(best_test_f1)"""

    def eval_model(self, name):
        if name == "train":
            features = self.data.train_features
            examples = self.data.train_examples
        elif name == "valid":
            features = self.data.valid_features
            examples = self.data.valid_examples
        elif name == 'test':
            features = self.data.test_features
            examples = self.data.test_examples
        else:
            raise Exception("Unsupport evaluation set: %s" % (name))

        pred_results = []
        gold_results = []
        self.model.eval()
        batch_size = self.args.batch_size
        start_time = time.time()
        eval_num = len(features)
        total_batch = eval_num // batch_size + 1
        for batch_id in tqdm(range(total_batch)):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > eval_num:
                end = eval_num
            batch_features = features[start:end]
            batch_examples = examples[start:end]
            if not batch_features:
                continue
            batch = self.model.batchify(batch_features)
            tag_seq = self.model(batch)
            pred_label, gold_label = self.recover_label(tag_seq, batch["label_ids"], batch["label_mask"])
            pred_results += pred_label
            gold_results += gold_label
        decode_time = time.time() - start_time
        speed = eval_num / decode_time
        if self.args.ner_type == "Flat_NER":
            acc, p, r, f = get_flat_ner_fmeasure(gold_results, pred_results, self.args.schema)
        else:
            acc, p, r, f = get_nested_ner_fmeasure(gold_results, pred_results, self.args.schema)
        if name == "train":
            print(
                "Train: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                decode_time, speed, acc, p, r, f))
        elif name == "valid":
            print(
                "Valid: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                    decode_time, speed, acc, p, r, f))
        else:
            print(
                "Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
                    decode_time, speed, acc, p, r, f))
        return speed, acc, p, r, f, pred_results

    def load_state_dict(self, state_dict):
        self.model.load_state_dict(state_dict)

    @staticmethod
    def lr_decay(optimizer, epoch, decay_rate):
        # lr = init_lr * ((1 - decay_rate) ** epoch)
        if epoch != 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] * (1 - decay_rate)
                # print(param_group['lr'])
        return optimizer

    def recover_label(self, pred_variable, gold_variable, mask_variable):
        """
            input:
                pred_variable (batch_size, sent_len): pred tag result
                gold_variable (batch_size, sent_len): gold result variable
                mask_variable (batch_size, sent_len): mask variable
        """
        seq_len = gold_variable.size(1)
        mask = mask_variable.cpu().data.numpy()
        pred_tag = pred_variable.cpu().data.numpy()
        gold_tag = gold_variable.cpu().data.numpy()
        batch_size = mask.shape[0]
        pred_label = []
        gold_label = []
        for idx in range(batch_size):
            pred = [self.label_alphabet.get_instance(pred_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            gold = [self.label_alphabet.get_instance(gold_tag[idx][idy]) for idy in range(seq_len) if mask[idx][idy] != 0]
            assert (len(pred) == len(gold))
            pred_label.append(pred)
            gold_label.append(gold)
        return pred_label, gold_label

    def assign_optimizer(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        component = ['crf', 'audio_encoder', "fusion_layer"]
        try:
            grouped_params = [
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay) and component[0] not in n and component[
                                   1] not in n and component[2] not in n],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.textual_encoder_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and component[0] not in n and component[1] not in n and
                               component[2] not in n],
                    'weight_decay': 0.0,
                    'lr': self.args.textual_encoder_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay) and component[0] in n],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.crf_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and component[0] in n],
                    'weight_decay': 0.0,
                    'lr': self.args.crf_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay) and component[1] in n],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.audio_encoder_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and component[1] in n],
                    'weight_decay': 0.0,
                    'lr': self.args.audio_encoder_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay) and component[2] in n],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.fusion_layer_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and component[2] in n],
                    'weight_decay': 0.0,
                    'lr': self.args.fusion_layer_lr
                }
            ]
        except:
            grouped_params = [
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay) and component[0] not in n and component[
                                   1] not in n and component[2] not in n],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.textual_encoder_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and component[0] not in n and component[1] not in n and
                               component[2] not in n],
                    'weight_decay': 0.0,
                    'lr': self.args.textual_encoder_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               not any(nd in n for nd in no_decay) and component[0] in n],
                    'weight_decay': self.args.weight_decay,
                    'lr': self.args.crf_lr
                },
                {
                    'params': [p for n, p in self.model.named_parameters() if
                               any(nd in n for nd in no_decay) and component[0] in n],
                    'weight_decay': 0.0,
                    'lr': self.args.crf_lr
                }
            ]
        if self.args.optimizer == 'Adam':
            self.optimizer = optim.Adam(grouped_params)
        elif self.args.optimizer == 'AdamW':
            self.optimizer = AdamW(grouped_params)
        elif self.args.optimizer == "RAdam":
            self.optimizer = RAdam(grouped_params)
        elif self.args.optimizer == "RAdamW":
            self.optimizer = RAdamW(grouped_params)
        elif self.args.optimizer == "SGD":
            self.optimizer = optim.SGD(grouped_params)
        elif self.args.optimizer == "AdaBelief":
            self.optimizer = AdaBelief(grouped_params)
        else:
            raise Exception("Invalid optimizer.")

    def output(self, name, file_name):
        if name == "train":
            features = [ele for ele in self.data.train_features if ele.guid in self.wrong_ids]
            examples = self.data.train_examples
        elif name == "valid":
            features = self.data.valid_features
            examples = self.data.valid_examples
        elif name == 'test':
            features = self.data.test_features
            examples = self.data.test_examples
        else:
            raise Exception("Unsupport evaluation set: %s" % (name))
        pred_results = []
        gold_results = []
        self.model.eval()
        batch_size = self.args.batch_size
        start_time = time.time()
        eval_num = len(features)
        total_batch = eval_num // batch_size + 1
        file = open(file_name, "w")

        for batch_id in tqdm(range(total_batch)):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > eval_num:
                end = eval_num
            batch_features = features[start:end]
            batch_examples = examples[start:end]

            if not batch_features:
                continue
            batch = self.model.batchify(batch_features)
            tag_seq = self.model(batch)
            pred_label, gold_label = self.recover_label(tag_seq, batch["label_ids"], batch["label_mask"])
            if self.args.ner_type == "Flat_NER":
                golden_full, predict_full, right_full = get_flat_ner(gold_label, pred_label, label_type=self.args.schema)
            else:
                golden_full, predict_full, right_full = get_nested_ner(gold_label, pred_label,
                                                                     label_type=self.args.schema)
            for i in range(len(examples[start:end])):
                example = batch_examples[i]
                text = example.text
                label = example.label
                gold = golden_full[i]
                pred = predict_full[i]
                right = right_full[i]
                outstring = json.dumps({"text": text, "label": label, "gold": gold, "pred": pred, "rigth": right}, ensure_ascii=False)
                file.write(outstring+'\n')
import torch, random, gc, copy, time, json
from torch import nn, optim
from tqdm import tqdm
try:
    from transformers import AdamW
except:
    from pytorch_transformers import AdamW
from utils.average_meter import AverageMeter


class Trainer(nn.Module):
    def __init__(self, model, data, args):
        super().__init__()
        self.args = args
        self.model = model
        self.data = data
        self.optimizer = optim.Adam(self.model.named_parameters(), self.args.audio_encoder_lr)
        if args.use_gpu:
            self.model = self.model.to(torch.device("cuda"))

    def train_model(self):
        best_test_f1 = 0
        # self.eval_model("test")
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

                if (batch_id + 1) % self.args.gradient_accumulation_steps == 0:
                    self.optimizer.step()
                    self.model.zero_grad()
                if batch_id % 2 == 0 and batch_id != 0:
                    print("     Instance: %d; loss: %.4f" % (start, avg_loss.avg), flush=True)


            gc.collect()
            torch.cuda.empty_cache()
            # print("=== Epoch %d Test ===" % epoch, flush=True)
            # speed, acc, p, r, f, pred_results = self.eval_model("valid")
            # self.eval_model("train")
            #
            # # speed, acc, p, r, f, pred_results = self.eval_model("train")
            #
            # # best_param_name = self.args.generated_param_directory + "%s_%s_epoch_%d_f1_%.4f.model" % (
            # # self.model.name, self.args.ner_type, epoch, f)
            # # best_param = copy.deepcopy(self.model.state_dict())
            # # torch.save(best_param, best_param_name)
            # """@nni.report_intermediate_result(f)"""
            # if f > best_test_f1:
            #     print("Achieving Best Result on Test Set.", flush=True)
            #     best_param_name = self.args.generated_param_directory + "%s_%s_epoch_%d_f1_%.4f.model" %(self.model.name, self.args.ner_type, epoch, f)
            #     best_param = copy.deepcopy(self.model.state_dict())
            #     best_test_f1 = f
            #     best_test_result_epoch = epoch
            # gc.collect()
            # torch.cuda.empty_cache()
        # print("Best result on test set is %f achieving at epoch %d." % (best_test_f1, best_test_result_epoch),
        #       flush=True)
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
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > eval_num:
                end = eval_num
            batch_features = features[start:end]
            # batch_examples = examples[start:end]
            if not batch_features:
                continue
            batch = self.model.batchify(batch_features)
            self.model(batch)
        #     pred_label, gold_label = self.recover_label(tag_seq, batch["label_ids"], batch["label_mask"])
        #     pred_results += pred_label
        #     gold_results += gold_label
        # decode_time = time.time() - start_time
        # speed = eval_num / decode_time
        # if self.args.ner_type == "Flat_NER":
        #     acc, p, r, f = get_flat_ner_fmeasure(gold_results, pred_results, self.args.schema)
        # else:
        #     acc, p, r, f = get_nested_ner_fmeasure(gold_results, pred_results, self.args.schema)
        # if name == "train":
        #     print(
        #         "Train: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
        #         decode_time, speed, acc, p, r, f))
        # elif name == "valid":
        #     print(
        #         "Valid: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
        #             decode_time, speed, acc, p, r, f))
        # else:
        #     print(
        #         "Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (
        #             decode_time, speed, acc, p, r, f))
        # return speed, acc, p, r, f, pred_results

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

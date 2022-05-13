import argparse, os, torch
import random
import numpy as np
import json
import pickle
from utils.data import Data
from module.multitask_tagging_model import MultitaskTaggingModel
import nni


def str2bool(v):
    return v.lower() in ('true')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    return arg


def get_args():
    args, unparsed = parser.parse_known_args()
    if len(unparsed) > 1:
        print("Unparsed args: {}".format(unparsed))
    return args, unparsed


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    textual_encoder_lr = 1e-5
    lr_decay = 0.05
    audio_encoder_lr = 0.00002
    max_grad_norm = 5
    ctc_coef = 0.05
    n_blocks = 9

    parser = argparse.ArgumentParser()
    data_arg = add_argument_group('Data')
    data_arg.add_argument('--dataset_name', type=str, default="Aishell-NER")
    data_arg.add_argument('--train_file', type=str, default="./data/new_text/train.json")
    data_arg.add_argument('--valid_file', type=str, default="./data/new_text/new_valid.json")
    data_arg.add_argument('--test_file', type=str, default="./data/new_text/new_test.json")
    data_arg.add_argument('--generated_data_directory', type=str, default="./data/generated_data/")
    data_arg.add_argument('--generated_param_directory', type=str, default="./data/generated_data/model_param/")
    data_arg.add_argument('--bert_directory', type=str, default="./plm/bert_base_chinese/")
    data_arg.add_argument('--macbert_directory', type=str, default="./plm/macbert/")
    data_arg.add_argument('--wobert_directory', type=str, default="./plm/wobert/")
    data_arg.add_argument('--zen_directory', type=str, default="./plm/zen/")
    data_arg.add_argument('--emb_file', type=str, default="/home/suidianbo/data/ch_embeddings/gigaword_chn.char50d.vec")

    data_arg.add_argument('--max_seq_length', type=int, default=50)
    data_arg.add_argument('--schema', type=str, default="BIO", choices=["BIO", "BILOU"])
    data_arg.add_argument('--ner_type', type=str, default="Flat_NER", choices=['Nested_NER', 'Flat_NER'])
    data_arg.add_argument('--use_audio_feature', type=str2bool, default=True)
    data_arg.add_argument('--audio_directory', type=str, default="/home/suidianbo/aishell_ner_audio/")
    data_arg.add_argument('--vocab_path', type=str, default="./data/vocab.json")

    feature_arg = add_argument_group('Audio-Feature')
    feature_arg.add_argument('--normalization', type=str2bool, default=True)
    feature_arg.add_argument('--apply_spec_augment', type=str2bool, default=True)
    feature_arg.add_argument('--gaussian_noise', type=float, default=0)
    feature_arg.add_argument('--num_mel_bins', type=int, default=40, choices=[40, 80])

    model_arg = add_argument_group('Model')
    model_arg.add_argument('--text_encoder', type=str, default="MacBERT", choices=['BERT', 'ZEN', 'MacBERT', 'WoBERT'])
    model_arg.add_argument('--audio_hidden_dim', type=int, default=256, help="the dimension of audio encoder")
    model_arg.add_argument('--n_blocks', type=int, default=n_blocks, help="the num of transformer blocks in audio encoder")
    model_arg.add_argument("--fusion_layer", type=str, default="MultiHeadAttn",
                           choices=["DCTC", "MultiHeadAttn", "UMT", "CTC"])

    model_arg.add_argument('--use_emb', type=str2bool, default=True)
    model_arg.add_argument('--emb_dim', type=int, default=256)
    model_arg.add_argument('--random_emb', type=str2bool, default=True)

    learning_arg = add_argument_group('Learning')
    learning_arg.add_argument('--max_audio_length', type=int, default=1468)
    learning_arg.add_argument('--fix_embeddings', type=str2bool, default=False)
    learning_arg.add_argument('--crf_reduction', type=str, default="mean", choices=["sum", "mean"])
    learning_arg.add_argument('--ctc_reduction', type=str, default="mean", choices=["sum", "mean"])
    learning_arg.add_argument('--batch_ctc_loss', type=str2bool, default=False)

    learning_arg.add_argument('--dropout', type=float, default=0.1)
    learning_arg.add_argument('--batch_size', type=int, default=16)
    learning_arg.add_argument('--max_epoch', type=int, default=20)
    learning_arg.add_argument('--gradient_accumulation_steps', type=int, default=1)
    learning_arg.add_argument('--textual_encoder_lr', type=float, default=textual_encoder_lr)
    learning_arg.add_argument('--audio_encoder_lr', type=float, default=audio_encoder_lr)
    learning_arg.add_argument('--fusion_layer_lr', type=float, default=1e-5)
    learning_arg.add_argument('--crf_lr', type=float, default=0.1)
    learning_arg.add_argument('--lr_decay', type=float, default=lr_decay)
    learning_arg.add_argument('--weight_decay', type=float, default=1e-5)
    learning_arg.add_argument('--max_grad_norm', type=float, default=max_grad_norm)
    learning_arg.add_argument('--optimizer', type=str, default='AdamW',
                              choices=['Adam', 'AdamW', 'RAdam', "SGD", 'RAdamW', 'AdaBelief'])
    learning_arg.add_argument('--adversarial_training', type=str2bool, default=False)
    learning_arg.add_argument('--ctc_coef', type=float, default=ctc_coef)
    learning_arg.add_argument('--crf_coef', type=float, default=1)

    misc_arg = add_argument_group('MISC')
    misc_arg.add_argument('--refresh', type=str2bool, default=True)
    misc_arg.add_argument('--use_gpu', type=str2bool, default=True)
    misc_arg.add_argument('--visible_gpu', type=int, default=3)
    misc_arg.add_argument('--random_seed', type=int, default=10)

    args, unparsed = get_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpu)
    for arg in vars(args):
        print(arg, ":", getattr(args, arg))
    set_seed(args.random_seed)
    # data = Data(args)
    # with open(args.generated_data_directory+"multitask_giga_emb.pkl", "wb") as f:
    #     pickle.dump(data, f)
    with open(args.generated_data_directory + "unified_data_full.pkl", "rb") as f:
        data = pickle.load(f)
    model = MultitaskTaggingModel(args, data)
    print("Loading pretrained audio encoder param")
    # pretrained_audioencoder_dict = torch.load(args.generated_param_directory + "masked_pretrained_audio_full_epoch_150.model")
    # # model.load_state_dict(audio_param)
    # model_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_audioencoder_dict.items() if k in model_dict}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    for n, p in model.named_parameters():
        print(n)
    if args.adversarial_training:
        from trainer.fgm_trainer import Trainer
    else:
        from trainer.trainer import Trainer
    trainer = Trainer(model, data, args)
    trainer.train_model()
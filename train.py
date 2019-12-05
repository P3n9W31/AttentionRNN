import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from terminaltables import AsciiTable
from tensorboardX import SummaryWriter

import model
from bleu import bleu
from dataset import dataset
from util import (
    convert_data,
    convert_str,
    get_logger,
    invert_vocab,
    load_vocab,
    sort_batch,
)


parser = argparse.ArgumentParser(
    description="Training Attention-based Neural Machine Translation Model"
)
# data
parser.add_argument("--src_vocab", type=str, help="source vocabulary")
parser.add_argument("--trg_vocab", type=str, help="target vocabulary")
parser.add_argument(
    "--src_max_len", type=int, default=50, help="maximum length of source"
)
parser.add_argument(
    "--trg_max_len", type=int, default=50, help="maximum length of target"
)
parser.add_argument("--train_src", type=str, help="source for training")
parser.add_argument("--train_trg", type=str, help="target for training")
parser.add_argument("--valid_src", type=str, help="source for validation")
parser.add_argument(
    "--valid_trg", type=str, nargs="+", help="references for validation"
)
parser.add_argument("--vfreq", type=int, default=3, help="frequency for validation")
parser.add_argument("--eval_script", type=str, help="script for validation")
# model
parser.add_argument("--model", type=str, help="the name of model")
parser.add_argument("--name", type=str, default="", help="the name of checkpoint")
parser.add_argument(
    "--enc_num_input", type=int, default=512, help="size of source word embedding"
)
parser.add_argument(
    "--dec_num_input", type=int, default=512, help="size of target word embedding"
)
parser.add_argument(
    "--enc_num_hidden", type=int, default=1024, help="number of source hidden layer"
)
parser.add_argument(
    "--dec_num_hidden", type=int, default=1024, help="number of target hidden layer"
)
parser.add_argument(
    "--dec_natt", type=int, default=1000, help="number of target attention layer"
)
parser.add_argument("--nreadout", type=int, default=620, help="number of maxout layer")
parser.add_argument(
    "--enc_emb_dropout",
    type=float,
    default=0.4,
    help="dropout rate for encoder embedding",
)
parser.add_argument(
    "--dec_emb_dropout",
    type=float,
    default=0.4,
    help="dropout rate for decoder embedding",
)
parser.add_argument(
    "--enc_hid_dropout",
    type=float,
    default=0.4,
    help="dropout rate for encoder hidden state",
)
parser.add_argument(
    "--readout_dropout", type=float, default=0.4, help="dropout rate for readout layer"
)
# optimization
parser.add_argument(
    "--optim", type=str, default="RMSprop", help="optimization algorihtim"
)
parser.add_argument(
    "--batch_size", type=int, default=64, help="input batch size for training"
)
parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
parser.add_argument("--l2", type=float, default=0, help="L2 regularization")
parser.add_argument("--grad_clip", type=float, default=1, help="gradient clipping")
parser.add_argument(
    "--finetuning", action="store_true", help="whether or not fine-tuning"
)
parser.add_argument("--decay_lr", action="store_true", help="decay learning rate")
parser.add_argument(
    "--half_epoch",
    action="store_true",
    help="decay learning rate at the beginning of epoch",
)
parser.add_argument(
    "--epoch_best", action="store_true", help="store best model for epoch"
)
parser.add_argument(
    "--restore",
    action="store_true",
    help="decay learning rate at the beginning of epoch",
)
parser.add_argument("--beam_size", type=int, default=10, help="size of beam search")
parser.add_argument("--sfreq", type=int, default=3, help="frequency for sampling")
# bookkeeping
parser.add_argument("--seed", type=int, default=42, help="random number seed")
parser.add_argument(
    "--checkpoint", type=str, default="./checkpoint/", help="path to save the model"
)
parser.add_argument("--freq", type=int, help="frequency for save")
# GPU
parser.add_argument("--cuda", action="store_true", help="use cuda")
parser.add_argument("--local_rank", type=int, help="use cuda")
# Misc
parser.add_argument("--nepoch", type=int, default=40, help="number of epochs to train")
parser.add_argument("--epoch", type=int, default=0, help="epoch of checkpoint")
parser.add_argument("--info", type=str, help="info of model")

# log
if not os.path.exists("log"):
    os.mkdir("log")

log_path = os.path.join(
    "log", "log-" + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()) + ".txt"
)
logger = get_logger(log_path)

# parameters
opt = parser.parse_args()

# seed
if opt.local_rank:
    opt.seed += opt.local_rank
torch.manual_seed(opt.seed)

# cuda
opt.cuda = opt.cuda and torch.cuda.is_available()
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

# device
device_type = "cuda" if opt.cuda else "cpu"
device_ids = None
if opt.local_rank is not None:
    device_type += ":" + str(opt.local_rank)
    device_ids = [opt.local_rank]
device = torch.device(device_type)

# tensorboardX
writer = SummaryWriter()

# load vocabulary for source and target
src_vocab, trg_vocab = {}, {}
src_vocab["stoi"] = load_vocab(opt.src_vocab)
trg_vocab["stoi"] = load_vocab(opt.trg_vocab)
src_vocab["itos"] = invert_vocab(src_vocab["stoi"])
trg_vocab["itos"] = invert_vocab(trg_vocab["stoi"])
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"
PAD = "<pad>"
opt.enc_pad = src_vocab["stoi"][PAD]
opt.dec_sos = trg_vocab["stoi"][SOS]
opt.dec_eos = trg_vocab["stoi"][EOS]
opt.dec_pad = trg_vocab["stoi"][PAD]
opt.enc_num_token = len(src_vocab["stoi"])
opt.dec_num_token = len(trg_vocab["stoi"])

# load dataset for training and validation
train_dataset = dataset(opt.train_src, opt.train_trg, opt.src_max_len, opt.trg_max_len)
valid_dataset = dataset(opt.valid_src, opt.valid_trg)
train_iter = torch.utils.data.DataLoader(
    train_dataset,
    opt.batch_size,
    shuffle=True,
    num_workers=6,
    collate_fn=lambda x: list(zip(*x)),
)
valid_iter = torch.utils.data.DataLoader(
    valid_dataset, 1, num_workers=6, shuffle=False, collate_fn=lambda x: list(zip(*x))
)

opt.sfreq = len(train_iter)
opt.vfreq = len(train_iter)

paras = [["Parameters", "Value"]]
for key, value in opt.__dict__.items():
    paras.append([str(key), str(value)])
paras_table = AsciiTable(paras)
logger.info("\n" + str(paras_table.table))

# create the model
model = getattr(model, opt.model)(opt).to(device)

# initialize the parameters
for p in model.parameters():
    p.data.uniform_(-0.1, 0.1)

if opt.name:
    state_dict = torch.load(os.path.join(opt.checkpoint, opt.name))
    model.load_state_dict(state_dict)

param_list = list(model.parameters())
param_group = param_list

# create the optimizer
optimizer = getattr(optim, opt.optim)(param_group, lr=opt.lr, weight_decay=opt.l2)

opt.score_list = []
opt.epoch_best_score = -float("inf")
opt.cur_lr = " ".join([str(g["lr"]) for g in optimizer.param_groups])
opt.tmp_name = None
opt.best_name = None
opt.epoch_best_name = None


def save_model(model, batch_idx, epoch, info="tmp"):
    date = time.strftime("%m-%d|%H:%M", time.localtime(time.time()))
    name = "model_%s_%s_lr_%.1e_cur_lr_%s_l2_%.1e_batch_%d_e%d-%d_%s.%s.pt" % (
        opt.model,
        opt.info,
        opt.lr,
        opt.cur_lr,
        opt.l2,
        opt.batch_size,
        epoch,
        batch_idx,
        date,
        info,
    )
    torch.save(model.state_dict(), os.path.join(opt.checkpoint, name))
    return name


def adjust_learningrate(score_list):
    if len(score_list) > 1 and score_list[-1][0] < 0.999 * score_list[-2][0]:
        if opt.restore:
            m_state_dict = torch.load(os.path.join(opt.checkpoint, opt.best_name))
            model.load_state_dict(m_state_dict, strict=False)
        cur_lr_list = []
        for k, group in enumerate(optimizer.param_groups):
            group["lr"] = group["lr"] * 0.1
            cur_lr_list.append(group["lr"])
        opt.cur_lr = " ".join([str(v) for v in cur_lr_list])
        logger.info(str("Current learning rate:" + opt.cur_lr))


def train(epoch):
    model.train()
    opt.epoch_best_score = -float("inf")
    opt.epoch_best_name = None
    for batch_idx, batch in enumerate(train_iter, start=1):
        batch = sort_batch(batch)
        src_raw = batch[0]
        trg_raw = batch[1]
        src, src_mask = convert_data(
            src_raw, src_vocab, device, True, UNK, PAD, SOS, EOS
        )
        f_trg, f_trg_mask = convert_data(
            trg_raw, trg_vocab, device, False, UNK, PAD, SOS, EOS
        )
        b_trg, b_trg_mask = convert_data(
            trg_raw, trg_vocab, device, True, UNK, PAD, SOS, EOS
        )
        optimizer.zero_grad()
        if opt.cuda and torch.cuda.device_count() > 1 and opt.local_rank is None:
            loss, w_loss = nn.parallel.data_parallel(
                model, (src, src_mask, f_trg, f_trg_mask, b_trg, b_trg_mask), device_ids
            )
        else:
            loss, w_loss = model(src, src_mask, f_trg, f_trg_mask, b_trg, b_trg_mask)
        global_batches = len(train_iter) * epoch + current_batches
        writer.add_scalar(
            "./loss", scalar_value=loss.item(), global_step=global_batches,
        )
        loss.mean().backward()
        torch.nn.utils.clip_grad_norm_(param_list, opt.grad_clip)
        optimizer.step()
        if batch_idx % 10 == 0 or batch_idx == len(train_iter) or batch_idx == 0:
            logger.info(
                str(
                    "Epoch: {} batch: {}/{}({:.3%}), loss: {:.6}, lr: {}".format(
                        epoch,
                        batch_idx,
                        len(train_iter),
                        batch_idx / len(train_iter),
                        loss.item(),
                        opt.cur_lr,
                    )
                )
            )

        # validation
        if batch_idx % opt.vfreq == 0:
            logger.info(str("===========validation / test START==========="))
            evaluate(batch_idx, epoch)
            model.train()
            if opt.decay_lr:
                adjust_learningrate(opt.score_list)
            if len(opt.score_list) == 1 or opt.score_list[-1][0] > max(
                [x[0] for x in opt.score_list[:-1]]
            ):
                if opt.best_name is not None:
                    os.remove(os.path.join(opt.checkpoint, opt.best_name))
                opt.best_name = save_model(model, batch_idx, epoch, "best")
            if opt.epoch_best and opt.score_list[-1][0] > opt.epoch_best_score:
                opt.epoch_best_score = opt.score_list[-1][0]
                if opt.epoch_best_name is not None:
                    os.remove(os.path.join(opt.checkpoint, opt.epoch_best_name))
                opt.epoch_best_name = save_model(model, batch_idx, epoch, "epoch-best")
            logger.info("===========validation / test DONE===========")

        # sampling
        if batch_idx % opt.sfreq == 0:
            length = len(src_raw)
            ix = np.random.randint(0, length)
            samp_src_raw = [src_raw[ix]]
            samp_trg_raw = [trg_raw[ix]]
            samp_src, samp_src_mask = convert_data(
                samp_src_raw, src_vocab, device, True, UNK, PAD, SOS, EOS
            )
            model.eval()
            with torch.no_grad():
                output = model.beamsearch(samp_src, samp_src_mask, opt.beam_size)
            best_hyp, best_score = output[0]
            best_hyp = convert_str([best_hyp], trg_vocab)
            sampling_result = []
            sampling_result.append(["Key", "Value"])
            sampling_result.append(["Source", str(" ".join(samp_src_raw[0]))])
            sampling_result.append(["Target", str(" ".join(samp_trg_raw[0]))])
            sampling_result.append(["Predict", str(" ".join(best_hyp[0]))])
            sampling_result.append(["Best Score", str(round(best_score, 5))])
            sampling_table = AsciiTable(sampling_result)
            logger.info("===========sampling START===========")
            logger.info("\n" + str(sampling_table.table))
            logger.info("===========sampling DONE===========")
            model.train()

        # saving model
        if opt.freq and batch_idx % opt.freq == 0:
            if opt.tmp_name is not None:
                os.remove(os.path.join(opt.checkpoint, opt.tmp_name))
            opt.tmp_name = save_model(model, batch_idx, epoch, "tmp")


def bleu_script(f):
    ref_stem = opt.valid_trg[0][:-1] + "*"
    cmd = "{eval_script} {refs} {hyp}".format(
        eval_script=opt.eval_script, refs=ref_stem, hyp=f
    )
    p = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if p.returncode > 0:
        sys.stderr.write(err)
        sys.exit(1)
    bleu = float(out)
    return bleu


def evaluate(batch_idx, epoch):
    model.eval()
    hyp_list = []
    ref_list = []
    start_time = time.time()
    for ix, batch in enumerate(valid_iter, start=1):
        src_raw = batch[0]
        trg_raw = batch[1:]
        src, src_mask = convert_data(
            src_raw, src_vocab, device, True, UNK, PAD, SOS, EOS
        )
        with torch.no_grad():
            output = model.beamsearch(src, src_mask, opt.beam_size, normalize=True)
            best_hyp, best_score = output[0]
            best_hyp = convert_str([best_hyp], trg_vocab)
            hyp_list.append(best_hyp[0])
            ref = [x[0] for x in trg_raw]
            ref_list.append(ref)
    elapsed = time.time() - start_time

    hyp_list = [" ".join(x) for x in hyp_list]
    p_tmp = tempfile.mktemp()
    f_tmp = open(p_tmp, "w")
    f_tmp.write("\n".join(hyp_list))
    f_tmp.close()
    bleu2 = bleu_script(p_tmp)

    bleu_1_gram = bleu(hyp_list, ref_list, smoothing=True, n=1)
    bleu_2_gram = bleu(hyp_list, ref_list, smoothing=True, n=2)
    bleu_3_gram = bleu(hyp_list, ref_list, smoothing=True, n=3)
    bleu_4_gram = bleu(hyp_list, ref_list, smoothing=True, n=4)
    writer.add_scalar("./bleu_1_gram", bleu_1_gram, epoch)
    writer.add_scalar("./bleu_2_gram", bleu_2_gram, epoch)
    writer.add_scalar("./bleu_3_gram", bleu_3_gram, epoch)
    writer.add_scalar("./bleu_4_gram", bleu_4_gram, epoch)
    writer.add_scalar("./multi-bleu", bleu2, epoch)
    bleu_result = [
        ["multi-bleu", "bleu_1-gram", "bleu_2-gram", "bleu_3-gram", "bleu_4-gram"],
        [bleu2, bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram],
    ]
    bleu_table = AsciiTable(bleu_result)
    logger.info(
        "BLEU score for Epoch-{}-batch-{}: ".format(epoch, batch_idx)
        + "\n"
        + bleu_table.table
    )
    opt.score_list.append(
        (bleu2, bleu_1_gram, bleu_2_gram, bleu_3_gram, bleu_4_gram, batch_idx, epoch)
    )


if __name__ == "__main__":
    for epoch in range(opt.epoch, opt.epoch + opt.nepoch):
        train(epoch)

    best = max(opt.score_list, key=lambda x: x[0])
    logger.info(str("best BLEU Epoch{}/batch-{}: {}".format(best[-1], best[-2], best[0])))

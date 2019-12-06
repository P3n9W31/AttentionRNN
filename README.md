# Attention-based RNN for Seq2Seq Machine Translation

Attention-based RNN model for Chinese-English translation

## Basic Structure

<p align="center">
<img src="https://github.com/P3n9W31/AttEncDecRNN/blob/master/figures/structure.png" width="600">
</p>

> Britz D, Goldie A, Luong M T, et al. Massive exploration of neural machine translation architectures[J]. arXiv preprint arXiv:1703.03906, 2017.

## Process

<p align="center">
<img src="https://github.com/P3n9W31/AttEncDecRNN/blob/master/figures/process.gif" width="600">
</p>

> google/seq2seq. GitHub.  https://github.com/google/seq2seq

## Data Explanation

The Chinese-English translation data used in this project is just sample data, change them as you like.(Cn-863k, En-1.1M)

**Data Format**: 

```
sentence-1-word-1 sentence-1-word-2 sentence-1-word-3. [\n]
sentence-2-word-1 sentence-2-word-2 sentence-2-word-3 sentence-2-word-4. [\n]
......
```

Chinese-English data should be paired.



## Installation

Python3.6+ needed.

The following packages are needed:

```txt
terminaltables==3.1.0
numpy==1.14.0
torch==1.3.0
tensorboardX==1.9
```

Easily, you can install all requirement with:

```
pip3 install -r requirements.txt
```

You better do it under a Virtual environment, for Pytorch is kinda annoying to migrate between different versions.

## Usage

1. **Generating vocabulary**

   First thing first, Generating vocabulary for training, run **bulid_vocab.sh** in terminal:

   ```bash
   ./bulid_vocab.sh
   ```

   or you can modify the data source and path in bulid_vocab.sh:

   ```bash
   python3 scripts/buildvocab.py \
   --corpus data/cn.txt \
   --output data/cn.voc.pkl \
   --limit 300000 \
   --groundhog
   
   python3 scripts/buildvocab.py \
   --corpus data/en.txt \
   --output data/en.voc.pkl \
   --limit 300000 \
   --groundhog
   
   python3 scripts/generate_test_data.py
   ```

   then run.

2. **Modifying hyperparameters**

   Some of the hyperparameters are as follow:

   ```TXT
   +-----------------+----------------------+-----------------------------------------------+
   | Parameters      | Value                |                                               |
   +-----------------+----------------------+-----------------------------------------------+
   | src_vocab       | data/cn.voc.pkl      |source vocabulary                              |
   | trg_vocab       | data/en.voc.pkl      |target vocabulary                              |
   | src_max_len     | 50                   |maximum length of source                       |
   | trg_max_len     | 50                   |maximum length of target                       |
   | train_src       | data/cn.txt          |source for training                            |
   | train_trg       | data/en.txt          |source for validation                          |
   | valid_src       | data/cn.test.txt     |source for validation                          |
   | valid_trg       | ['data/en.test.txt'] |references for validation                      |
   | vfreq           | 54                   |frequency for validation                       |
   | eval_script     | scripts/validate.sh  |script for validation                          |
   | model           | AttEncDecRNN         |the name of model                              |
   | name            |                      |the name of checkpoint                         |
   | enc_num_input   | 512                  |size of source word embedding                  |
   | dec_num_input   | 512                  |size of target word embedding                  |
   | enc_num_hidden  | 1024                 |number of source hidden layer                  |
   | dec_num_hidden  | 1024                 |number of target hidden layer                  |
   | dec_natt        | 1024                 |number of target attention layer               |
   | nreadout        | 620                  |number of maxout layer                         |
   | enc_emb_dropout | 0.4                  |dropout rate for encoder embedding             |
   | dec_emb_dropout | 0.4                  |dropout rate for decoder embedding             |
   | enc_hid_dropout | 0.4                  |dropout rate for encoder hidden state          |
   | readout_dropout | 0.4                  |dropout rate for readout layer                 |
   | optim           | RMSprop              |optimization algorihtim                        |
   | batch_size      | 128                  |input batch size for training                  |
   | lr              | 0.0005               |learning rate                                  |
   | l2              | 0                    |L2 regularization                              |
   | grad_clip       | 1                    |gradient clipping                              |
   | finetuning      | False                |whether or not fine-tuning                     |
   | decay_lr        | False                |decay learning rate                            |
   | half_epoch      | True                 |decay learning rate at the beginning of epoch  |
   | epoch_best      | False                |store best model for epoch                     |
   | restore         | False                |decay learning rate at the beginning of epoc   |
   | beam_size       | 10                   |size of beam search                            |
   | sfreq           | 54                   |frequency for sampling                         |
   | seed            | 42                   |random number seed                             |
   | checkpoint      | ./checkpoint/        |path to save the model                         |
   | freq            | None                 |frequency for save                             |
   | cuda            | False                |use cuda                                       |
   | local_rank      | None                 |cuda configuration                             |
   | nepoch          | 40                   |number of epochs to train                      |
   | epoch           | 0                    |epoch of checkpoint                            |
   | info            |                      |info of model                                  |
   +-----------------+----------------------+-----------------------------------------------+
   ```

   Change it as you like, pass them to model on next step.

3. **Training the model**

   Passing hyperparameters via **start.sh**: 

   ```bash
   #! /bin/bash
   python3 train.py \
   --src_vocab data/cn.voc.pkl \
   --trg_vocab data/en.voc.pkl \
   --train_src data/cn.txt \
   --train_trg data/en.txt \
   --valid_src data/cn.test.txt \
   --valid_trg data/en.test.txt \
   --eval_script scripts/validate.sh \
   --model AttEncDecRNN \
   --optim RMSprop \
   --batch_size 64 \
   --half_epoch \
   --cuda
   ```

   Then run it in terminal: 

   ```bash
   ./start.sh
   ```

4. Visualize the training process on tensorboard

   ```bash
   tensorboard --logdir runs
   ```

   

## Evaluation

The evaluation metric for Chinese-English we use is case-insensitive BLEU. We use the `muti-bleu.perl` script from [Moses](https://github.com/moses-smt/mosesdecoder) to compute the BLEU.

Loss and BLEU Score:

<p align="center">
<img src="https://github.com/P3n9W31/AttEncDecRNN/blob/master/figures/loss.png" width="400">
<img src="https://github.com/P3n9W31/AttEncDecRNN/blob/master/figures/bleu.png" width="400">
</p>

As the data I use is too simple, the results are **just a reference**.

## Results on Chinese-English translation

1. Epoch-1

   ```
   +------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Key        | Value                                                                                                                                                                                                                                                                                 |
   +------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Source     | 事实 的 真相 是 , 两 会 达成 的 共识 是 ‘ 海峡 两 岸 均 坚持 一 个 中国 原则 ’ 。                                                                                                                                                                                                              |
   | Target     | the truth is that the consensus reached by the arats and the sef is that " both sides on the taiwan strait stick to the one-china principle . "                                                                                                                                       |
   | Predict    | the , the , the , the , the , the , the , the , the , and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the and the |
   +------------+---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
   ```

2. Epoch-10

   ```
   +------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Key        | Value                                                                                                                                                        |
   +------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Source     | 干部们 的 出发点 是 好 的 , 但 不 能 急于 求成 , 以 牺牲 农民 的 权益 为 代价 。                                                                                       |
   | Target     | the intention of those cadres is good , but they should not be over-anxious to achieve an instant result at the expense of peasants ' rights and interests . |
   | Predict    | it is not only by the fundamental interests of the united states , but it is not only by the people 's interests and the people .                            |
   +------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
   ```

3. Epoch-20

   ```
   +------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Key        | Value                                                                                                                                                          |
   +------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Source     | 例如 美国 , 其 国内 存在 严重 违反 人权 的 状况 , 迄今 非但 没有 改善 , 反而 不断 恶化 。                                                                                 |
   | Target     | in the united states , for instance , there are serious violations of human rights , and far from being put right , this situation is actually deteriorating . |
   | Predict    | in the united states , for instance , human rights situation in the united states , for human rights , and this is not rather serious .                        |
   +------------+----------------------------------------------------------------------------------------------------------------------------------------------------------------+
   ```

4. Epoch-30

   ```
   +------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Key        | Value                                                                                                                                                       |
   +------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
   | Source     | 真正 的 马克思主义者 , 不 可能 要求 经典 作家 为 他们 身 后 所 产生 的 问题 提供 现成 的 答案 。                                                                        |
   | Target     | real marxists cannot require the authors of the [ marxist ] classics to provide ready-made solutions to the problems that have occurred since their death . |
   | Predict    | real marxists cannot require the authors of the [ marxist ] classics to provide ready-made solutions to the problems that have occurred since their death . |
   +------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------+
   ```

5. Epoch-40

   ```
   +------------+-------------------------------------------------------------------------------------------------------------------------------------------+
   | Key        | Value                                                                                                                                     |
   +------------+-------------------------------------------------------------------------------------------------------------------------------------------+
   | Source     | 一九九0 年 一月 二十三日 , 她 辞职 并 申请 赴 美 , 于 当年 定居 美国 , 一九九六年 加入 美国 国籍 。                                                   |
   | Target     | on 23 january 1990 she resigned ad requested to go to the united states , where she settled that year . she became a us citizen in 1996 . |
   | Predict    | on 23 january 1990 she resigned ad requested to go to the united states , where she settled that year . she became a us citizen in 1996 . |
   +------------+-------------------------------------------------------------------------------------------------------------------------------------------+
   ```

   Again, the results are **just a reference**.



## Device

Tested on CPU and Single GPU.

| Device Type | Device                                    | Speed                |
| ----------- | ----------------------------------------- | -------------------- |
| CPU         | Intel(R) Xeon(R) CPU E5-2678 v3 @ 2.50GHz | 3 min 20 sec / Epoch |
| GPU         | GeForce GTX 1080 Ti                       | 12 sec / Epoch       |

## To Do

* Train on public dataset
* Test script

## License

MIT License
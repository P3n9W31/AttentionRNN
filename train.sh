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
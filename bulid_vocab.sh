#! /bin/bash
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
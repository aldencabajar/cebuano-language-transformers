#!/usr/bin/bash

python nlptools/WikiExtractor.py /mnt/d/cebwiki-latest-pages-articles-multistream.xml \
 --min_text_length 100 --json -b 10G 

 

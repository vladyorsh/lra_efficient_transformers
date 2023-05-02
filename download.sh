#!/bin/sh
# Downloads LRA data necessary to run models and installs the Huggingface libraries

# Text classification setup needs only repo, since its training dataset is already in the TF library
git clone https://github.com/google-research/long-range-arena.git

# The rest needs a full data archive downloaded from GDrive
wget https://storage.googleapis.com/long-range-arena/lra_release.gz
gzip -d lra_release.gz
tar -xf lra_release

#Install HF libraries for the AG setup
pip install transformers datasets sentencepiece
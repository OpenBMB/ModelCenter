# coding=utf-8

# The current implementation is mainly to adapt the training framework of the Transformers toolkit, 
# and replace the original model implementation.
# TODO we will change to our SAM implementation in the future, which will be 

from transformers import BertTokenizer as transformers_BertTokenizer

BertTokenizer = transformers_BertTokenizer

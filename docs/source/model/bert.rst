=======================
bert
=======================

`Bert <https://arxiv.org/abs/1810.04805>`_

BertConfig
------------------------------------
.. autoclass:: model_center.model.BertConfig
   :members:

BertModel
------------------------------------
.. autoclass:: model_center.model.Bert
   :members:

BertTokenizer
------------------------------------
.. class:: model_center.tokenizer.BertTokenizer

The current implementation is mainly an alias to BertTokenizer of `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`_.
we will change to our SAM implementation in the future, which will be a more efficient tokenizer.
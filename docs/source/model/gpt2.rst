=======================
gpt2
=======================

`GPT2 <https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf>`_

GPT2Config
------------------------------------
.. autoclass:: model_center.model.GPT2Config
   :members:

GPT2Model
------------------------------------
.. autoclass:: model_center.model.GPT2
   :members:

GPT2Tokenizer
------------------------------------
.. class:: model_center.tokenizer.GPT2Tokenizer

The current implementation is mainly an alias to GPT2Tokenizer of `Hugging Face Transformers <https://huggingface.co/docs/transformers/index>`_.
we will change to our SAM implementation in the future, which will be a more efficient tokenizer.
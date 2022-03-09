from .config import Config

class BertConfig(Config): #mBert

    def __init__(
        self,
        vocab_size=119547,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        chunk_size_feed_forward=0,
        add_cross_attention=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        torchscript=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.pad_token_id = pad_token_id
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.add_cross_attention = add_cross_attention
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states
        self.return_dict = return_dict
        self.torchscript = torchscript

    @property
    def use_return_dict(self) -> bool:
        """
        `bool`: Whether or not return [`~file_utils.ModelOutput`] instead of tuples.
        """
        # If torchscript is set, force `return_dict=False` to avoid jit errors
        return self.return_dict and not self.torchscript

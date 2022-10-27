import contextlib
from typing import Optional

import numpy as np
from dataclasses import dataclass, field


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    model_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    pooling_model: bool = field(
        default=False, metadata={"help": "Whether to use mean pooling of text encoding for triplet modeling"}
    )
    text_loss_weight: float = field(
        default=0.1, metadata={"help": "The weight of text loss"}
    )
    pos_weight: Optional[float] = field(
        default=None, metadata={
            "help": "The weight of positive labels in knowledge loss. This should be equal to the number of pos-neg pairs (not the number of negative samples)"
        }
    )


@dataclass
class DataArguments:
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    data_cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the examples and features"}
    )
    num_neg: int = field(
        default=1, metadata={"help": "The number of negative samples."}
    )
    margin: float = field(
        default=1., metadata={"help": "The margin of knowledge loss"}
    )
    no_text: bool = field(
        default=False, metadata={"help": "Whether not to use text as part of input"}
    )
    data_debug: bool = field(
        default=False, metadata={"help": "Whether use only a small part of data for debugging"}
    )
    max_seq_length: int = field(
        default=128, metadata={
            "help": '''The maximum total input sequence length after WordPiece tokenization. 
                                          Sequences longer than this will be truncated, and sequences shorter 
                                          than this will be padded.'''
        }
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )
    mask_ratio: float = field(
        default=0.15, metadata={"help": "The ratio of examples to be masked"}
    )
    group_shuffle: bool = field(
        default=False, metadata={
            "help": "Whether use group shuffle such that the positive and negative samples are always in the same batch"
        })
    test_ratio: float = field(
        default=1.0, metadata={"help": "The ratio of test data used to evaluate the performance"}
    )
    text_sep_token: Optional[str] = field(
        default=None,
        metadata={"help": "The token added between entity, relation and tail text. Add no tokens by default"}
    )
    type_constrain: bool = field(
        default=False
    )
    no_mid: bool = field(
        default=False
    )
    data_split: bool = field(default=False)
    num_split: int = field(default=5)
    rank: int = field(default=0)
    only_corrupt_entity: bool = field(default=False)
    train_file: str = field(default='train.tsv')

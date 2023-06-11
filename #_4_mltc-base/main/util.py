import copy
import json
import pickle
from typing import Union, List, Optional, Dict

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, GPT2Tokenizer


def get_verbalization_ids(word: str, tokenizer: PreTrainedTokenizer, force_single_token: bool) -> Union[int, List[int]]:
    """
    Get the token ids corresponding to a verbalization

    :param word: the verbalization
    :param tokenizer: the tokenizer to use
    :param force_single_token: whether it should be enforced that the verbalization corresponds to a single token.
           If set to true, this method returns a single int instead of a list and throws an error if the word
           corresponds to multiple tokens.
    :return: either the list of token ids or the single token id corresponding to this word
    """
    kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
    ids = tokenizer.encode(word, add_special_tokens=False, **kwargs)
    if not force_single_token:
        return ids
    assert len(ids) == 1, \
        f'Verbalization "{word}" does not correspond to a single token, got {tokenizer.convert_ids_to_tokens(ids)}'
    verbalization_id = ids[0]
    assert verbalization_id not in tokenizer.all_special_ids, \
        f'Verbalization {word} is mapped to a special token {tokenizer.convert_ids_to_tokens(verbalization_id)}'
    return verbalization_id


class InputExample(object):
    """原始样本类"""

    def __init__(self, guid, text, word_label, index_label, encode_label):
        """
        Create a new InputExample.

        :param guid: a unique textual identifier
        :param text: the sequence of text
        :param word_label: 单词形式的标签 如['sport','tech']
        :param index_label: 索引形式的标签 从总标签集里面取出word_label对应的索引 如[2,0]
        :param encode_label: 独热编码形式的标签 根据总标签集 把index_label转化成独热标签 如[0,1,1,0]
        """
        self.guid = guid
        self.text = text
        self.word_label = word_label
        self.index_label = index_label
        self.encode_label = encode_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @staticmethod
    def load_examples(path: str) -> List['InputExample']:
        """Load a set of input examples from a file"""
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    @staticmethod
    def save_examples(examples: List['InputExample'], path: str) -> None:
        """Save a set of input examples to a file"""
        with open(path, 'wb') as fh:
            pickle.dump(examples, fh)


class InputFeatures(object):
    """从`InputExample`获取的特征存入`InputFeatures`类"""

    def __init__(self, input_ids, attention_mask, token_type_ids, encode_label, labels, mlm_labels=None):
        """
        Create new InputFeatures.

        :param input_ids: the input ids corresponding to the original text or text sequence
        :param attention_mask: an attention mask, with 0 = no attention, 1 = attention
        :param token_type_ids: segment ids as used by BERT
        :param encode_label: 标签的独热编码
        :param mlm_labels: an optional sequence of labels used for auxiliary language modeling
        """
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.encode_label = encode_label
        self.mlm_labels = mlm_labels
        self.labels = labels

    def __repr__(self):
        return str(self.to_json_string())

    # ## 便于格式化输出
    def pretty_print(self, tokenizer):
        return f'input_ids         = {tokenizer.convert_ids_to_tokens(self.input_ids)}\n' + \
               f'attention_mask    = {self.attention_mask}\n' + \
               f'token_type_ids    = {self.token_type_ids}\n' + \
               f'mlm_labels        = {self.mlm_labels}\n' + \
               f'encode_label      = {self.encode_label}\n' + \
               f'labels            = {self.labels}'

    def to_dict(self):
        """Serialize this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class DictDataset(Dataset):
    """A dataset of tensors that uses a dictionary for key-value mappings"""

    def __init__(self, **tensors):
        tensors.values()

        assert all(next(iter(tensors.values())).size(0) == tensor.size(0) for tensor in tensors.values())
        self.tensors = tensors

    def __getitem__(self, index):
        return {key: tensor[index] for key, tensor in self.tensors.items()}

    def __len__(self):
        return next(iter(self.tensors.values())).size(0)

# logits: [32, 50, 30522] labels: [32, 50]
def LabelSmoothLoss(logits, labels, epsilon=0.1):
    # 代表在计算损失时需要忽略的标签的索引, 例如当标签不存在于词表时, 置标签为-100, 在损失计算时不参与计算
    ignore_index = -100

    # log_probs: [32, 50, 30522]
    log_probs = -nn.functional.log_softmax(logits, dim=-1)

    # labels: [32, 50, 1] 将标签的维数扩展一维以匹配log_probs的维数
    if labels.dim() == log_probs.dim() - 1:
        labels = labels.unsqueeze(-1)

    # padding_mask: [32, 50, 1]
    padding_mask = labels.eq(ignore_index)

    # labels: [32, 50, 1] In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
    # will ignore them in any case.
    labels = torch.clamp(labels, min=0)

    # nll_loss: [32, 50, 1] 在log_probs的最后一个维度按labels收集概率
    nll_loss = log_probs.gather(dim=-1, index=labels)

    # smoothed_loss: [32, 50, 1] works for fp16 input tensor too, by internally upcasting it to fp32
    smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

    nll_loss.masked_fill_(padding_mask, 0.0)
    smoothed_loss.masked_fill_(padding_mask, 0.0)

    # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
    num_active_elements = padding_mask.numel() - padding_mask.long().sum()

    nll_loss = nll_loss.sum() / num_active_elements
    smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
    return (1 - epsilon) * nll_loss + epsilon * smoothed_loss
from abc import ABC, abstractmethod

from .pvp import PVPS
from .util import InputExample, InputFeatures


class Preprocessor(ABC):
    """
    预处理器 把 class:`InputExample` 转化为 class:`InputFeatures` 以便于模型可以使用数据集中的数据
    """

    def __init__(self, wrapper, task_name, pattern_id: int = 0, verbalizer_cal_type: str = 'avg'):
        """
        :param wrapper: the wrapper for the language model to use
        :param task_name: the name of the task/dataset
        :param pattern_id: the id of the PVP to be used
        """
        self.wrapper = wrapper
        self.pvp = PVPS[task_name](self.wrapper, pattern_id, verbalizer_cal_type)
        self.label_map = {label: i for i, label in enumerate(self.wrapper.config.label_list)}

        @abstractmethod
        def get_input_features(self, example: InputExample, labelled: bool, priming: bool = False,
                               **kwargs) -> InputFeatures:
            """Convert the given example into a set of input features"""
            pass


class MLMPreprocessor(Preprocessor):
    """Preprocessor for models pretrained using a masked language modeling objective (e.g., BERT)."""

    def get_input_features(self, example: InputExample, labelled: bool,
                           **kwargs) -> InputFeatures:

        input_ids, token_type_ids, labels = self.pvp.encode(example)

        attention_mask = [1] * len(input_ids)
        padding_length = self.wrapper.config.max_seq_length - len(input_ids)

        if padding_length < 0:
            raise ValueError(f"Maximum sequence length is too small, got {len(input_ids)} input ids")

        input_ids = input_ids + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        labels = labels + ([self.wrapper.tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        assert len(input_ids) == self.wrapper.config.max_seq_length
        assert len(attention_mask) == self.wrapper.config.max_seq_length
        assert len(token_type_ids) == self.wrapper.config.max_seq_length

        encode_label = example.encode_label

        if labelled:
            mlm_labels = self.pvp.get_mask_positions(input_ids)
            if self.wrapper.config.model_type == 'gpt2':
                # shift labels to the left by one
                mlm_labels.append(mlm_labels.pop(0))
        else:
            mlm_labels = [-1] * self.wrapper.config.max_seq_length

        return InputFeatures(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                             encode_label=encode_label, mlm_labels=mlm_labels, labels=labels)
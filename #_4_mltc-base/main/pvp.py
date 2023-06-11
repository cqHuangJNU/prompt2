from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch
from transformers import GPT2Tokenizer

from .util import InputExample,get_verbalization_ids


FilledPattern = Tuple[List[Union[str, Tuple[str, bool]]], List[Union[str, Tuple[str, bool]]]]


class PVP(ABC):
    """
    PVP类实现PVP结构, 对于不同的数据集会有不同的PVP实现.
    """

    def __init__(self, wrapper, pattern_id: int = 0, verbalizer_cal_type: str = 'avg'):
        """
        :param wrapper: the wrapper for the underlying language model
        :param pattern_id: the pattern id to use
        """
        self.wrapper = wrapper
        self.pattern_id = pattern_id
        self.verbalizer_cal_type = verbalizer_cal_type

        if self.wrapper.config.wrapper_type in ['mlm', 'plm']:
            self.mlm_logits_to_cls_logits_tensor = self._build_mlm_logits_to_cls_logits_tensor()

    # 根据verbalizer构建(label_num, max_num_verbalizer_len),对应的位置存单词的token_id, 没有单词的存-1
    def _build_mlm_logits_to_cls_logits_tensor(self):
        label_list = self.wrapper.config.label_list
        m2c_tensor = torch.ones([len(label_list), self.max_num_verbalizers], dtype=torch.long) * -1

        for label_idx, label in enumerate(label_list):
            verbalizers = self.verbalize(label)
            for verbalizer_idx, verbalizer in enumerate(verbalizers):
                verbalizer_id = get_verbalization_ids(verbalizer, self.wrapper.tokenizer, force_single_token=True)
                assert verbalizer_id != self.wrapper.tokenizer.unk_token_id, "verbalization was tokenized as <UNK>"
                m2c_tensor[label_idx, verbalizer_idx] = verbalizer_id
        return m2c_tensor

    @property
    def mask(self) -> str:
        """返回 LM 中定义的 mask token"""
        return self.wrapper.tokenizer.mask_token

    @property
    def mask_id(self) -> int:
        """返回 LM 中定义的 mask token id"""
        return self.wrapper.tokenizer.mask_token_id

    @property
    def max_num_verbalizers(self) -> int:
        """返回所有 verbalizers 数组最长的一个数组的长度"""
        return max(len(self.verbalize(label)) for label in self.wrapper.config.label_list)

    @property
    def label_list_id(self) -> List[int]:
        """获取标签集的token_id"""
        label_list = self.wrapper.config.label_list
        return [get_verbalization_ids(label, self.wrapper.tokenizer, force_single_token=True) for label in label_list]

    @staticmethod
    def shortenable(s):
        """Return an instance of this string that is marked as shortenable"""
        return s, True

    @staticmethod
    def _seq_length(parts: List[Tuple[str, bool]], only_shortenable: bool = False):
        return sum([len(x) for x, shortenable in parts if not only_shortenable or shortenable]) if parts else 0

    @staticmethod
    def _remove_last(parts: List[Tuple[str, bool]]):
        last_idx = max(idx for idx, (seq, shortenable) in enumerate(parts) if shortenable and seq)
        parts[last_idx] = (parts[last_idx][0][:-1], parts[last_idx][1])

    @abstractmethod
    def verbalize(self, label) -> List[str]:
        """
        给定一个 label, 返回对应的 verbalizations 数组.

        :param label: the label
        :return: the list of verbalizations
        """
        pass

    @abstractmethod
    def get_parts(self, example: InputExample) -> FilledPattern:
        """
        Given an input example, apply a pattern to obtain two text sequences (text_a and text_b) containing exactly one
        mask token (or one consecutive sequence of mask tokens for PET with multiple masks). If a task requires only a
        single sequence of text, the second sequence should be an empty list.

        :param example: the input example to process
        :return: Two sequences of text. All text segments can optionally be marked as being shortenable.
        """
        pass

    def get_mask_positions(self, input_ids: List[int]) -> List[int]:
        label_idx = input_ids.index(self.mask_id)
        labels = [-1] * len(input_ids)
        labels[label_idx] = 1
        return labels

    def convert_mlm_logits_to_cls_logits(self, mlm_labels: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        masked_logits = logits[mlm_labels >= 0] # (batch_size, vocab_size)
        cls_logits = torch.stack([self._convert_single_mlm_logits_to_cls_logits(ml, self.verbalizer_cal_type) for ml in masked_logits])
        return cls_logits

    def _convert_single_mlm_logits_to_cls_logits(self, logits: torch.Tensor, type) -> torch.Tensor:
        # (label_num, max_num_verbalizer_len), 对应的位置存单词的token_id, 没有单词的存-1
        m2c = self.mlm_logits_to_cls_logits_tensor.to(logits.device)
        # (label_num) 里面存每个verbalizer的长度
        filler_len = torch.tensor([len(self.verbalize(label)) for label in self.wrapper.config.label_list],
                                  dtype=torch.float)
        filler_len = filler_len.to(logits.device)

        # (label_num, filler_len[i]) 根据m2c里面存的token_id去cls_logits里面找对应logit
        cls_logits = logits[torch.max(torch.zeros_like(m2c), m2c)]
        # (label_num, max_num_verbalizer_len) 改形状
        cls_logits = cls_logits * (m2c > 0).float()
        if type == 'avg':
            # (label_num) 计算每个标签映射集的平均得分
            cls_logits = cls_logits.sum(axis=1) / filler_len
        elif type == 'max':
            # (label_num) 计算每个标签映射集的最大得分
            cls_logits, max_indices = torch.max(cls_logits, dim=1)
        elif type == 'sum':
            # (label_num) 计算每个标签映射集的总和得分
            cls_logits = cls_logits.sum(axis=1)
        return cls_logits

    def truncate(self, parts_a: List[Tuple[str, bool]], parts_b: List[Tuple[str, bool]], max_length: int):
        """Truncate two sequences of text to a predefined total maximum length"""
        total_len = self._seq_length(parts_a) + self._seq_length(parts_b)
        total_len += self.wrapper.tokenizer.num_special_tokens_to_add(bool(parts_b))
        num_tokens_to_remove = total_len - max_length

        if num_tokens_to_remove <= 0:
            return parts_a, parts_b

        for _ in range(num_tokens_to_remove):
            if self._seq_length(parts_a, only_shortenable=True) > self._seq_length(parts_b, only_shortenable=True):
                self._remove_last(parts_a)
            else:
                self._remove_last(parts_b)

    def encode(self, example: InputExample, labeled: bool = False) \
            -> Tuple[List[int], List[int], List[int]]:
        """
        Encode an input example using this pattern-verbalizer pair.

        :param example: the input example to encode
        :param labeled: if ``priming=True``, whether the label should be appended to this example
        :return: A tuple, 加入了特殊token的input, token_type_ids, labels只把inputs的MASK位置设为-100
        """
        tokenizer = self.wrapper.tokenizer
        text, _ = self.get_parts(example)

        kwargs = {'add_prefix_space': True} if isinstance(tokenizer, GPT2Tokenizer) else {}
        # 设置除了原始本文 其他的拼接文本不可以cut掉
        parts_a = [x if isinstance(x, tuple) else (x, False) for x in text]
        parts_a = [(tokenizer.encode(x, add_special_tokens=False, **kwargs), s) for x, s in parts_a if x]
        # 根据设置的max_seq_length 对parts_a进行切除 之后parts_a的长度为max_seq_length-2
        self.truncate(parts_a, _, max_length=self.wrapper.config.max_seq_length)
        tokens_a = [token_id for part, _ in parts_a for token_id in part]
        # 加上前后的特殊字符后 inputs的长度才为max_seq_length
        input_ids = tokenizer.build_inputs_with_special_tokens(tokens_a, None)

        token_type_ids = tokenizer.create_token_type_ids_from_sequences(tokens_a, None)

        labels = tokenizer.build_inputs_with_special_tokens(tokens_a, None)
        labels[input_ids.index(self.mask_id)] = -100

        return input_ids, token_type_ids, labels


class ReutersEnPVP(PVP):
    VERBALIZER = {
        '0': ['earn'],
        '1': ['acquire'],# '1': ['acq'],
        '2': ['exchange'],# '2': ['money-fx'],
        '3': ['grain'],
        '4': ['crude'],
        '5': ['trade'],
        '6': ['interest'],
        '7': ['ship'],
        '8': ['wheat'],
        '9': ['corn'],
        '10': ['dl'],# '10': ['dlr'],
        '11': ['supply'],# '11': ['money-supply'],
        '12': ['see'],# '12': ['oilseed'],
        '13': ['sugar'],
        '14': ['coffee'],
        '15': ['gross'],#'15': ['gnp'],
        '16': ['vegetable'], #'16': ['veg-oil'],
        '17': ['gold'],
        '18': ['bean'],#'18': ['soybean'],
        '19': ['natural'],#'19': ['nat-gas'],
        '20': ['payment'],#'20': ['bop'],
        '21': ['livestock'],
        '22': ['cpi'],
        '23': ['cocoa'],
        '24': ['reserves'],
        '25': ['meat'],#'25': ['carcass'],
        '26': ['jobs'],
        '27': ['copper'],
        '28': ['yen'],
        '29': ['rice'],
        '30': ['cotton'],
        '31': ['al'],#'31': ['alum'],
        '32': ['gas'],
        '33': ['steel'],#'33': ['iron-steel'],
        '34': ['ip'],#'34': ['ipi'],
        '35': ['barley'],
        '36': ['rubber'],
        '37': ['feed'],#'37': ['meal-feed'],
        '38': ['palm'],#'38': ['palm-oil'],
        '39': ['zinc'],
        '40': ['so'],#'40': ['sorghum'],
        '41': ['pet'],#'41': ['pet-chem'],
        '42': ['tin'],
        '43': ['silver'],
        '44': ['lead'],
        '45': ['w'],#'45': ['wpi'],
        '46': ['strategic'],#'46': ['strategic-metal'],
        '47': ['rap'],#'47': ['rapeseed'],
        '48': ['orange'],
        '49': ['soy'],#'49': ['soy-meal'],
        '50': ['retail'],
        '51': ['o'],#'51': ['soy-oil'],
        '52': ['fuel'],
        '53': ['hog'],
        '54': ['housing'],
        '55': ['heat'],
        '56': ['seed'],#'56': ['sunseed'],
        '57': ['lumber'],
        '58': ['income'],
        '59': ['lei'],
        '60': ['deutsche'],#'60': ['dmk'],
        '61': ['oath'],# '61': ['oat'],
        '62': ['tea'],
        '63': ['platinum'],
        '64': ['nickel'],
        '65': ['ground'],#'65': ['groundnut'],
        '66': ['rape'],#'66': ['rape-oil'],
        '67': ['cattle'],#'67': ['l-cattle'],
        '68': ['co'],#'68': ['coconut-oil'],
        '69': ['sun'],#'69': ['sun-oil'],
        '70': ['potato'],
        '71': ['nap'],#'71': ['naphtha'],
        '72': ['debt'],#'72': ['instal-debt'],
        '73': ['prop'],#'73': ['propane'],
        '74': ['coconut'],
        '75': ['jet'],
        '76': ['rate'],#'76': ['nzdlr'],
        '77': ['cpu'],
        '78': ['kernel'],#'78': ['palmkernel'],
        '79': ['d'],#'79': ['dfl'],
        '80': ['rand'],
        '81': ['won'],#'81': ['nkr'],
        '82': ['pal'],#'82': ['palladium'],
        '83': ['cake'],#'83': ['copra-cake'],
        '84': ['oil'],#'84': ['cotton-oil'],
        '85': ['meal'],#'85': ['sun-meal'],
        '86': ['nut'],#'86': ['groundnut-oil'],
        '87': ['rye'],
        '88': ['cast'],#'88': ['castor-oil'],
        '89': ['lin']#'89': ['lin-oil']
        }

    def get_parts(self, example: InputExample) -> FilledPattern:

        text = self.shortenable(example.text)

        if self.pattern_id == 0:
            return [self.mask, ':', text], []
        elif self.pattern_id == 1:
            return [self.mask, 'News:', text], []
        elif self.pattern_id == 2:
            return [text, '(', self.mask, ')'], []
        elif self.pattern_id == 3:
            return [text, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text], []
        else:
            raise ValueError("未匹配到id为{}的Pattern".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return ReutersEnPVP.VERBALIZER[label]


class AAPDEnPVP(PVP):
    VERBALIZER = {
    }
    def get_parts(self, example: InputExample) -> FilledPattern:

        text = self.shortenable(example.text)

        if self.pattern_id == 0:
            return [self.mask, ':', text], []
        elif self.pattern_id == 1:
            return [self.mask, 'News:', text], []
        elif self.pattern_id == 2:
            return [text, '(', self.mask, ')'], []
        elif self.pattern_id == 3:
            return [text, '(', self.mask, ')'], []
        elif self.pattern_id == 4:
            return ['[ Category:', self.mask, ']', text], []
        elif self.pattern_id == 5:
            return [self.mask, '-', text], []
        else:
            raise ValueError("未匹配到id为{}的Pattern".format(self.pattern_id))

    def verbalize(self, label) -> List[str]:
        return ReutersEnPVP.VERBALIZER[label]


PVPS = {
    'reuters_en': ReutersEnPVP,
    'aapd': AAPDEnPVP
}
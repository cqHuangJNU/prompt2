import csv
import os
import random
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import List
import logging
from .util import InputExample

# 配置日志记录器
logging.basicConfig(filename='runtime-logs/task.txt', level=logging.DEBUG, filemode='w')

def _shuffle_and_restrict(examples: List[InputExample], num_examples: int, seed: int = 42) -> List[InputExample]:
    """
    对传入的数据集乱序并返回num_examples条文本数据.

    :param examples: the examples to shuffle and restrict
    :param num_examples: the maximum number of examples
    :param seed: the random seed for shuffling
    :return: the first ``num_examples`` elements of the shuffled list
    """
    if 0 < num_examples < len(examples):
        random.Random(seed).shuffle(examples)
        examples = examples[:num_examples]
    return examples

class DataProcessor(ABC):
    """
    Abstract class that provides methods for loading training, testing, development and unlabeled examples for a given
    task
    """

    @abstractmethod
    def get_train_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the train set."""
        pass

    @abstractmethod
    def get_dev_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the dev set."""
        pass

    @abstractmethod
    def get_test_examples(self, data_dir) -> List[InputExample]:
        """Get a collection of `InputExample`s for the test set."""
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """Get the list of labels for this data set."""
        pass

class ReutersEnProcessor(DataProcessor):
    """Processor for the Example data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.csv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.csv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(os.path.join(data_dir, "test.csv"), "test")

    def get_labels(self):
        return ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21',
                '22','23','24','25','26','27','28','29','30','31','32','33','34','35','36','37','38','39','40','41',
                '42','43','44','45','46','47','48','49','50','51','52','53','54','55','56','57','58','59','60','61',
                '62','63','64','65','66','67','68','69','70','71','72','73','74','75','76','77','78','79','80','81',
                '82','83','84','85','86','87','88','89']

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        LABELS = ['earn', 'acq', 'money-fx', 'grain', 'crude', 'trade', 'interest', 'ship', 'wheat', 'corn', 'dlr',
         'money-supply', 'oilseed', 'sugar', 'coffee', 'gnp', 'veg-oil', 'gold', 'soybean', 'nat-gas', 'bop',
         'livestock', 'cpi', 'cocoa', 'reserves', 'carcass', 'jobs', 'copper', 'yen', 'rice', 'cotton', 'alum', 'gas',
         'iron-steel', 'ipi', 'barley', 'rubber', 'meal-feed', 'palm-oil', 'zinc', 'sorghum', 'pet-chem', 'tin',
         'silver', 'lead', 'wpi', 'strategic-metal', 'rapeseed', 'orange', 'soy-meal', 'retail', 'soy-oil', 'fuel',
         'hog', 'housing', 'heat', 'sunseed', 'lumber', 'income', 'lei', 'dmk', 'oat', 'tea', 'platinum', 'nickel',
         'groundnut', 'rape-oil', 'l-cattle', 'coconut-oil', 'sun-oil', 'potato', 'naphtha', 'instal-debt', 'propane',
         'coconut', 'jet', 'nzdlr', 'cpu', 'palmkernel', 'dfl', 'rand', 'nkr', 'palladium', 'copra-cake', 'cotton-oil',
         'sun-meal', 'groundnut-oil', 'rye', 'castor-oil', 'lin-oil']

        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, content = row
                # 解析字符串格式的标签
                word_label = [_ for _ in re.compile(r"'(.*?)'").findall(label)]
                #  并转化为索引值数组
                index_label = [LABELS.index(_) for _ in word_label]
                # 转化为独热向量
                encode_label = [0] * len(LABELS)
                for i in index_label:
                    encode_label[i] = 1
                # 定制id
                guid = "%s-%s" % (set_type, idx)
                # 文本
                content = content.replace('\\', ' ')

                example = InputExample(guid=guid, text=content, word_label=word_label, index_label=index_label, encode_label=encode_label)
                examples.append(example)

        return examples


class AAPDEnProcessor(DataProcessor):
    """Processor for the Example data set."""

    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "train.tsv"), "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, "dev.tsv"), "dev")

    def get_test_examples(self, data_dir) -> List[InputExample]:
        return self._create_examples(os.path.join(data_dir, "test.tsv"), "test")

    def get_labels(self):
        return ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18','19', '20', '21','22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38','39', '40', '41','42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54']

    @staticmethod
    def _create_examples(path: str, set_type: str) -> List[InputExample]:
        LABELS = ["cs.it","math.it","cs.lg","cs.ai","stat.ml","cs.ds","cs.si","cs.dm","physics.soc-ph","cs.lo","math.co","cs.cc","math.oc","cs.ni","cs.cv","cs.cl","cs.cr","cs.sy","cs.dc","cs.ne","cs.ir","quant-ph","cs.gt","cs.cy","cs.pl","cs.se","math.pr","cs.db","cs.cg","cs.na","cs.hc","math.na","cs.ce","cs.ma","cs.ro","cs.fl","math.st","stat.th","cs.dl","cmp-lg","cs.mm","cond-mat.stat-mech","cs.pf","math.lo","stat.ap","cs.ms","stat.me","cs.sc","cond-mat.dis-nn","q-bio.nc","physics.data-an","nlin.ao","q-bio.qm","math.nt"]

        examples = []

        with open(path) as f:
            reader = csv.reader(f, delimiter='\t')
            for idx, row in enumerate(reader):
                label, content = row
                # 把字符串转化为独热向量
                encode_label = [int(x) for x in list(label)]
                # 定制id
                guid = "%s-%s" % (set_type, idx)
                # 文本
                content = content.replace('\\', ' ')

                example = InputExample(guid=guid, text=content, word_label=None, index_label=None,
                                       encode_label=encode_label)
                examples.append(example)

        return examples

PROCESSORS = {
    "reuters_en": ReutersEnProcessor,
    "aapd_en": AAPDEnProcessor
}

TRAIN_SET = "train"
DEV_SET = "dev"
TEST_SET = "test"

SET_TYPES = [TRAIN_SET, DEV_SET, TEST_SET]

def load_examples(task, data_dir: str, set_type: str, *_, num_examples: int = None,  seed: int = 42) -> List[InputExample]:
    """Load examples for a given task."""
    assert (num_examples is not None) , "参数 'num_examples' 必选设置."

    processor = PROCESSORS[task]()

    ex_str = f"num_examples={num_examples}"
    logging.info(f"加载数据集 {data_dir} 配置为 ({ex_str}, set_type={set_type})")

    if set_type == DEV_SET:
        examples = processor.get_dev_examples(data_dir)
    elif set_type == TEST_SET:
        examples = processor.get_test_examples(data_dir)
    elif set_type == TRAIN_SET:
        examples = processor.get_train_examples(data_dir)
    else:
        raise ValueError(f"'set_type' 必须是 {SET_TYPES} 中的一个, got '{set_type}' instead")

    if num_examples is not None:
        examples = _shuffle_and_restrict(examples, num_examples, seed)

    label_distribution = Counter(tuple(_) for _ in [example.index_label for example in examples])
    logging.info(f"返回 {len(examples)} 条 {set_type} 样本, 标签集与文本数为: {sorted(label_distribution.items(), key=lambda x: x[1], reverse=True)}")

    return examples
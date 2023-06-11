import json
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, precision_score, hamming_loss, f1_score, \
    recall_score
from transformers import Trainer, TrainingArguments, IntervalStrategy, \
    EarlyStoppingCallback

from main.system import SYSTEM
from main.task import load_examples, PROCESSORS
from main.util import LabelSmoothLoss
from main.wrapper import WrapperConfig, TransformerModelWrapper

SYSTEM.config_system()

# Python自带的日志模块 有5种级别
logging.basicConfig(level=logging.INFO)


# 构建trainer
def create_trainer(wrapper, train_dataset, val_dataset, checkpoint_dir, batch_size, epoch_num):
    tokenizer, model = wrapper.tokenizer, wrapper.model

    args = TrainingArguments(
        checkpoint_dir,  # 模型预测和检查点的输出目录
        save_strategy=IntervalStrategy.STEPS,  # 模型保存策略
        save_steps=10,  # 每n步保存一次模型  1步表示一个batch训练结束
        evaluation_strategy=IntervalStrategy.STEPS,
        eval_steps=10,
        overwrite_output_dir=True,  # 设置overwrite_output_dir参数为True，表示覆盖输出目录中已有的模型文件
        report_to=["tensorboard"],  # 设置可视化 控制台命令为 tensorboard --logdir=logs
        logging_dir='./logs',  # 可视化数据文件存储地址
        log_level="warning",
        logging_steps=10,  # 每n步保存一次评价指标  1步表示一个batch训练结束 | 还控制着控制台的打印频率 每n步打印一下评价指标 | n过大时，只会保存最后一次的评价指标
        disable_tqdm=False,  # 是否不显示数据训练进度条
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epoch_num,
        weight_decay=0.01,
        save_total_limit=10,
        label_names=['labels','encode_label'],
        load_best_model_at_end=True,
        remove_unused_columns= False, # 在compute_loss 时需要额外输入来构建verbalizer的平均分数
        include_inputs_for_metrics= True # compute_metrics 时需要原始输出来计算[MASK]的位置
    )

    # 重写评价指标计算
    def compute_metrics(pred):
        # 真实标签
        true_label = pred.label_ids[1]
        # 预测标签
        pred_label = np.zeros_like(true_label)

        # 过滤阈值
        alpha = 0.5

        # 标签集的token_id
        LABELS_TKONE_ID = wrapper.preprocessor.pvp.label_list_id

        # 求布尔向量 方便从logits、labels中提取mask位置上的数据
        mlm_bool = pred.inputs == wrapper.preprocessor.pvp.mask_id
        # MASK位置的预测数据 (batch_size, vocab_size)
        # sigmoid = nn.Sigmoid()
        # preds = sigmoid(torch.from_numpy(pred.predictions[mlm_bool]))
        preds = 1 / (1 + np.exp(-pred.predictions[mlm_bool]))

        # 构建预测标签
        for i in range(len(preds)):
            # 从标签里面构建
            indices = np.squeeze(np.where(preds[i] > alpha))
            for _ in indices:
                if _ in LABELS_TKONE_ID:
                    pred_label[i, LABELS_TKONE_ID.index(_)] = 1
            # 从Verbalizer的映射集构建
            for j,x in enumerate(wrapper.preprocessor.pvp.mlm_logits_to_cls_logits_tensor):
                for _ in indices:
                    if _ in x:
                        pred_label[i, LABELS_TKONE_ID.index(wrapper.preprocessor.pvp.label_list_id[j])] = 1

        precision, recall, f1, _ = precision_recall_fscore_support(true_label, pred_label, average='weighted')

        ham = hamming_loss(np.array(true_label), np.array(pred_label))
        acc = accuracy_score(true_label, pred_label)
        return {'acc': acc, 'ham': ham, 'f1': f1, 'pre': precision, 'rec': recall}

    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=3,  # 如果3个epoch的验证集性能没有提升，则停止训练
        early_stopping_threshold=1e-9,  # 验证集的性能提高不到0.001时也停止训练
    )

    class CustomTrainer(Trainer):
        # 重写loss计算
        def compute_loss(self, model, inputs, return_outputs=False):
            # 运行模型
            new_inputs = {k:v for k, v in inputs.items() if k not in ['mlm_labels', 'encode_label']}
            outputs = model(**new_inputs)

            # 计算verbalizer中每个标签的得分
            prediction_scores = wrapper.preprocessor.pvp.convert_mlm_logits_to_cls_logits(inputs['mlm_labels'], outputs['logits'])

            # 为mask位置上的logits重新赋值
            for text_idx in range(len(outputs['logits'])):
                mlm_labels = inputs['mlm_labels'][text_idx]
                for i, label_token_id in enumerate(wrapper.preprocessor.pvp.label_list_id):
                    outputs['logits'][text_idx, mlm_labels.tolist().index(1), label_token_id] = prediction_scores[text_idx, i]

            # 多标签分类损失 BCEWithLogitsLoss sigmoid标签内的logit
            loss_fuc = nn.BCEWithLogitsLoss()
            mlm_loss = loss_fuc(prediction_scores, inputs['encode_label'].float())

            # 多标签分类损失 零边界Loss
            # sigmoid = nn.Sigmoid()
            # prediction_scores = sigmoid(prediction_scores)
            # Np = prediction_scores * inputs['encode_label']
            # Nm = prediction_scores * (1 - inputs['encode_label'])
            # mlm_loss = torch.log(1 + torch.exp(Np)).sum() + torch.log(1 + torch.exp(-Nm)).sum()

            # 计算其他位置上的损失
            labels = inputs['labels']
            auxiliary_loss = LabelSmoothLoss(outputs['logits'], labels, epsilon=0) # 当epsilon=0会完全退化为交叉熵损失

            # 计算联合损失
            loss = 0.8 * mlm_loss + 0.2 * auxiliary_loss

            # 计算联合损失，只适用于单标签分类
            #loss = LabelSmoothLoss(outputs['logits'], inputs['labels'], epsilon=0.1) # 当epsilon=0会完全退化为交叉熵损失

            return (loss, outputs) if return_outputs else loss

    trainer = CustomTrainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        # tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping],  # 添加EarlyStoppingCallback回调函数
    )
    return trainer

def main():
    # ##
    # @通用配置
    # checkpoint_dir: 模型保存地址. 不需要修改.
    # dataset_name: 数据集名称. 从DATASETS里面选, 方便修改, 而不用老是写字符串.
    # model_type: 需要使用的模型. 是在wrapper.py中提供的, optional: ['bert','roberta','xlm-roberta','xlnet','albert','gpt2']
    # model_name_or_path: 模型对应的文件地址或版本名称. eg: 'bert-base-chinese'
    # ##
    DATASETS = ['reuters_en','aapd_en']
    MODELS = ['bert','roberta','xlm-roberta','albert']
    MODELPATHS = [
        '../@_PLMs/bert/bert-base-chinese','../@_PLMs/bert/bert-base-uncased',
        '../@_PLMs/roberta/roberta-base',
        '../@_PLMs/xlm-roberta/xlm-roberta-base',
        '../@_PLMs/albert/albert-base-v2'
    ]
    checkpoint_dir = "./results"
    dataset_name = DATASETS[0]
    model_type= MODELS[0]
    model_name_or_path=  MODELPATHS[1]
    # ##
    # @数据集加载器
    # task_name: 任务/数据集名称. 需要指定才能加载数据集对应的DataProcessor和PVP. 1)在加载数据集的时候, 会在task.py中根据task_name找到
    # 对应的数据加载类 2)在初始化Wrapper时, 会初始化三种Preprocessor之一, Preprocessor又会创建PVP, 不同的task_name对应不同的PVP类.
    # data_dir: 数据集对应的目录.
    # @wrapper配置器
    # wrapper_type: 包装类的类型. optional: 'mlm','plm','sequence_classifier'
    # task_name: 任务/数据集名称. 介绍同上.
    # max_seq_length: 文本最大长度.
    # label_list: 标签集数组, 要与PVP/DataProcessor中保持对应.
    # pattern_id: prompt模板编号.
    # verbalizer_cal_type: verbalizer映射集中多个token的logit计算方式. optional: 'avg','sum','max'
    # ##
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)[dataset_name]
    # 自动赋值label_list
    processor = PROCESSORS[config['task_name']]()
    config['label_list'] = processor.get_labels()
    # ##
    # Step 1: Load three datasets' examples.
    # Descript: 调用task.load_examples, 根据传递的task_name(dataset)区分不同数据集的DataProcessor, 将数据加载成 List[InputExample] 格式,
    # 不同数据集的 DataProcessor 需要不同的实现. task.load_examples 有2种数据集加载策略
    # Params:
    #    num_examples: 数据集乱序后取num_examples条数据出来. 只有当num_examples严格小于该数据集总条数时才会乱序.
    # ##
    train_data = load_examples(config['task_name'], config['data_dir'], set_type="train", num_examples = 100)
    dev_data = load_examples(config['task_name'], config['data_dir'], set_type="dev", num_examples = 20)
    test_data = load_examples(config['task_name'], config['data_dir'], set_type="test", num_examples = 20)

    # ##
    # Step 2: Create a wrapper configuration.
    # Descript: 定义wrapper配置类, 然后构建模型包装类, 该包装类包装了各种各样的PLM. 类中有tokenizer/model/preprocessor. 其中preprocessor是
    # 预处理器, 根据传入的wrapper_type区分是plm/mlm/sequence_classifier模型之一, 根据传入的task_name区分是哪个数据集, 以便创建对应的PVP.
    # ##
    wrapper_config = WrapperConfig(
        model_type= model_type,
        model_name_or_path= model_name_or_path,
        wrapper_type= config['wrapper_type'],
        task_name= config['task_name'],
        max_seq_length= config['max_seq_length'],
        label_list= config['label_list'],
        pattern_id= config['pattern_id'],
        verbalizer_cal_type= config['verbalizer_cal_type']
    )
    wrapper = TransformerModelWrapper(wrapper_config)

    for name, param in wrapper.model.named_parameters():
        # if 'cls' not in name and 'embeddings' not in name:
        if 'cls' not in name :
            param.requires_grad = False
            
    # print(model)
    print(f'模型有 {sum(p.numel() for p in wrapper.model.parameters() if p.requires_grad):,} 个训练参数')
    """
    # Step 3: Transfer `InputExample` to `InputFeatures` by Preprocessor.
    # Descript:
    """
    get_parts = wrapper.preprocessor.pvp.get_parts
    wrapper.preprocessor.pvp.get_parts = lambda example: (get_parts(example)[0] + get_parts(example)[1], [])
    #wrapper.preprocessor.pvp.convert_mlm_logits_to_cls_logits = lambda mask, x, _=None: x[mask >= 0]

    train_dataset= wrapper._generate_dataset(train_data)
    val_dataset = wrapper._generate_dataset(dev_data)
    test_dataset = wrapper._generate_dataset(test_data)


    trainer = create_trainer(wrapper, train_dataset, val_dataset, checkpoint_dir, config['batch_size'], config['epoch_num'])
    trainer.train()

    # for obj in trainer.state.log_history:
    #     print(obj)

    # 在测试集上评估模型性能
    test_result = trainer.predict(test_dataset)

    # 输出测试集上的性能指标
    print("在测试集的评价指标:")
    print(test_result.metrics)

    k = 15
    n = 30
    print([config['label_list'][x['label_idx']] for x in test_dataset])

    print("===================================\n前",n,"条文本预测出的前",k,"个MASK:")
    mask_index = np.concatenate([np.where(row['input_ids'] == wrapper.preprocessor.pvp.mask_id)[0] for row in test_dataset]) # (batch_size) 取出每条文本中[MASK]位置的索引
    # print(mask_index)
    b_indices = np.array([(i, j) for i, j in enumerate(mask_index)])
    P = test_result.predictions[b_indices[:, 0], b_indices[:, 1], :]
    top_k_index = np.argsort(-P, kind='heapsort')[:, :k]
    top_k_valus = np.array([P[i, top_k_index[i]] for i in range(top_k_index.shape[0])])
    top_k_logi = nn.functional.softmax(torch.from_numpy(top_k_valus), dim=1)
    for i,x in enumerate(top_k_index):
        if i < n:
            print("真实标签:",config['label_list'][test_dataset[i]['label_idx']])
            print("预测MASK:",wrapper.tokenizer.convert_ids_to_tokens(x))
            print("预测概率:",top_k_logi[i])


if __name__ == '__main__':
    main()



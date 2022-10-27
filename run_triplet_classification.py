import logging
import os

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from model.bert_model import BertPoolingForTripletPrediction
from model.trainer import KGCTrainer
from model.data_processor import DictDataset, KGProcessor
from model.roberta_model import RobertaPoolingForTripletPrediction
from model.data_collator import PoolingCollator
from model.utils import DataArguments, ModelArguments

logger = logging.getLogger(__name__)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def compute_metrics(task_name, preds, labels):
    assert len(preds) == len(labels)
    if task_name == 'kg':
        return {'acc': simple_accuracy(preds, labels)}
    else:
        raise KeyError(task_name)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    (model_args, data_args, training_args) = parser.parse_args_into_dataclasses()
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and (not training_args.overwrite_output_dir)
    ):
        raise ValueError(
            f'Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.')
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN
    )
    logger.warning(
        'Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s',
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16
    )
    logger.info('Training/evaluation parameters %s', training_args)
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.model_cache_dir)
    is_world_process_zero = training_args.local_rank == -1 or torch.distributed.get_rank() == 0
    processor = KGProcessor(data_args, tokenizer, is_world_process_zero)
    (train_data, dev_data, test_data) = processor.get_dataset(training_args)
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.model_cache_dir
    )
    if not hasattr(config, 'real_vocab_size'):
        config.real_vocab_size = config.vocab_size
    if model_args.pos_weight is not None:
        model_args.pos_weight = torch.tensor([model_args.pos_weight]).to(training_args.device)
    if model_args.pooling_model:
        print('using pooling model!')
        if tokenizer.__class__.__name__.startswith("Roberta"):
            tokenizer_cls = RobertaPoolingForTripletPrediction
        elif tokenizer.__class__.__name__.startswith("Bert"):
            tokenizer_cls = BertPoolingForTripletPrediction
        else:
            raise NotImplementedError()
        model = tokenizer_cls.from_pretrained(
            model_args.model_name_or_path,
            margin=data_args.margin,
            from_tf=bool('.ckpt' in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.model_cache_dir,
            pos_weight=model_args.pos_weight,
            text_loss_weight=model_args.text_loss_weight
        )
        data_collator = PoolingCollator(tokenizer)
    else:
        raise NotImplementedError()
    trainer = KGCTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_data,
        eval_dataset=dev_data,
        prediction_loss_only=True
    )
    if data_args.group_shuffle:
        print('using group shuffle')
        trainer.use_group_shuffle(data_args.num_neg)
    if training_args.do_train:
        model_path = (
            model_args.model_name_or_path
            if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.train(model_path=model_path)
        trainer.save_model()
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)
    if training_args.do_predict:
        label_map = {'-1': 0, '1': 1}
        trainer.model.set_predict_mode()
        trainer.prediction_loss_only = False
        trainer.data_collator.set_predict_mode()
        (dev_triples, dev_labels) = processor.get_dev_triples(return_label=True)
        dev_labels = np.array([label_map[l] for l in dev_labels], dtype=int)
        (_, tmp_features) = processor._create_examples_and_features(dev_triples)
        all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
        all_pos_indicator = torch.tensor([f.pos_indicator for f in tmp_features], dtype=torch.long)
        eval_data = DictDataset(input_ids=all_input_ids, pos_indicator=all_pos_indicator)
        trainer.data_collator.predict_mask_part = 0
        preds = trainer.predict(eval_data).predictions
        mean_dev = np.mean(preds)
        print('mean_dev: ', mean_dev)
        a = -5
        b = 5
        max_acc = 0
        for i in range(1000):
            m = (b - a) / 1000 * i + a
            tmp_preds = preds - m
            acc = np.mean((tmp_preds > 0).astype(int) == dev_labels)
            if acc > max_acc:
                max_acc = acc
                max_m = m
        print('max acc: ', max_acc)
        print('max m: ', max_m)
        mean_dev = max_m
        (test_triples, test_labels) = processor.get_test_triples(return_label=True)
        test_labels = np.array([label_map[l] for l in test_labels], dtype=int)
        (_, tmp_features) = processor._create_examples_and_features(test_triples)
        all_input_ids = torch.tensor([f.input_ids for f in tmp_features], dtype=torch.long)
        all_pos_indicator = torch.tensor([f.pos_indicator for f in tmp_features], dtype=torch.long)
        eval_data = DictDataset(input_ids=all_input_ids, pos_indicator=all_pos_indicator)
        preds = trainer.predict(eval_data).predictions
        preds = preds - mean_dev
        acc = np.mean((preds > 0).astype(int) == test_labels)
        print('test acc: ', acc)


if __name__ == '__main__':
    main()

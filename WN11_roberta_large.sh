export EXP_NAME=WN11_roberta-large
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export EXP_ROOT=exp_root
export MODEL_CACHE_DIR=cache

mkdir -p ${EXP_ROOT}/cache_${EXP_NAME}
python run_triplet_classification.py \
--do_train \
--do_eval \
--do_predict \
--data_dir ./data/WN11 \
--per_device_train_batch_size 16 \
--per_device_eval_batch_size 32 \
--data_cache_dir ${EXP_ROOT}/cache_${EXP_NAME} \
--model_cache_dir ${MODEL_CACHE_DIR} \
--model_name_or_path roberta-large \
--pooling_model \
--num_neg 1 \
--only_corrupt_entity \
--margin 7 \
--no_mid \
--max_seq_length 192 \
--learning_rate 2e-5 \
--adam_epsilon 1e-6 \
--num_train_epochs 3 \
--output_dir ${EXP_ROOT}/out_${EXP_NAME} \
--gradient_accumulation_steps 1 \
--save_total_limit 2 \
--save_steps 1760 \
--warmup_steps 528 \
--weight_decay 0.1 \
--text_loss_weight 0.0 \
--test_ratio 1. \
--overwrite_output_dir

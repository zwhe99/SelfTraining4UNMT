NGPU=8
PTM=/path/to/pretrained/model
DUMP_PATH=/path/to/this/experiment
DATA=/path/to/processed/data
EXP_NAME={experiment name}
TRAIN_SCRIPT=../train.py

python3 -m torch.distributed.launch --nproc_per_node=$NGPU ${TRAIN_SCRIPT} \
    --optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
    --lgs en-fr \
    --max_epoch 70 \
    --exp_id main \
    --lambda_ae 0:1,100000:0.1,300000:0 \
    --bt_steps en-fr-en,fr-en-fr \
    --attention_dropout 0.1 \
    --save_periodic 5 \
    --word_dropout 0.1 \
    --encoder_only false \
    --word_blank 0.1 \
    --exp_name $EXP_NAME \
    --gelu_activation true \
    --reload_model $PTM,$PTM \
    --dropout 0.1 \
    --epoch_size 200000 \
    --ae_steps en,fr \
    --n_layers 6 \
    --batch_size 32 \
    --validation_metrics valid_en-fr_mt_bleu,valid_fr-en_mt_bleu \
    --attention_setting v0 \
    --eval_bleu true \
    --n_heads 8 \
    --emb_dim 1024 \
    --word_shuffle 3 \
    --data_path $DATA \
    --bptt 256 \
    --tokens_per_batch 2500 \
    --exp_id main \
    --dump_path $DUMP_PATH \
    --accumulate_gradients 4 \
    --amp 0
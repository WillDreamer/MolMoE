#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:
# MODEL_VERSION=phi3-mini
# PROMPT_VERSION=phi

PROMPT_VERSION="llama3"
MODEL_VERSION="llama3-1b"

GRAPH_TOWER="himol"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="checkpoints/graphMVP.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="checkpoints/molecule_model.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="/root/autodl-tmp/MoleculeMoE/MolMoE/checkpoints/himol_encoder.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="_checkpoints/pretrain"
DATA_PATH="/root/autodl-tmp/MoleculeMoE/MolMoE/chem_data_cleaned.csv"
PROJECTOR="multilevel_type2"
HEAD="gap"
TASK="pub_chem"
MODEL_PATH="/root/autodl-tmp/MolMoE/downloads/Llama-3.2-1B-Instruct"
REMARK="projector-test"

deepspeed main.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --model_name_or_path $MODEL_PATH \
    --base_model $MODEL_PATH \
    --language_backbone_name $MODEL_VERSION \
    --training_recipe stage1 \
    --version $PROMPT_VERSION \
    --data_path $DATA_PATH \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --head_type $HEAD \
    --projector_aux_loss_coeff 0.1 \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/llava-$TASK-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --moe_enable False \
    --split_eval False \
    --eval_on_start False \
    --logging_dir /root/tf-logs/llava-$TASK-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
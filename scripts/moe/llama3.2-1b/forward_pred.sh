#!/bin/bash

# Uncomment and set the following variables correspondingly to run this script:

# PROMPT_VERSION="phi3"
# MODEL_VERSION="phi3-mini"

PROMPT_VERSION="llama3"
MODEL_VERSION="llama3-1b"

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="checkpoints/graphMVP.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="/root/autodl-tmp/MolMoE/downloads/moleculestm.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="/root/autodl-tmp/MoleculeMoE/MolMoE/checkpoints/himol_encoder.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="_checkpoints/moe"
TASK="forward_pred"
PROJECTOR="multilevel_type1"
BASE_MODEL="/root/autodl-tmp/MolMoE/downloads/Llama-3.2-1B-Instruct"
REMARK="test"


CHECKPOINT_FOLDER_PREFIX="_checkpoints/moe"
TASK="forward_pred"


deepspeed main.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --training_recipe stage3 \
    --model_name_or_path $BASE_MODEL \
    --base_model $BASE_MODEL \
    --stage2_model /root/autodl-tmp/MolMoE/_checkpoints/basline/lora/forward_pred-llava-moleculestm-llama3-1b-naive_linear-/checkpoint-19430 \
    --version $PROMPT_VERSION \
    --data_path /root/autodl-tmp/MolMoE/data_files/forward_reaction_prediction.json \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "no" \
    --split_eval False \
    --eval_on_start False \
    --save_total_limit 1 \
    --learning_rate 4e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.06 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --logging_dir /root/tf-logs/$TASK-stage3-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --moe_mode second_quarter \
    --ep_size 1 \
    --num_experts 3 \
    --use_residual False \
    --router_aux_loss_coef 0.01 \
    --is_training True \
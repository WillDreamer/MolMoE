#!/bin/bash

# PROMPT_VERSION="phi3"
# MODEL_VERSION="phi3-mini"

PROMPT_VERSION="llama3"
MODEL_VERSION="llama3-1b"

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="checkpoints/graphMVP.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="/home/whx/MolMoE/checkpoints/moleculestm.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="/home/whx/MolMoE/checkpoints/himol_encoder.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="_checkpoints/conflict"
TASK="forward_pred/retrosynthesis/reagent_pred/property_pred/molcap"
# TASK="reagent_pred/property_pred"
PROJECTOR="naive_linear"
# REMARK="all-task"
REMARK="all-task-rslora"


deepspeed --include localhost:2,3,4 --master_port 29504 experiment_tools/conflict_test.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --training_recipe stage2 \
    --model_name_or_path /home/whx/MolMoE/checkpoints/Llama-3.2-1B-Instruct \
    --base_model /home/whx/MolMoE/checkpoints/Llama-3.2-1B-Instruct \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /home/whx/MolMoE/Molecule-oriented_Instructions \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter /home/whx/MolMoE/checkpoints/mm_projector.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/lora/conflict-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --num_train_epochs 15 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --stop_epoch 10 \
    --eval_strategy "steps" \
    --eval_steps 5000 \
    --split_eval True \
    --eval_on_start False \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 8e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.0075 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --report_to tensorboard \
    --moe_enable False \
    --logging_dir /home/whx/MolMoE/tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --is_training True \
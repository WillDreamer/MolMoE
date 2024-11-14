#!/bin/bash

# PROMPT_VERSION="phi3"
# MODEL_VERSION="phi3-mini"
cd /wanghaixin/MolMoE

PROMPT_VERSION="llama3"
MODEL_VERSION="llama3-3b"

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="checkpoints/graphMVP.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="/wanghaixin/MolMoE/checkpoints/moleculestm.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="/wanghaixin/MolMoE/checkpoints/himol_encoder.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="_checkpoints/conflict"
TASK="forward_pred/retrosynthesis/reagent_pred/property_pred/molcap"
# TASK="reagent_pred/property_pred"
PROJECTOR="naive_linear"
# REMARK="all-task"
REMARK="all-task-1.5"


/root/anaconda3/bin/deepspeed --master_port 29503 /wanghaixin/MolMoE/experiment_tools/conflict_test.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --training_recipe stage2 \
    --model_name_or_path /wanghaixin/MolMoE/checkpoints/Llama-3.2-3B-Instruct \
    --base_model /wanghaixin/MolMoE/checkpoints/Llama-3.2-3B-Instruct \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /wanghaixin/MolMoE/Molecule-oriented_Instructions \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter /wanghaixin/MolMoE/_checkpoints/pretrain/basline/llava-pub_chem-moleculestm-llama3-3b-naive_linear-/checkpoint-61385/mm_projector.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/lora/conflict-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
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
    --dataloader_num_workers 16 \
    --report_to tensorboard \
    --moe_enable False \
    --logging_dir /wanghaixin/MolMoE/tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --is_training True \
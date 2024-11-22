#!/bin/bash

# PROMPT_VERSION="phi3"
# MODEL_VERSION="phi3-mini"

PROMPT_VERSION="llama3"
MODEL_VERSION="llama3-3b"

cd /wanghaixin/MolMoE

GRAPH_TOWER="moleculestm"
if [ "$GRAPH_TOWER" == "graphmvp" ]; then
    INIT_CHECKPOINT_GNN="checkpoints/graphMVP.pth"
elif [ "$GRAPH_TOWER" == "moleculestm" ]; then
    INIT_CHECKPOINT_GNN="/wanghaixin/MolMoE/checkpoints/moleculestm.pth"
elif [ "$GRAPH_TOWER" == "himol" ]; then
    INIT_CHECKPOINT_GNN="/root/autodl-tmp/MoleculeMoE/MolMoE/checkpoints/himol_encoder.pth"
else
    echo "Not supported graph tower"
fi

CHECKPOINT_FOLDER_PREFIX="_checkpoints/spin"
TASK="forward_pred/retrosynthesis/reagent_pred/property_pred/molcap"
PROJECTOR="naive_linear"
REMARK="spin-training_lr5e-6"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
/root/anaconda3/bin/deepspeed /wanghaixin/MolMoE/post_train/train_spin.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --training_recipe lora2lora \
    --stage2_model /wanghaixin/MolMoE/_checkpoints/conflict/lora/conflict-llava-moleculestm-llama3-3b-naive_linear-all-task-rslora-15epoch/checkpoint-161850 \
    --model_name_or_path /wanghaixin/MolMoE/checkpoints/Llama-3.2-3B-Instruct \
    --base_model /wanghaixin/MolMoE/checkpoints/Llama-3.2-3B-Instruct \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /wanghaixin/MolMoE/spin_data/spin_iter_0.json \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type $PROJECTOR \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter /wanghaixin/MolMoE/_checkpoints/pretrain/basline/llava-pub_chem-moleculestm-llama3-3b-naive_linear-/checkpoint-61385/mm_projector.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/lora/conflict-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --stop_epoch 3 \
    --eval_strategy "no" \
    --eval_steps 500 \
    --split_eval False \
    --eval_on_start False \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 5e-6 \
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
    --logging_dir /wanghaixin/MolMoE/tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --is_training True \

#!/bin/bash

PROMPT_VERSION="phi"
MODEL_VERSION="phi3-mini"

# PROMPT_VERSION="llama3"
# MODEL_VERSION="llama3-1b"

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

CHECKPOINT_FOLDER_PREFIX="_checkpoints"
TASK="forward_pred"
PROJECTOR="multilevel_type1"
REMARK="seed0-phi-prompt-old-projector-bug-fixed"


deepspeed main.py \
    --deepspeed scripts/zero_configs/zero2.json \
    --training_recipe stage2 \
    --model_name_or_path /root/autodl-tmp/MoleculeMoE/MolMoE/checkpoints/phi3-mini \
    --base_model /root/autodl-tmp/MoleculeMoE/MolMoE/checkpoints/phi3-mini \
    --language_backbone_name $MODEL_VERSION \
    --version $PROMPT_VERSION \
    --data_path /root/autodl-tmp/MoleculeMoE/MolMoE/Molecule-oriented_Instructions/forward_reaction_prediction.json \
    --data_type $TASK \
    --graph_tower $GRAPH_TOWER \
    --mm_projector_type "multilevel_type1" \
    --projector_aux_loss_coeff 0.0 \
    --graph_init_checkpoint $INIT_CHECKPOINT_GNN \
    --pretrain_mm_mlp_adapter /root/autodl-tmp/MolMoE/_checkpoints/mm_projector_moe_type1_meanpool.bin \
    --bf16 True \
    --output_dir $CHECKPOINT_FOLDER_PREFIX/lora/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --num_train_epochs 7 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --stop_epoch 6 \
    --eval_strategy "no" \
    --split_eval False \
    --eval_on_start False \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 1e-4 \
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
    --logging_dir /root/tf-logs/$TASK-llava-$GRAPH_TOWER-$MODEL_VERSION-$PROJECTOR-$REMARK \
    --is_training True \
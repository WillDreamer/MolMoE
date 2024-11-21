TYPE=lora
TASK=forward_pred/retrosynthesis/reagent_pred/property_pred/molcap
DATA=/root/autodl-tmp/MolMoE/data_files/downsample33
MODEL_PATH=/root/autodl-tmp/MolMoE/_checkpoints/conflict/lora/conflict-llava-moleculestm-tinyllama-naive_linear-all-task-without-eval-10/15epoch/checkpoint-79130
BACKBONE=downloads/TinyLlama-1.1B-Chat-v1.0
PROMPT=tinyllama
REMARK=donwsample-33-mix-train-11epoch



accelerate launch post_train/prepare_dataset.py \
    --model_type $TYPE \
    --task $TASK \
    --model_path $MODEL_PATH \
    --language_backbone $BACKBONE \
    --prompt_version $PROMPT \
    --graph_tower moleculestm \
    --graph_path /root/autodl-tmp/MolMoE/downloads/moleculestm.pth \
    --num_beams 1 \
    --top_p 1.0 \
    --temperature 0.2 \
    --data_path $DATA \
    --output_path spin_data/spin_iter_0.json \
    --batch_size 1 \
    --dtype bfloat16 \
    --use_flash_atten True \
    --device cuda \
    --add_selfies True \
    --is_training False \
    --max_new_tokens 512 \
    --repetition_penalty 1.0
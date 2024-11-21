TYPE=lora
TASK=forward_pred/retrosynthesis/reagent_pred/property_pred/molcap
DATA=/wanghaixin/MolMoE/Molecule-oriented_Instructions
MODEL_PATH=/wanghaixin/MolMoE/_checkpoints/conflict/lora/conflict-llava-moleculestm-llama3-3b-naive_linear-all-task-8epoch/checkpoint-86320
BACKBONE=/wanghaixin/MolMoE/checkpoints/Llama-3.2-3B-Instruct
PROMPT=llama3
REMARK=train-10epoch

cd /wanghaixin/MolMoE


/root/anaconda3/bin/accelerate launch post_train/prepare_dataset.py \
    --model_type $TYPE \
    --task $TASK \
    --model_path $MODEL_PATH \
    --language_backbone $BACKBONE \
    --prompt_version $PROMPT \
    --graph_tower moleculestm \
    --graph_path /wanghaixin/MolMoE/checkpoints/moleculestm.pth \
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
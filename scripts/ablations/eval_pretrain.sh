TYPE=lora
DATA=/root/autodl-tmp/MolMoE/data_files/
MODEL_PATH=/root/autodl-tmp/MolMoE/_checkpoints/pretrain/basline/llava-pub_chem-moleculestm-llama3-1b-naive_linear-/checkpoint-30690
BACKBONE=/root/autodl-tmp/MolMoE/downloads/Llama-3.2-1B-Instruct
PROMPT=llama3



python experiment_tools/eval_pretrain.py \
    --model_path $MODEL_PATH \
    --language_backbone $BACKBONE \
    --prompt_version $PROMPT \
    --graph_tower moleculestm \
    --graph_path /root/autodl-tmp/MolMoE/downloads/moleculestm.pth \
    --num_beams 1 \
    --top_p 1.0 \
    --temperature 0.2 \
    --data_path $DATA \
    --output_path eval_result/pretrain/ \
    --batch_size 1 \
    --dtype bfloat16 \
    --use_flash_atten True \
    --device cuda \
    --add_selfies True \
    --is_training False \
    --max_new_tokens 1024 \
    --repetition_penalty 1.0
TYPE=lora
TASK=molcap
# TASK=reag_pred, retrosyn, prop_pred, molcap, fwd_pred
DATA=/root/autodl-tmp/MolMoE/data_files/molcap_test.txt
MODEL_PATH=/root/autodl-tmp/MolMoE/_checkpoints/basline/lora/molcap-llava-moleculestm-llama3-1b-naive_linear-/checkpoint-11133
BACKBONE=/root/autodl-tmp/MolMoE/downloads/Llama-3.2-1B-Instruct
PROMPT=llama3
REMARK=baseline_test



python eval_engine.py \
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
    --output_path eval_result/baseline/$TASK-$TYPE-$PROMPT-$REMARK-answer.json \
    --batch_size 1 \
    --dtype bfloat16 \
    --use_flash_atten True \
    --device cuda \
    --add_selfies True \
    --is_training False \
    --max_new_tokens 1024 \
    --repetition_penalty 1.0
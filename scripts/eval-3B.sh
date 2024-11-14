TYPE=lora

# TASK=fwd_pred
# DATA=/wanghaixin/MolMoE/Molecule-oriented_Instructions/forward_reaction_prediction.json
# TASK=reag_pred
# DATA=/wanghaixin/MolMoE/Molecule-oriented_Instructions/reagent_prediction.json
# TASK=retrosyn
# DATA=/wanghaixin/MolMoE/Molecule-oriented_Instructions/retrosynthesis.json
# TASK=prop_pred
# DATA=/wanghaixin/MolMoE/Molecule-oriented_Instructions/property_prediction.json
TASK=molcap
DATA=/wanghaixin/MolMoE/Molecule-oriented_Instructions/molcap_test.txt



MODEL_PATH=/wanghaixin/MolMoE/_checkpoints/conflict/lora/conflict-llava-moleculestm-llama3-3b-naive_linear-all-task-8epoch/checkpoint-86320

BACKBONE=/wanghaixin/MolMoE/checkpoints/Llama-3.2-3B-Instruct
PROMPT=llama3
# REMARK=conflict_test_re+pro
REMARK=all-task-8epoch

export CUDA_VISIBLE_DEVICES=0

python eval_engine.py \
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
    --output_path eval_result/$TASK-$TYPE-$PROMPT-$REMARK-answer.json \
    --batch_size 1 \
    --dtype bfloat16 \
    --use_flash_atten True \
    --device cuda \
    --add_selfies True \
    --is_training False \
    --max_new_tokens 512 \
    --repetition_penalty 1.0
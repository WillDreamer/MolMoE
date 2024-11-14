TYPE=lora

# TASK=fwd_pred
# DATA=/home/whx/MolMoE/Molecule-oriented_Instructions/forward_reaction_prediction.json
# TASK=reag_pred
# DATA=/home/whx/MolMoE/Molecule-oriented_Instructions/reagent_prediction.json
# TASK=retrosyn
# DATA=/home/whx/MolMoE/Molecule-oriented_Instructions/retrosynthesis.json
# TASK=prop_pred
# DATA=/home/whx/MolMoE/Molecule-oriented_Instructions/property_prediction.json
TASK=molcap
DATA=/home/whx/MolMoE/Molecule-oriented_Instructions/molcap_test.txt


# MODEL_PATH=/home/whx/MolMoE/_checkpoints/conflict/lora/conflict-llava-moleculestm-llama3-1b-naive_linear-all-task/checkpoint-1294740
# MODEL_PATH=/home/whx/MolMoE/_checkpoints/conflict/lora/conflict-llava-moleculestm-llama3-1b-naive_linear-reagent_pred/property_pred/checkpoint-181690
# MODEL_PATH=/home/whx/MolMoE/_checkpoints/conflict/lora/conflict-llava-moleculestm-llama3-1b-naive_linear-all-task-0.33/checkpoint-94950
MODEL_PATH=/home/whx/MolMoE/_checkpoints/conflict/lora/conflict-llava-moleculestm-llama3-1b-naive_linear-all-task-0.8/checkpoint-230180

BACKBONE=/home/whx/MolMoE/checkpoints/Llama-3.2-1B-Instruct
PROMPT=llama3
# REMARK=conflict_test_re+pro
REMARK=all-task-0.8

export CUDA_VISIBLE_DEVICES=2

python eval_engine.py \
    --model_type $TYPE \
    --task $TASK \
    --model_path $MODEL_PATH \
    --language_backbone $BACKBONE \
    --prompt_version $PROMPT \
    --graph_tower moleculestm \
    --graph_path /home/whx/MolMoE/checkpoints/moleculestm.pth \
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

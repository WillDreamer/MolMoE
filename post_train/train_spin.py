# @ MolMoE projects 2024

import os
import sys
sys.path.append(os.getcwd())
from post_train.engine import SPINTrainer
import transformers
from helper_utils import initialize_distributed_training, model_profiler
from arguments import ModelArguments, DataArguments, TrainingArguments
import train_engine
from model import model_factory
import pathlib
from pathlib import Path
import os
import json
from data_pipeline.datasets import DATASET_MAP, GraphDatasetCollator
from dataclasses import asdict
from helper_utils import seperate_save_lora
from dataclasses import dataclass, field
from torch.utils.data import ConcatDataset, random_split
from torch.utils.data import DataLoader
import torch
from typing import Optional
from post_train.data_utils import AllTaskDataset, SPINMultiTaskCollator



@dataclass
class ExperimentArgs:
    task: str = field(default="forward_pred/retrosynthesis/reagent_pred/property_pred/molcap")
    beta: Optional[float] = field(
        default=0.1,
        metadata={"help": "The beta factor in SPIN loss. Higher beta means less divergence from the initial policy."},
    )

def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments, ExperimentArgs]:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ExperimentArgs))
    model_args, data_args, training_args, exp_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args, training_args, exp_args

def build_dataset(tokenizer, data_args: DataArguments, exp_args: ExperimentArgs):
    train_set = AllTaskDataset(data_args, tokenizer)
    print("Dataset length:", len(train_set))

    return {
        "train_dataset": train_set,
        "eval_dataset": None,
        "data_collator": SPINMultiTaskCollator(tokenizer=tokenizer)
    }


def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, exp_args: ExperimentArgs):
    # override built-in print to print only on master rank
    initialize_distributed_training(training_args.local_rank)
    # seed everything
    transformers.set_seed(0)
    # Dump args
    args = {
        "Model Args": asdict(model_args), 
        "Data Args": asdict(data_args), 
        "Training Args": asdict(model_args),
        "Experiment Args": asdict(exp_args)
        }
    if not os.path.exists(training_args.output_dir):
        Path(training_args.output_dir).mkdir(parents=True)
    with open(os.path.join(training_args.output_dir, "args.json"), mode="w") as f:
        json.dump(args, f, indent=4)
        f.close()
    
    # Create model, tokenizer
    tokenizer, model = model_factory.create_model(model_args, data_args, training_args)
    # create dataset
    data_module = build_dataset(tokenizer=tokenizer, data_args=data_args, exp_args=exp_args)
    
    model_profiler(model, training_args.output_dir)
    
    # save callback
    from transformers import TrainerCallback
    # callback function for model saving
    class SaveCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # get saving dir from args
            checkpoint_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(state.global_step))
            # checkpoint_dir = args.output_dir
            seperate_save_lora(args, checkpoint_dir, model)
            
    class TruncateCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if args.stop_epoch is not None:
                if state.epoch > args.stop_epoch:
                    return {"should_training_stop": True}
    
    # train
    training_args.is_mixup_training = True
    training_args.tasks = exp_args.task
    
    trainer = SPINTrainer(
        model,
        None,
        args=training_args,
        beta=exp_args.beta,
        train_dataset=data_module["train_dataset"],
        eval_dataset=data_module["eval_dataset"],
        data_collator=data_module["data_collator"],
        tokenizer=tokenizer,
        callbacks=[SaveCallback, TruncateCallback]
    )
    
    # If we have saved ckpts, we resume from it and continue training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:  # No savings, train from scratch
        trainer.train()
    
    # Save state dict that is related to training
    trainer.save_state()
    logs = trainer.state.log_history
        
    with open(os.path.join(training_args.output_dir, "logs.json"), mode="w") as f:
        json.dump(logs, f, indent=4)
        f.close()

    model.config.use_cache = True
    model.config.text_config.use_cache = True
    
    
if __name__ == "__main__":
    model_args, data_args, training_args, exp_args = parse_args()
    main(model_args, data_args, training_args, exp_args)
    
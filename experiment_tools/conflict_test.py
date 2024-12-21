# @ MolMoE projects 2024

import os
import sys
sys.path.append(os.getcwd())
import torch
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



@dataclass
class ExperimentArgs:
    # task: str = field(default="reagent_pred/property_pred")
    task: str = field(default="forward_pred/retrosynthesis/reagent_pred/property_pred/molcap")
    # task: str = field(default="forward_pred/retrosynthesis/reagent_pred/property_pred/molcap/catalyst_pred/solvent_pred/yields_regression")
def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments, ExperimentArgs]:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, ExperimentArgs))
    model_args, data_args, training_args, exp_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args, training_args, exp_args

def build_dataset(tokenizer, data_args: DataArguments, exp_args: ExperimentArgs):
    tasks = exp_args.task.split("/")
    print("Using datasets", tasks)
    SIMPLE_MAP = { 
        "forward_pred": "forward",
        "retrosynthesis": "retrosynthesis",
        "reagent_pred": "reagent",
        "property_pred": "property",
        "molcap": "molcap_train",
        "catalyst_pred": "catalyst",
        "solvent_pred": "solvent",
        "yields_regression": "yields",
    }
    datasets = []
    dataset_files = os.listdir(data_args.data_path)
    parent = data_args.data_path
    for task in tasks:
        file_mask = [SIMPLE_MAP[task] in file_name for file_name in dataset_files]
        position = file_mask.index(True)
        data_args.data_path = os.path.join(parent, dataset_files[position])
        data = DATASET_MAP[task](tokenizer=tokenizer, data_args=data_args)
        sampling = False
        if sampling:
            train_size = int(len(data) * 0.8)
            data, _ = random_split(data, [train_size, len(data) - train_size])
        if data_args.over_sampling:
            from torch.utils.data import WeightedRandomSampler
            target_size = int(len(data) * 1.5)  # 目标大小是原始数据集的x倍
            sampler = WeightedRandomSampler(
                weights=[1] * len(data),  # 均匀采样权重
                num_samples=target_size,
                replacement=True  # 允许重复采样
            )
            datasets.append(torch.utils.data.Subset(data, list(sampler)))
        else:
            datasets.append(data)
    
    # (Hao Li): Allow average sampling for each dataset
    if data_args.split_eval:
        train_sets = []
        val_sets = []
        for data in datasets:
            train_set, eval_set = random_split(data, [0.9, 0.1])
            train_sets.append(train_set)
            val_sets.append(eval_set)
            
        train_set, eval_set = ConcatDataset(train_sets), ConcatDataset(val_sets)
            
        print("=======Split eval======")
        print("Length of train:", len(train_set))
        print("Length of eval:", len(eval_set))
    else:
        train_set, eval_set = ConcatDataset(datasets), None

    return {
        "train_dataset": train_set,
        "eval_dataset": eval_set,
        "data_collator": GraphDatasetCollator(tokenizer=tokenizer)
    }


def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments, exp_args: ExperimentArgs):
    # override built-in print to print only on master rank
    training_args.tasks = exp_args.task
    initialize_distributed_training(training_args.local_rank)
    # seed everything
    transformers.set_seed(0)
    # Dump args
    args = {
        "Model Args": asdict(model_args), 
        "Data Args": asdict(data_args), 
        "Training Args": asdict(training_args),
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
                    exit(0)
    
    # train
    trainer = train_engine.MoETrainer(
        model=model,
        args=training_args,
        callbacks=[SaveCallback(), TruncateCallback()],
        **data_module
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
    
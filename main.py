# @ Omni-Mol projects 2024

import transformers
from helper_utils import initialize_distributed_training, model_profiler
from arguments import ModelArguments, DataArguments, TrainingArguments
import train_engine
from model import model_factory
from data_pipeline.datasets import build_dataset
import pathlib
from pathlib import Path
import os
import json
from dataclasses import asdict
from helper_utils import seperate_save_lora


def parse_args() -> tuple[ModelArguments, DataArguments, TrainingArguments]:
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args, training_args


def main(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    # override built-in print to print only on master rank
    initialize_distributed_training(training_args.local_rank)
    # seed everything (Optional)
    transformers.set_seed(0)
    # Dump args
    args = {
        "Model Args": asdict(model_args), 
        "Data Args": asdict(data_args), 
        "Training Args": asdict(training_args)
        }
    if not os.path.exists(training_args.output_dir):
        Path(training_args.output_dir).mkdir(parents=True)
    with open(os.path.join(training_args.output_dir, "args.json"), mode="w") as f:
        json.dump(args, f, indent=4)
        f.close()
    
    # Create model, tokenizer
    tokenizer, model = model_factory.create_model(model_args, data_args, training_args)
    # create dataset
    data_module = build_dataset(tokenizer=tokenizer, data_args=data_args)
    
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
    model_args, data_args, training_args = parse_args()
    main(model_args, data_args, training_args)

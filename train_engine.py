import transformers
from transformers import Trainer, PreTrainedModel, PreTrainedTokenizer
from arguments import TrainingArguments, ModelArguments
import pathlib
import torch
import os
import logging
from transformers.trainer import get_parameter_names, ALL_LAYERNORM_LAYERS
from deepspeed.moe.utils import is_moe_param, split_params_into_different_moe_groups_for_optimizer
from transformers.utils import is_peft_available
from transformers.trainer_utils import EvalLoopOutput
from typing import List, Optional
import importlib
from torch.utils.data import ConcatDataset, DataLoader
from packaging import version
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from model.modelling_llava import MoECausalLMOutputWithPast
import numpy as np
from helper_utils import maybe_zero_3

if is_peft_available():
    from peft import PeftModel
    


def _is_peft_model(model):
    if is_peft_available():
        classes_to_check = (PeftModel,) if is_peft_available() else ()
        
        if version.parse(importlib.metadata.version("peft")) >= version.parse("0.7.0"):
            from peft import PeftMixedModel

            classes_to_check = (*classes_to_check, PeftMixedModel)
        return isinstance(model, classes_to_check)
    return False


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


class MoETrainer(Trainer):
    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is not None:
            return self.optimizer

        
        decay_parameters = [
            n for n in get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            if "bias" not in n
        ]
        
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n in decay_parameters and p.requires_grad
                ],
                "weight_decay": self.args.weight_decay,
                "name": "decay_parameters"
            },
            {
                "params": [
                    p for n, p in opt_model.named_parameters()
                    if n not in decay_parameters and p.requires_grad
                ],
                "weight_decay": 0.0,
                "name": "no_decay_parameters"
            },
        ]
        
        
        for name, param in opt_model.named_parameters():
            if is_moe_param(param):
                print("Detected MoE parameters:", name)

        if self.args.moe_enable:
            print("Splitting params for MoE...")
            optimizer_grouped_parameters = split_params_into_different_moe_groups_for_optimizer(
                optimizer_grouped_parameters
            )

        
        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
    
    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'training_recipe') == "stage1":
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            from transformers.trainer import TRAINER_STATE_NAME
            
            
            

            
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            if self.hp_search_backend is None and trial is None:
                self.store_flos()

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            
            keys_to_match = ['mm_projector']
            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
            
            

            if not self.args.save_only_model:
                
                self._save_optimizer_and_scheduler(output_dir)
                
                self._save_rng_state(output_dir)

            
            if metrics is not None and self.args.metric_for_best_model is not None:
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                try:
                    metric_value = metrics[metric_to_check]
                except KeyError as exc:
                    raise KeyError(
                        f"The `metric_for_best_model` training argument is set to '{metric_to_check}', which is not found in the evaluation metrics. "
                        f"The available evaluation metrics are: {list(metrics.keys())}. Consider changing the `metric_for_best_model` via the TrainingArguments."
                    ) from exc

                operator = np.greater if self.args.greater_is_better else np.less
                if (
                    self.state.best_metric is None
                    or self.state.best_model_checkpoint is None
                    or operator(metric_value, self.state.best_metric)
                ):
                    self.state.best_metric = metric_value
                    self.state.best_model_checkpoint = output_dir

            
            if self.args.should_save:
                
                self.state.stateful_callbacks["TrainerControl"] = self.control.state()
                self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

            if self.args.push_to_hub:
                self._push_from_checkpoint(output_dir)

            
            if self.args.should_save:
                
                
                self._rotate_checkpoints(use_mtime=False, output_dir=run_dir)

        else:
            super(MoETrainer, self)._save_checkpoint(model, trial, metrics)
            
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        
        
        if getattr(outputs, "projector_aux_loss", None) is not None and getattr(outputs, "moe_loss", None) is None:
            self.log(
                {"projector_loss": outputs.projector_aux_loss.item(), 
                 "model_loss": outputs.model_loss,
                 "projector_coeff": outputs.proj_aux_coeff}
                )
        elif getattr(outputs, "projector_aux_loss", None) is None and getattr(outputs, "moe_loss", None) is not None:
            self.log(
                {"moe_loss": outputs.moe_loss.item(), 
                 "model_loss": outputs.model_loss,
                 "router_coeff": outputs.router_aux_coeff}
                )
        elif getattr(outputs, "projector_aux_loss", None) is not None and getattr(outputs, "moe_loss", None) is not None:
            self.log(
                {"projector_loss": outputs.projector_aux_loss.item(), 
                 "moe_loss": outputs.moe_loss.item(),
                 "model_loss": outputs.model_loss,
                 "projector_coeff": outputs.proj_aux_coeff,
                 "router_coeff": outputs.router_aux_coeff}
                )
        
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    
    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        
        
        
        if self.args.is_mixup_training:
            datasets: ConcatDataset = dataloader.dataset
            dataset_list = datasets.datasets
            
            
            data_loaders = []
            
            for data in dataset_list:
                
                
                loader = self.get_eval_dataloader(data)
                data_loaders.append(loader)
                
            tasks = self.args.tasks.split("/")
            for idx, loader in enumerate(data_loaders):
                print("Evaluation on task", tasks[idx], "...")
                output = super().evaluation_loop(
                    loader,
                    description,
                    prediction_loss_only,
                    ignore_keys,
                    metric_key_prefix
                )
                self.log({f"eval_loss_{tasks[idx]}": output.metrics["eval_loss"]})
                print(" ")
                
            base_output = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix
            )
            return base_output
        
        else:
            return super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix
            )
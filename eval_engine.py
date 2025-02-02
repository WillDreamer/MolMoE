from model.model_factory import load_pretrain_model, load_lora_model, load_lora2lora_model, load_full_model, load_moe_lora_model
from transformers import HfArgumentParser, GenerationConfig, PreTrainedTokenizer
from dataclasses import dataclass, field
from pathlib import Path
import torch
from data_pipeline import conversation_lib
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from model.modelling_llava import GraphLlavaForConditionalGeneration
import json
from data_pipeline.data_structure import MolGraph, smiles2graph
from data_pipeline.preprocess_engine import tokenizer_image_token
from typing import Sequence, Dict, Tuple, List
from torch_geometric.data import Data, Batch
import selfies
from tqdm import tqdm
import os
import random
from metric_factory import calc_fingerprints, calc_mocap_metrics, calc_mol_trans, compute_mae
from accelerate import Accelerator
from accelerate.utils import gather_object, InitProcessGroupKwargs
from helper_utils import initialize_distributed_training
from datetime import timedelta
import logging
from contextlib import nullcontext
local_rank = os.environ.get("LOCAL_RANK", None)
if os.environ.get("LOCAL_RANK", None) is not None:
    logging.basicConfig(
            level=logging.INFO,
            format=f'[rank {local_rank}]' + '[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
    accelerator = Accelerator(kwargs_handlers=[kwargs])

IGNORE_INDEX = -100


def apply_chat_template(message, tokenizer, has_image):
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    
    return prompt


MODEL_LOADER_MAP = {
    "pretrain": load_pretrain_model,
    "lora": load_lora_model,
    "lora2lora": load_lora2lora_model,
    "full": load_full_model,
    "lora+moe": load_moe_lora_model
}

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float": torch.float
}

@dataclass
class EvalArguments:
    model_type: str = field(default="lora")
    task: str = field(default="fwd_pred", metadata={"choices": ["fwd_pred", "reag_pred", "retrosyn", "molcap", "prop_pred"]})
    model_path: str = field(default=None)
    language_backbone: str = field(default="checkpoints/phi3-mini")
    prompt_version: str = field(default="phi3")
    graph_tower: str = field(default="himol")
    graph_path: str = field(default=None)
    num_beams: int = field(default=1)
    top_p: float = field(default=1.0)
    temperature: float = field(default=0.2)
    data_path: str = field(default="forward_reaction_prediction.json")
    output_path: str = field(default="eval_result")
    batch_size: int = field(default=1)
    dtype: str = field(default="bfloat16", metadata={"choices": ["bfloat16", "float16", "float"]})
    use_flash_atten:bool = field(default=True)
    device:str = field(default="cuda", metadata={"choices": ["cpu", "cuda"]})
    add_selfies: bool = field(default=True)
    is_training: bool = False
    max_new_tokens: int = field(default=512)
    repetition_penalty: float = field(default=1.0)
    
    
@dataclass       
class GraphEvalCollator(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, gt, prompt = self._extract_tensors(instances, ("input_ids", "gt", "prompt"))
        
        batch_input = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        batch = {
            'input_ids': batch_input["input_ids"][:, :self.tokenizer.model_max_length],
            'gt': gt,
            'prompt': prompt,
            'attention_mask': batch_input["attention_mask"][:, :self.tokenizer.model_max_length],
        }
        if 'graphs' in instances[0]:
            batch['graphs'] = Batch.from_data_list(
                [self._convert_dict_to_Data(instance["graphs"]) for instance in instances]
            )
        return batch
    
    def _extract_tensors(self, instances, keys: Tuple[str, str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return tuple([instance[key] for instance in instances] for key in keys)

    def _pad_sequence(self, sequence: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        return self.tokenizer.pad({"input_ids": sequence}, padding=True)

    def _convert_dict_to_Data(self, data_dict: Dict) -> Data:
        if getattr(data_dict, "num_part", None) is not None: 
            return Data(
            x=torch.asarray(data_dict.x),
            edge_index=torch.asarray(data_dict.edge_index),
            edge_attr=torch.asarray(data_dict.edge_attr),
            num_part=data_dict.num_part
            )
            
        return Data(
            x=torch.asarray(data_dict['node_feat']),
            edge_attr=torch.asarray(data_dict['edge_feat']),
            edge_index=torch.asarray(data_dict['edge_index']),
        )
    
    
class MetaEvalDataset(Dataset):
    def __init__(self, args: EvalArguments, tokenizer):
        super().__init__()
        with open(args.data_path, "rb") as f:
            self.list_data_dict = json.load(f)
            f.close()
        
        self.tokenizer = tokenizer
        self.args=args
            
        print("Total number of samples:", self.__len__())
        self.list_data_dict = [raw for raw in self.list_data_dict if raw['metadata']['split'] == 'test']
        print("Filtered", self.__len__(), "for test")
        
    def selfies2smiles(self, selfies_str: str) -> str | None:
        try:
            smiles_str = selfies.decoder(selfies_str)
        except:
            smiles_str = None
            
        return smiles_str
        
    def __len__(self):
        return len(self.list_data_dict)
    
    def __getitem__(self, i):
        raise NotImplementedError
    

class EvalForwardPredDataset(MetaEvalDataset):
    def __init__(self, tokenizer, args: EvalArguments):
        super().__init__(args, tokenizer)
        
    def __getitem__(self, i):
        
        raw = self.list_data_dict[i]

        
        instruction = raw['instruction']
        inputs, output_selfies = raw['input'].split('.'), raw['output']
        
        
        reactant_smiles = self.selfies2smiles(inputs[0])
        graph_for_first_reactant = MolGraph(reactant_smiles) \
        if self.args.graph_tower == "himol" else smiles2graph(reactant_smiles)
        
        instruction = "<image>\n" + instruction

        
        if self.args.add_selfies:
            instruction += " " + raw['input']
        elif len(inputs) > 1:
            instruction += f" The other joint reactants are: {','.join(inputs[1:])}.\n"

        
        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph_for_first_reactant is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        
        data = {
            "input_ids": input_ids,
            "graphs": graph_for_first_reactant,
            "gt": output_selfies,
            "prompt": prompt
        }
        
        return data
    
    
class EvalReagentPredDataset(MetaEvalDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: EvalArguments) -> None:
        super().__init__(args, tokenizer)
        print("=====Reagent Prediction dataset=====")
        
    @staticmethod
    def construct_instruct_question(product:str):
        """
        Construct instruct question for each graph
        """
        question_pools = [
            'Can you suggest some possible reagents that could have been used in the following chemical reaction?',
            'Give some possible reagents that could have been used in the following chemical reaction.',
            'Please propose potential reagents that might have been utilized in the provided chemical reaction.',
            'Please provide possible reagents based on the following chemical reaction.',
        ]
        question = random.choice(question_pools)
        question += f"\nThe product is {product}"
        return question

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        
        reactant, product = input.split(">>")
        
        reactant_smiles = self.selfies2smiles(reactant)
        if not self.args.add_selfies:
            
            instruction = self.construct_instruct_question(product)
        else:
            instruction = raw['instruction'] + f" The reaction is {input}"

        instruction = "<image>\n" + instruction

            
        if self.args.graph_tower == "himol":
            graph = MolGraph(reactant_smiles)
        else:
            graph=smiles2graph(reactant_smiles)

        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        
        data = {
            "input_ids": input_ids,
            "graphs": graph,
            "gt": output_selfies,
            "prompt": prompt
        }
        
        return data
    
    
class EvalRetrosynDataset(MetaEvalDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: EvalArguments) -> None:
        super().__init__(args, tokenizer)
        print("=====Retrosynthesis dataset=====")
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        if self.args.add_selfies:
            instruction += f" The product is: {raw['input']}"

        instruction = "<image>\n" + instruction
        
        input_selfies, output_selfies = raw['input'], raw['output']
        
        reactant_smiles = self.selfies2smiles(input_selfies)
        
        if self.args.graph_tower == "himol":
            graph = MolGraph(reactant_smiles)
        else:
            
            graph=smiles2graph(reactant_smiles)
            
        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        
        data = {
            "input_ids": input_ids,
            "graphs": graph,
            "gt": output_selfies,
            "prompt": prompt
        }
        
        return data
    
    
class EvalPropertyPredDataset(MetaEvalDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: EvalArguments) -> None:
        super().__init__(args, tokenizer)
        print("=====Property Prediction dataset=====")
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        instruction = raw['instruction']
        if self.args.add_selfies:
            instruction += f" The compound SELFIES sequence is: {raw['input']}"

        instruction = "<image>\n" + instruction
        
        input_selfies, target = raw['input'], str(raw['output'])
        
        if self.args.graph_tower == "himol":
            graph = MolGraph(self.selfies2smiles(input_selfies))
        else:
            
            graph=smiles2graph(self.selfies2smiles(input_selfies))
            
        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        
        data = {
            "input_ids": input_ids,
            "graphs": graph,
            "gt": target,
            "prompt": prompt
        }
        
        return data
    
    
class EvalMolcapDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args: EvalArguments) -> None:
        super().__init__()
        with open(args.data_path, "rt") as f:
            f.readline()  
            self.list_data = f.readlines()
            f.close()
        
        self.args = args
        self.tokenizer = tokenizer
        print("Read test file from", args.data_path)
        print("Total length of test file:", self.__len__())
        print("=====Molcap dataset=====")
        self.question_pool = [
            'Could you give me a brief overview of this molecule?',
            'Could you provide a description of this molecule?',
            'Describe this molecule.',
            'Please give me some details about this molecule.',
            'Provide a brief overview of this molecule.',
            'Provide a description of this molecule.',
            'What can you tell me about this molecule?'
        ]
        
    def smiles2selfies(self, smiles_str):
        try:
            selfies_str = selfies.encoder(smiles_str)
        except:
            selfies_str = None
        return selfies_str
        
    def __len__(self):
        return len(self.list_data)    
    
    def __getitem__(self, i):
        line = self.list_data[i].rstrip("\n").split("\t")
        cid, smi, gt = line
        instruction = random.choice(self.question_pool)
        if self.args.add_selfies:
            selfies_str = self.smiles2selfies(smi)
            if selfies_str is not None:
                instruction += f" The compound SELFIES sequence is: {selfies_str}."
                
        instruction = "<image>\n" + instruction
                
        if self.args.graph_tower == "himol":
            graph = MolGraph(smi)
        else:
            graph = smiles2graph(smi)

        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        data = {
            "input_ids": input_ids,
            "graphs": graph,
            "gt": gt,
            "prompt": prompt
        }

        return data

    
def build_pretrained_model(
    model_type,
    model_path,
    language_backbone,
    graph_path,
    use_flash_attn
    ) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    tokenizer, model = MODEL_LOADER_MAP[model_type](model_path, language_backbone, graph_path, use_flash_attn)
    
    return tokenizer, model


TASK_MAP = {
    "fwd_pred": EvalForwardPredDataset,
    "reag_pred": EvalReagentPredDataset,
    "retrosyn": EvalRetrosynDataset,
    "prop_pred": EvalPropertyPredDataset,
    "molcap": EvalMolcapDataset
}

@torch.inference_mode
def start_eval(args: EvalArguments):
    tokenizer, model = build_pretrained_model(args.model_type, args.model_path, args.language_backbone, args.graph_path, args.use_flash_atten)
    tokenizer.padding_side="left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
        
    if args.use_flash_atten:
        print("Using flash attention, will force the computation dtype to bfloat16...")
        assert args.device == "cuda", "Flash attention only supports running on CUDA devices!"
        model.to(torch.bfloat16)
    else:
        model.to(DTYPE_MAP[args.dtype])
        
    model.to(args.device)
    
    print(model)
    
    generation_config = GenerationConfig.from_pretrained(args.language_backbone)
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.prompt_version]
    print("Using conversation template of", args.prompt_version)
    print("Conversation template:", conversation_lib.default_conversation)
    
    dataset = TASK_MAP[args.task](args=args, tokenizer=tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=GraphEvalCollator(tokenizer), drop_last=False)
    if local_rank is not None:
        all_batches = []
        for data in tqdm(loader, desc="Sampling all data"):
            all_batches.append(data)
        accelerator.wait_for_everyone()
        output = []
        cnt = 0
        with accelerator.split_between_processes(all_batches) as batch:
            for each in tqdm(batch, desc=f"Inference [rank {local_rank}]"):
                input_ids, graphs = each["input_ids"].to(args.device), each["graphs"].to(args.device)
                output_ids = model.generate(
                    input_ids,
                    graphs=graphs,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    repetition_penalty=args.repetition_penalty,
                    use_cache=True,
                    attention_mask=each["attention_mask"].to(args.device),
                    generation_config=generation_config
                )
                
                for idx, (result, input_id, prompt, gt) in enumerate(zip(output_ids, input_ids, each["prompt"], each["gt"])):
                    this_output = {
                        "prompt": prompt,
                        "gt": gt,
                        "pred": tokenizer.decode(result[input_id.shape[0]:])
                    }
                    output.append(this_output)
                    if cnt < 10:
                        print("\n", this_output, "\n")
                    
                cnt += 1
        print("Gathering object from processes...")       
        output = gather_object(output)
    else:      
        output = []
        for idx, batch in enumerate(tqdm(loader, desc="inference")):
            input_ids, graphs = batch["input_ids"].to(args.device), batch["graphs"].to(args.device)
            output_ids = model.generate(
                input_ids,
                graphs=graphs,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                use_cache=True,
                attention_mask=batch["attention_mask"].to(args.device),
                generation_config=generation_config
            )
            
            
            
            
            for idx, (result, input_id, prompt, gt) in enumerate(zip(output_ids, input_ids, batch["prompt"], batch["gt"])):
                this_output = {
                    "prompt": prompt,
                    "gt": gt,
                    "pred": tokenizer.decode(result[input_id.shape[0]:])
                }
                output.append(this_output)
                print("\n", this_output, "\n")
    
    return output, tokenizer




if __name__ == "__main__":
    if local_rank is not None:
        initialize_distributed_training(local_rank)
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]
    assert args.batch_size == 1, "Batched evaluation is under development!"
    output, tokenizer = start_eval(args)
    if local_rank is not None:
        if accelerator.is_local_main_process:
            path = args.output_path.split("/")[:-1]
            path = "/".join(path)
            file = args.output_path.split("/")[-1]
            if not os.path.exists(path):
                Path(path).mkdir(parents=True)
                
            with open(args.output_path, mode="w") as f:
                json.dump(output, f, indent=2)
                f.close()
                
            EOS_MAP = {
                "phi": "<|endoftext|>",
                "phi3": "<|end|>",
                "llama3": "<|eot_id|>",
                "tinyllama": "</s>"
            }
                
            if args.task in ["fwd_pred", "reag_pred", "retrosyn"]:
                calc_mol_trans(args.output_path, EOS_MAP[args.prompt_version])
                calc_fingerprints(args.output_path, eos_token=EOS_MAP[args.prompt_version])
            
            
                
            if args.task == "molcap":
                if tokenizer.pad_token is None and args.prompt_version == "llama3":
                    tokenizer.pad_token = "<|finetune_right_pad_id|>"
                calc_mocap_metrics(args.output_path, EOS_MAP[args.prompt_version], tokenizer)
                
            if args.task == "prop_pred":
                compute_mae(args.output_path, EOS_MAP[args.prompt_version])
    else:
        path = args.output_path.split("/")[:-1]
        path = "/".join(path)
        file = args.output_path.split("/")[-1]
        if not os.path.exists(path):
            Path(path).mkdir(parents=True)
            
        with open(args.output_path, mode="w") as f:
            json.dump(output, f, indent=2)
            f.close()
            
        EOS_MAP = {
            "phi": "<|endoftext|>",
            "phi3": "<|end|>",
            "llama3": "<|eot_id|>",
            "tinyllama": "</s>"
        }
            
        if args.task in ["fwd_pred", "reag_pred", "retrosyn"]:
            calc_mol_trans(args.output_path, EOS_MAP[args.prompt_version])
            calc_fingerprints(args.output_path, eos_token=EOS_MAP[args.prompt_version])
        
        
            
        if args.task == "molcap":
            if tokenizer.pad_token is None and args.prompt_version == "llama3":
                tokenizer.pad_token = "<|finetune_right_pad_id|>"
            calc_mocap_metrics(args.output_path, EOS_MAP[args.prompt_version], tokenizer)
            
        if args.task == "prop_pred":
            compute_mae(args.output_path, EOS_MAP[args.prompt_version])
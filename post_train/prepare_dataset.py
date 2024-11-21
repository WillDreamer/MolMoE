import os
import sys
sys.path.append(os.getcwd())

import json
import random
import selfies
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from datetime import timedelta
import torch
from typing import Sequence, Dict, Tuple, List
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import HfArgumentParser, GenerationConfig, PreTrainedTokenizer
from accelerate import Accelerator
from accelerate.utils import gather_object, InitProcessGroupKwargs
from torch_geometric.data import Data, Batch
from dataclasses import dataclass, field
from model.model_factory import load_pretrain_model, load_lora_model
from model.modelling_llava import GraphLlavaForConditionalGeneration
from data_pipeline import conversation_lib
from data_pipeline.data_structure import MolGraph, smiles2graph
from data_pipeline.preprocess_engine import tokenizer_image_token
from helper_utils import initialize_distributed_training

import logging
local_rank = os.environ['LOCAL_RANK']
logging.basicConfig(
            level=logging.INFO,
            format=f'[rank {local_rank}]' + '[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)

kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=72000))
accelerator = Accelerator(kwargs_handlers=[kwargs])

IGNORE_INDEX = -100
MODEL_LOADER_MAP = {
    "pretrain": load_pretrain_model,
    "lora": load_lora_model
}
DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float": torch.float
}

@dataclass
class EvalArguments:
    model_type: str = field(default="lora")
    task: str = field(default="forward_pred/retrosynthesis/reagent_pred/property_pred/molcap")
    model_path: str = field(default=None)
    language_backbone: str = field(default="checkpoints/phi3-mini")
    prompt_version: str = field(default="phi3")
    graph_tower: str = field(default="himol")
    graph_path: str = field(default=None)
    num_beams: int = field(default=1)
    top_p: float = field(default=1.0)
    temperature: float = field(default=0.2)
    data_path: str = field(default="data_files")
    output_path: str = field(default="spin_data/spin_iter0.json")
    batch_size: int = field(default=1)
    dtype: str = field(default="bfloat16", metadata={"choices": ["bfloat16", "float16", "float"]})
    use_flash_atten:bool = field(default=True)
    device:str = field(default="cuda", metadata={"choices": ["cpu", "cuda"]})
    add_selfies: bool = field(default=True)
    is_training: bool = False
    max_new_tokens: int = field(default=512)
    repetition_penalty: float = field(default=1.0)
    num_sample: int = field(default=5e4)
    

def apply_chat_template(message, tokenizer, has_image):
    conv = conversation_lib.default_conversation.copy()
    conv.append_message(conv.roles[0], message)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    
    return prompt


def load_json(file_path: str):
    with open(file_path, mode="r") as fptr:
        file = json.load(fptr)
        fptr.close()
        
    return file

def concat_files(tasks: str, data_path: str) -> list[dict]:
    tasks = tasks.split("/")
    print("Using datasets", tasks)
    SIMPLE_MAP = {
        "forward_pred": "forward",
        "retrosynthesis": "retrosynthesis",
        "reagent_pred": "reagent",
        "property_pred": "property",
        "molcap": "molcap_train"
    }
    all_datafiles = []
    dataset_files = os.listdir(data_path)
    parent = data_path
    for task in tqdm(tasks, desc="Loading data files"):
        file_mask = [SIMPLE_MAP[task] in file_name for file_name in dataset_files]
        position = file_mask.index(True)
        data_path = os.path.join(parent, dataset_files[position])
        
        data: list[dict] = load_json(data_path)
        all_datafiles.extend(data)
        
    return all_datafiles
        
        
@dataclass       
class GraphEvalCollator(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, raw = self._extract_tensors(instances, ("input_ids", "raw"))
        
        batch_input = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        batch = {
            'input_ids': batch_input["input_ids"][:, :self.tokenizer.model_max_length],
            "raw": raw,
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
        if getattr(data_dict, "num_part", None) is not None: # which means we are using himol
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
        
        
class AllTaskDataset(Dataset):
    def __init__(self, data_list: list, args, tokenizer):
        super().__init__()
        self.data_list = data_list
        self.args = args
        self.tokenizer = tokenizer
        self.question_pool = [
            'Could you give me a brief overview of this molecule?',
            'Could you provide a description of this molecule?',
            'Describe this molecule.',
            'Please give me some details about this molecule.',
            'Provide a brief overview of this molecule.',
            'Provide a description of this molecule.',
            'What can you tell me about this molecule?'
        ]
        self.MAP_TO_PROCESS = {
            "forward reaction prediction": self.preprocess_forward,
            "reagent prediction": self.preprocess_reagent,
            "retrosynthesis": self.preprocess_retrosyn,
            "property prediction": self.preprocess_property,
            "molecular description generation": self.proprocess_molcap
        }
        
        self.data_list = [raw for raw in self.data_list if raw["metadata"]["split"] == "train"]
            
        
        
    def __len__(self):
        return len(self.data_list)
        
    def selfies2smiles(self, selfies_str: str) -> str | None:
        try:
            smiles_str = selfies.decoder(selfies_str)
        except:
            smiles_str = None
            
        return smiles_str
    
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
        
    def preprocess_forward(self, raw):
        # 2. Get instruction, input selfies, output selfies
        instruction = raw['instruction']
        inputs, output_selfies = raw['input'].split('.'), raw['output']
        
        # 3. Convert to Graph
        reactant_smiles = self.selfies2smiles(inputs[0])
        graph_for_first_reactant = MolGraph(reactant_smiles) \
        if self.args.graph_tower == "himol" else smiles2graph(reactant_smiles)
        
        instruction = "<image>\n" + instruction

        # 4. Add SELFIES
        if self.args.add_selfies:
            instruction += " " + raw['input']
        elif len(inputs) > 1:
            instruction += f" The other joint reactants are: {','.join(inputs[1:])}.\n"

        # chat template
        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph_for_first_reactant is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        
        data = {
            "input_ids": input_ids,
            "graphs": graph_for_first_reactant,
            "raw": raw
        }
        
        return data
    
    def preprocess_reagent(self, raw):
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if not self.args.add_selfies:
            # insert product to the instruction end
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
            "raw": raw
        }
        
        return data
    
    def preprocess_retrosyn(self, raw):
        instruction = raw['instruction']
        if self.args.add_selfies:
            instruction += f" The product is: {raw['input']}"

        instruction = "<image>\n" + instruction
        
        input_selfies, output_selfies = raw['input'], raw['output']
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(input_selfies)
        
        if self.args.graph_tower == "himol":
            graph = MolGraph(reactant_smiles)
        else:
            # graph data containes the information of the first SELFIES 
            graph=smiles2graph(reactant_smiles)
            
        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        
        data = {
            "input_ids": input_ids,
            "graphs": graph,
            "raw": raw
        }
        
        return data
    
    def preprocess_property(self, raw):
        instruction = raw['instruction']
        if self.args.add_selfies:
            instruction += f" The compound SELFIES sequence is: {raw['input']}"

        instruction = "<image>\n" + instruction
        
        input_selfies, target = raw['input'], str(raw['output'])
        # convert input selfies to smiles for building graph
        if self.args.graph_tower == "himol":
            graph = MolGraph(self.selfies2smiles(input_selfies))
        else:
            # graph data containes the information of the first SELFIES 
            graph=smiles2graph(self.selfies2smiles(input_selfies))
            
        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        
        data = {
            "input_ids": input_ids,
            "graphs": graph,
            "raw": raw
        }
        
        return data
    
    def proprocess_molcap(self, raw):
        instruction = random.choice(self.question_pool)
        input = raw['input']
        output = raw['output']

        if self.args.add_selfies:
            instruction += f" The compound SELFIES sequence is: {input}"

        instruction = "<image>\n" + instruction

        if self.args.graph_tower == "himol":
            graph = MolGraph(self.selfies2smiles(input))
        else:
            graph = smiles2graph(self.selfies2smiles(input))

        prompt = apply_chat_template(instruction, self.tokenizer, has_image=(graph is not None))
        input_ids = tokenizer_image_token(prompt, self.tokenizer, -200, return_tensors="pt")
        
        data = {
            "input_ids": input_ids,
            "graphs": graph,
            "raw": raw
        }
        
        return data
        
        
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        processor = self.MAP_TO_PROCESS[data_dict["metadata"]["task"]]
        
        return processor(data_dict)
    
    
def build_pretrained_model(
    model_type,
    model_path,
    language_backbone,
    graph_path,
    use_flash_attn
    ) -> tuple[PreTrainedTokenizer, GraphLlavaForConditionalGeneration]:
    tokenizer, model = MODEL_LOADER_MAP[model_type](model_path, language_backbone, graph_path, use_flash_attn)
    
    return tokenizer, model
    
    
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
    
    
    all_data_list = concat_files(args.task, args.data_path)
    dataset = AllTaskDataset(all_data_list, args, tokenizer)
    dataset_len = len(dataset)
    print("All dataset length:", dataset_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=GraphEvalCollator(tokenizer), drop_last=False)
    all_batches = []
    for data in tqdm(loader, desc=f"Sampling data [rank {local_rank}]"):
        all_batches.append(data)
        
    accelerator.wait_for_everyone()
    all_batches = random.sample(all_batches, int(args.num_sample))
    print("Sampled length", len(all_batches))
    accelerator.wait_for_everyone()
    
    output = []
    with accelerator.split_between_processes(all_batches) as batch:
        for idx1, data in enumerate(tqdm(batch, desc=f"inference [rank {local_rank}]")):
            input_ids, graphs = data["input_ids"].to(args.device), data["graphs"].to(args.device)
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
                attention_mask=data["attention_mask"].to(args.device),
                generation_config=generation_config
            )
            
            for idx2, (result, input_id, raw) in enumerate(zip(output_ids, input_ids, data["raw"])):
                generated_batch = deepcopy(raw)
                generated_batch["output"] = tokenizer.decode(result[input_id.shape[0]:], skip_special_tokens=True)
                this_output = {
                    "real": raw,
                    "generated": generated_batch
                }
                output.append(this_output)
                if idx1 < 10:
                    print("\n", this_output, "\n")
    
    print("Gathering object between processes...")
    results_gathered=gather_object(output)
    
    return results_gathered, tokenizer




if __name__ == "__main__":
    initialize_distributed_training(os.environ["LOCAL_RANK"])
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]
    assert args.batch_size == 1, "Batched evaluation is under development!"
    output, tokenizer = start_eval(args)
    path = args.output_path.split("/")[:-1]
    path = "/".join(path)
    file = args.output_path.split("/")[-1]
    if not os.path.exists(path):
        Path(path).mkdir(parents=True)
        
    if accelerator.is_local_main_process:
        print("Saving on master rank...")
        with open(args.output_path, mode="w") as f:
            json.dump(output, f, indent=2)
            f.close()
        
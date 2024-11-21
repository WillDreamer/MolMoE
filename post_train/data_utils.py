import torch
from torch.utils.data import Dataset
import selfies
from data_pipeline import conversation_lib
from data_pipeline.data_structure import MolGraph, smiles2graph
from data_pipeline.preprocess_engine import tokenizer_image_token
from data_pipeline.preprocess_engine import apply_chat_template
import random
from dataclasses import dataclass
from transformers import PreTrainedTokenizer
from typing import List, Dict, Sequence, Tuple
from torch_geometric.data import Batch, Data
import json

IGNORE_INDEX=-100



"""
Structure of the data
{
    "Real":{
        
    },
    "Generated":{
        
    }
}
"""
class AllTaskDataset(Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        with open(args.data_path, mode="r") as fpt:
            data_list = json.load(fpt)
            fpt.close()
            
        self.data_list = data_list
        self.data_args = args
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
        if self.data_args.graph_tower == "himol" else smiles2graph(reactant_smiles)

        # 4. Add SELFIES
        if self.data_args.add_selfies:
            instruction += " " + raw['input']
        elif len(inputs) > 1:
            instruction += f" The other joint reactants are: {','.join(inputs[1:])}"
            
        # NOTE(Hao Li): Override to left, LLaVA pre-processing multimodal will change the 
        # <image> to the left anyway
        instruction = "<image>\n" + instruction

        # 5. Prepare conversations
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        # Tokenization
        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph_for_first_reactant is not None))

        # Simplify data_dict extraction
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # Add graph or check multimodal settings
        if graph_for_first_reactant is not None:
            data_dict['graphs'] = graph_for_first_reactant
            assert -200 in data_dict["input_ids"], "Input IDs missing expected <image> token"

        return data_dict
    
    def preprocess_reagent(self, raw):
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if not self.data_args.add_selfies:
            # insert product to the instruction end
            instruction = self.construct_instruct_question(product)
        else:
            instruction = raw['instruction'] + f" The reaction is {input}"

        instruction = "<image>\n" + instruction

            
        if self.data_args.graph_tower == "himol":
            graph = MolGraph(reactant_smiles)
        else:
            graph=smiles2graph(reactant_smiles)
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        if graph is not None:
            data_dict['graphs'] = graph
            assert -200 in data_dict["input_ids"]
            
        return data_dict
    
    def preprocess_retrosyn(self, raw):
        instruction = raw['instruction']
        if self.data_args.add_selfies:
            instruction += f" The product is: {raw['input']}"

        instruction = "<image>\n" + instruction
        
        input_selfies, output_selfies = raw['input'], raw['output']
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(input_selfies)
        
        if self.data_args.graph_tower == "himol":
            graph = MolGraph(reactant_smiles)
        else:
            # graph data containes the information of the first SELFIES 
            graph=smiles2graph(reactant_smiles)
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output_selfies}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        if graph is not None:
            data_dict['graphs'] = graph
            assert -200 in data_dict["input_ids"]
        
        return data_dict
    
    def preprocess_property(self, raw):
        instruction = raw['instruction']
        if self.data_args.add_selfies:
            instruction += f" The compound SELFIES sequence is: {raw['input']}"

        instruction = "<image>\n" + instruction
        
        input_selfies, target = raw['input'], str(raw['output'])
        # convert input selfies to smiles for building graph
        if self.data_args.graph_tower == "himol":
            graph = MolGraph(self.selfies2smiles(input_selfies))
        else:
            # graph data containes the information of the first SELFIES 
            graph=smiles2graph(self.selfies2smiles(input_selfies))
            
        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": target}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))

        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        if graph is not None:
            data_dict['graphs'] = graph
            assert -200 in data_dict["input_ids"]
        
        return data_dict
    
    def proprocess_molcap(self, raw):
        instruction = random.choice(self.question_pool)
        input = raw['input']
        output = raw['output']

        if self.data_args.add_selfies:
            instruction += f" The compound SELFIES sequence is: {input}"

        instruction = "<image>\n" + instruction

        if self.data_args.graph_tower == "himol":
            graph = MolGraph(self.selfies2smiles(input))  # 数据处理部分没问题
        else:
            graph = smiles2graph(self.selfies2smiles(input))

        messages = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": output}
            ]
        ]

        data_dict = apply_chat_template(messages, self.tokenizer, has_image=(graph is not None))
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # graph exist in the data
        if graph is not None:
            data_dict['graphs'] = graph
            assert -200 in data_dict["input_ids"]

        return data_dict
        
        
    def __getitem__(self, index):
        data_dict = self.data_list[index]
        real_data = data_dict["real"]
        generated_data = data_dict["generated"]
        processor = self.MAP_TO_PROCESS[real_data["metadata"]["task"]]
        batch = {
            "real": processor(real_data),
            "generated": processor(generated_data)
        }
        
        return batch
    
    
@dataclass       
class SPINMultiTaskCollator(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        real_batch_list, generated_batch_list = tuple([instance[key] for instance in instances] for key in ["real", "generated"])
        real_batch = self.processing(real_batch_list)
        generated_batch = self.processing(generated_batch_list)
        
        batch = {
            "real": real_batch,
            "generated": generated_batch
        }
        
        return batch
        
    
    def processing(self, instances):
        input_ids, labels = self._extract_tensors(instances, ("input_ids", "labels"))
        
        input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        labels = self._pad_sequence(labels, IGNORE_INDEX)
        batch = {
            'input_ids': input_ids[:, :self.tokenizer.model_max_length],
            'labels': labels[:, :self.tokenizer.model_max_length],
            'attention_mask': input_ids[:, :self.tokenizer.model_max_length].ne(self.tokenizer.pad_token_id),
        }
        if 'graphs' in instances[0]:
            batch['graphs'] = Batch.from_data_list(
                [self._convert_dict_to_Data(instance["graphs"]) for instance in instances]
            )
        return batch
    
    def _extract_tensors(self, instances, keys: Tuple[str, str]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        return tuple([instance[key] for instance in instances] for key in keys)

    def _pad_sequence(self, sequence: List[torch.Tensor], padding_value: int) -> torch.Tensor:
        return torch.nn.utils.rnn.pad_sequence(sequence, batch_first=True, padding_value=padding_value)

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
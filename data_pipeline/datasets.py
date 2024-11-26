from abc import ABC
import transformers
import json
import selfies
import torch
from torch_geometric.data import Batch, Data
from torch.utils.data import Dataset
from data_pipeline.data_structure import MolGraph
from data_pipeline.mol_utils import smiles2graph
from arguments import DataArguments
from data_pipeline.preprocess_engine import apply_chat_template
from transformers import PreTrainedTokenizer
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple, List, Any
from torch.utils import data
import random
import pandas as pd
IGNORE_INDEX = -100

 
 
@dataclass       
class GraphDatasetCollator(object):
    """Collate graph-QA examples for supervised fine-tuning."""
    
    tokenizer: PreTrainedTokenizer
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = self._extract_tensors(instances, ("input_ids", "labels"))
        
        input_ids = self._pad_sequence(input_ids, self.tokenizer.pad_token_id)
        labels = self._pad_sequence(labels, IGNORE_INDEX)

        batch = {
            'input_ids': input_ids[:, :self.tokenizer.model_max_length],
            'labels': labels[:, :self.tokenizer.model_max_length],
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
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


class MetaGraphDataset(Dataset):
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer,
        data_args: DataArguments
    ) -> None:
        super().__init__()
        with open(data_args.data_path, "rb") as f:
            self.list_data_dict = json.load(f)
            f.close()

        self.tokenizer = tokenizer
        print(f"Total number of samples: {self.__len__()}")
        if data_args.is_training:
            self.filter_for_training()
            print(f"Filtered {self.__len__()} for training")
        else:
            self.filter_for_test()
            print(f"Filtered {self.__len__()} for test")

        self.data_args = data_args
        
    def selfies2smiles(self, selfies_str: str) -> str | None:
        try:
            smiles_str = selfies.decoder(selfies_str)
        except:
            smiles_str = None
            
        return smiles_str
    
    def filter_for_training(self) -> None:
        self.list_data_dict = [raw for raw in self.list_data_dict if raw['metadata']['split'] == 'train']
    
    def filter_for_test(self) -> None:
        self.list_data_dict =  [raw for raw in self.list_data_dict if raw['metadata']['split'] == 'test']
        
    def __len__(self) -> int:
        return len(self.list_data_dict)
    
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        pass
    
    
class PretrainMolDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__()
        list_data_dict = pd.read_csv(data_args.data_path)

        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        print(f"Total number of samples: {self.__len__()}")
        print("====Pretrain Molecule Description Dataset====")
        self.question_pool = [
        'Could you give me a brief overview of this molecule?',
        'Could you provide a description of this molecule?',
        'Describe this molecule.',
        'Please give me some details about this molecule.',
        'Provide a brief overview of this molecule.',
        'Provide a description of this molecule.',
        'What can you tell me about this molecule?'
        ]
        
    def __len__(self):
        return len(self.list_data_dict)
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        smiles, description = self.list_data_dict["SMILES"][i], self.list_data_dict["Description"][i]
        
        instruction = random.choice(self.question_pool)
        instruction = "<image>\n" + instruction
        
        message = [
            [
                {"from": "human", "value": instruction},
                {"from": "gpt", "value": description}
            ]
        ]
        
        graph_for_molecule = MolGraph(smiles) \
        if self.data_args.graph_tower == "himol" else smiles2graph(smiles)
        
        data_dict = apply_chat_template(message, self.tokenizer, graph_for_molecule is not None)
        
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])

        # Add graph or check multimodal settings
        if graph_for_molecule is not None:
            data_dict['graphs'] = graph_for_molecule
            assert -200 in data_dict["input_ids"], "Input IDs missing expected <image> token"
            
        # ids = data_dict["input_ids"]
        # ids = list(ids.detach().numpy())
        # print("ids", ids)
        # print("labels", data_dict["labels"])
        # print(self.tokenizer.decode([518, 29949, 3816]))
        # ids.remove(-200)
        # print([self.tokenizer.decode(ids)])
        
        # exit(0)

        return data_dict
        
        
    

class ForwardPredDataset(MetaGraphDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__(tokenizer, data_args)
        print("=====Forward Prediction dataset=====")
        
    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        # 1. Get sample
        raw = self.list_data_dict[i]

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
            
        # ids = data_dict["input_ids"]
        # ids = list(ids.detach().numpy())
        # print("ids", ids)
        # print("labels", data_dict["labels"])
        # print(self.tokenizer.decode([518, 29949, 3816]))
        # ids.remove(-200)
        # print([self.tokenizer.decode(ids)])
        
        # exit(0)

        return data_dict
    
    
class ReagentPredDataset(MetaGraphDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__(tokenizer, data_args)
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
    
    
class RetrosynDataset(MetaGraphDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__(tokenizer, data_args)
        print("=====Retrosynthesis dataset=====")
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
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
    
    
class PropertyPredDataset(MetaGraphDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__(tokenizer, data_args)
        print("=====Property Prediction dataset=====")
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
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
    
class MolcapDataset(MetaGraphDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__(tokenizer, data_args)
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
        
    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
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
    

class CatalystPredDataset(MetaGraphDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__(tokenizer, data_args)
        print("=====Catalyst Prediction dataset=====")

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if self.data_args.add_selfies:
            # insert product to the instruction end
            instruction = raw['instruction'] + f" The reaction is {input}."
        else:
            instruction = raw['instruction']
        # elif len(input) > 1:
        #     instruction = ""
        #     instruction += f" The other joint reactants are: {','.join(input[1:])}"

        instruction = "<image>\n" + instruction

        if self.data_args.graph_tower == "himol":
            graph = MolGraph(reactant_smiles)
        else:
            graph = smiles2graph(reactant_smiles)

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


class SolventPredDataset(MetaGraphDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__(tokenizer, data_args)
        print("=====Solvent Prediction dataset=====")

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if self.data_args.add_selfies:
            # insert product to the instruction end
            instruction = raw['instruction'] + f" The reaction is {input}."
        else:
            instruction = raw['instruction']
        # elif len(input) > 1:
        #     instruction = ""
        #     instruction += f" The other joint reactants are: {','.join(input[1:])}"

        instruction = "<image>\n" + instruction

        if self.data_args.graph_tower == "himol":
            graph = MolGraph(reactant_smiles)
        else:
            graph = smiles2graph(reactant_smiles)

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


class YieldRegressionDataset(MetaGraphDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> None:
        super().__init__(tokenizer, data_args)
        print("=====Yield Regression dataset=====")

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        raw = self.list_data_dict[i]
        input, output_selfies = raw['input'], raw['output']
        # input: "reactant>>product"
        reactant, product = input.split(">>")
        # convert input selfies to smiles for building graph
        reactant_smiles = self.selfies2smiles(reactant)
        if self.data_args.add_selfies:
            # insert product to the instruction end
            instruction = raw['instruction'] + f" The reaction is {input}."
        else:
            instruction = raw['instruction']
        # elif len(input) > 1:
        #     instruction = ""
        #     instruction += f" The other joint reactants are: {','.join(input[1:])}"

        instruction = "<image>\n" + instruction

        if self.data_args.graph_tower == "himol":
            graph = MolGraph(reactant_smiles)
        else:
            graph = smiles2graph(reactant_smiles)

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
    
    
DATASET_MAP = {
    "forward_pred": ForwardPredDataset,
    "pub_chem": PretrainMolDataset,
    "reagent_pred": ReagentPredDataset,
    "retrosynthesis": RetrosynDataset,
    "molcap": MolcapDataset,
    "property_pred": PropertyPredDataset,
    "solvent_pred": SolventPredDataset,
    "catalyst_pred": CatalystPredDataset,
    "yields_regression": YieldRegressionDataset,
}
    
def build_dataset(tokenizer: PreTrainedTokenizer, data_args: DataArguments) -> Dict[str, Any]:
    dataset = DATASET_MAP[data_args.data_type](tokenizer=tokenizer, data_args=data_args)
    
    if data_args.split_eval:
        train_set, eval_set = data.random_split(dataset, [0.9, 0.1])
        print("=======Split eval======")
        print("Length of train:", len(train_set))
        print("Length of eval:", len(eval_set))
    else:
        train_set, eval_set = dataset, None

    return {
        "train_dataset": train_set,
        "eval_dataset": eval_set,
        "data_collator": GraphDatasetCollator(tokenizer=tokenizer)
    }
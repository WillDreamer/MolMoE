'''
Code from https://github.com/blender-nlp/MolT5

```bibtex
@article{edwards2022translation,
  title={Translation between Molecules and Natural Language},
  author={Edwards, Carl and Lai, Tuan and Ros, Kevin and Honke, Garrett and Ji, Heng},
  journal={arXiv preprint arXiv:2204.11817},
  year={2022}
}
```
'''

import argparse
import csv
import os
import numpy as np
import json
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import RDLogger
from nltk.translate.bleu_score import corpus_bleu
from Levenshtein import distance as lev
RDLogger.DisableLog('rdApp.*')
import selfies as sf
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers import AutoTokenizer
import nltk
from sklearn.metrics import mean_absolute_error


def sf_encode(selfies):
    try:
        smiles = sf.decoder(selfies)
        return smiles
    except Exception:
        return None

def convert_to_canonical_smiles(smiles):
    if smiles is None:
        return None
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is not None:
        canonical_smiles = Chem.MolToSmiles(molecule, isomericSmiles=False, canonical=True)
        return canonical_smiles
    else:
        return None

def build_evaluate_tuple(result:dict):
    # pred
    # func = lambda x: x.rsplit(']', 1)[0] + ']' if isinstance(x, str) else x
    func = lambda x: x
    result["pred_smi"] = convert_to_canonical_smiles(func(sf_encode(result["pred"])))
    # gt
    result["gt_smi"] = convert_to_canonical_smiles(sf_encode(result["gt"]))
    return result


def calc_fingerprints(input_file: str, morgan_r: int=2, eos_token='<|end|>'):
    outputs = []  # 只有合规的分子
    bad_mols = 0

    with open(input_file) as f:
        results = json.load(f)
        for i, result in enumerate(results):
            result['pred'] = result['pred'].split(eos_token)[0]
            result = build_evaluate_tuple(result)
            try:
                gt_smi = result['gt_smi']
                ot_smi = result['pred_smi']
                
                gt_m = Chem.MolFromSmiles(gt_smi)
                ot_m = Chem.MolFromSmiles(ot_smi)

                if ot_m == None: raise ValueError('Bad SMILES')
                outputs.append((result['prompt'], gt_m, ot_m))
            except:
                bad_mols += 1
    validity_score = len(outputs)/(len(outputs)+bad_mols)
    print('validity:', validity_score)


    MACCS_sims = []
    morgan_sims = []
    RDK_sims = []

    enum_list = outputs  # 只有合规的分子

    for i, (desc, gt_m, ot_m) in enumerate(enum_list):

        if i % 100 == 0:
            print(i, 'processed.')

        MACCS_sims.append(DataStructs.FingerprintSimilarity(MACCSkeys.GenMACCSKeys(gt_m), MACCSkeys.GenMACCSKeys(ot_m), metric=DataStructs.TanimotoSimilarity))
        RDK_sims.append(DataStructs.FingerprintSimilarity(Chem.RDKFingerprint(gt_m), Chem.RDKFingerprint(ot_m), metric=DataStructs.TanimotoSimilarity))
        morgan_sims.append(DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprint(gt_m,morgan_r), AllChem.GetMorganFingerprint(ot_m, morgan_r)))

    maccs_sims_score = np.mean(MACCS_sims)  # 这个指标只有合规的分子  
    # np.sum(MACCS_sims) / len(outputs) + bad_mols
    rdk_sims_score = np.mean(RDK_sims)
    morgan_sims_score = np.mean(morgan_sims)

    print('Average MACCS Similarity:', maccs_sims_score)
    print('Average RDK Similarity:', rdk_sims_score)
    print('Average Morgan Similarity:', morgan_sims_score)
    return validity_score, maccs_sims_score, rdk_sims_score, morgan_sims_score
        
        
def calc_mol_trans(input_file, eos_token='<|end|>'):
    outputs = []

    with open(input_file) as f:
        results = json.load(f)
        for i, result in enumerate(results):
            result['pred'] = result['pred'].split(eos_token)[0]
            result = build_evaluate_tuple(result)
            gt_self = result['gt']
            ot_self = result['pred']
            gt_smi = result['gt_smi']
            ot_smi = result['pred_smi']
            if ot_smi is None:
                continue
            outputs.append((result['prompt'], gt_self, ot_self, gt_smi, ot_smi))

    bleu_self_scores = []
    bleu_smi_scores = []

    references_self = []
    hypotheses_self = []
    
    references_smi = []
    hypotheses_smi = []

    for i, (des, gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        if i % 100 == 0:
            print(i, 'processed.')

        gt_self_tokens = [c for c in gt_self]
        out_self_tokens = [c for c in ot_self]

        references_self.append([gt_self_tokens])
        hypotheses_self.append(out_self_tokens)
        
        if ot_smi is None:
            continue
        
        gt_smi_tokens = [c for c in gt_smi]
        ot_smi_tokens = [c for c in ot_smi]

        references_smi.append([gt_smi_tokens])
        hypotheses_smi.append(ot_smi_tokens)
        

    # BLEU score
    bleu_score_self = corpus_bleu(references_self, hypotheses_self)
    print(f'SELFIES BLEU score', bleu_score_self)

    references_self = []
    hypotheses_self = []
    
    references_smi = []
    hypotheses_smi = []

    levs_self = []
    levs_smi = []

    num_exact = 0

    bad_mols = 0

    for i, (des, gt_self, ot_self, gt_smi, ot_smi) in enumerate(outputs):

        hypotheses_self.append(ot_self)
        references_self.append(gt_self)

        hypotheses_smi.append(ot_smi)
        references_smi.append(gt_smi)
        
        try:
            m_out = Chem.MolFromSmiles(ot_smi)
            m_gt = Chem.MolFromSmiles(gt_smi)

            if Chem.MolToInchi(m_out) == Chem.MolToInchi(m_gt): num_exact += 1
            #if gt == out: num_exact += 1 #old version that didn't standardize strings
        except:
            bad_mols += 1

        levs_self.append(lev(ot_self, gt_self))
        levs_smi.append(lev(ot_smi, gt_smi))


    # Exact matching score
    exact_match_score = num_exact/(i+1)
    print('Exact Match:')
    print(exact_match_score)

    # Levenshtein score
    levenshtein_score_smi = np.mean(levs_smi)
    print('SMILES Levenshtein:')
    print(levenshtein_score_smi)
        
    validity_score = 1 - bad_mols/len(outputs)
    print('validity:', validity_score)


def calc_mocap_metrics(input_file, eos_token, tokenizer: PreTrainedTokenizer):
    from nltk.translate.bleu_score import corpus_bleu
    from nltk.translate.meteor_score import meteor_score
    from rouge_score.rouge_scorer import RougeScorer
    
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'])
    
    output_tokens = []
    gt_tokens = []
    meteor_scores = []
    rouge_scores = []
    
    with open(input_file, 'r') as f:
        file = json.load(f)
        f.close()
        
    for i, log in tqdm(enumerate(file)):
        # cid, pred, gt = log['cid'], log['text'], log['gt']
        pred, gt = log['pred'], log['gt']
        output_tokens.append(tokenizer.tokenize(pred, truncation=True, max_length=512, padding='max_length'))
        # print(output_tokens)
        output_tokens[i] = list(filter((eos_token).__ne__, output_tokens[i]))
        output_tokens[i] = list(filter((tokenizer.pad_token).__ne__, output_tokens[i]))
        gt_tokens.append(tokenizer.tokenize(gt, truncation=True, max_length=512, padding='max_length'))
        gt_tokens[i] = list(filter((eos_token).__ne__, gt_tokens[i]))
        gt_tokens[i] = [list(filter((tokenizer.pad_token).__ne__, gt_tokens[i]))]
        meteor_scores.append(meteor_score(gt_tokens[i], output_tokens[i]))
        rouge_scores.append(scorer.score(gt, pred))
            
    bleu2 = corpus_bleu(gt_tokens, output_tokens, weights=[0.5, 0.5])
    bleu4 = corpus_bleu(gt_tokens, output_tokens, weights=(0.25, 0.25, 0.25, 0.25))
    
    # extract top-10 meteor scores
    meteor_scores = np.array(meteor_scores)
    Start,K = 500,100
    idxes = np.argsort(meteor_scores)[::-1][Start:Start+K]
    # cids = [log['cid'] for i,log in enumerate(json.load(open(input_file, "r"))) if i in idxes]
    # cids.sort(key=lambda x: int(x))
    
    final = {
        "BLEU-2": bleu2,
        "BLEU-4": bleu4,
        "Meteor": np.mean(meteor_scores),
        "ROUGE-1": np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]),
        "ROUGE-2": np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]),
        "ROUGE-L": np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]),
    }
    
    print(final)
    
    return final
    
    
def compute_mae(eval_result_file:str, eos_token):
    data_dict = {
        "homo_gts": [],
        "homo_preds": [],
        "lumo_gts": [],
        "lumo_preds": [],
        "gap_gts": [],
        "gap_preds": []
    }
    with open(eval_result_file) as f:
        results = json.load(f)
        
        for i, result in enumerate(results):
            pred = result['pred'].split(eos_token)[0]
            gt = result['gt']
            prompt = result['prompt']
            try:
                gt = float(gt)
                pred = float(pred)
            except:
                continue
            if "HOMO" in prompt and "LUMO" not in prompt:
                data_dict["homo_gts"].append(gt)
                data_dict["homo_preds"].append(pred)
            elif "HOMO" not in prompt and "LUMO" in prompt:
                data_dict["lumo_gts"].append(gt)
                data_dict["lumo_preds"].append(pred)
            elif "HOMO" in prompt and "LUMO" in prompt:
                data_dict["gap_gts"].append(gt)
                data_dict["gap_preds"].append(pred)
            else:
                raise NotImplementedError()
            
        homo_err = mean_absolute_error(data_dict["homo_gts"], data_dict["homo_preds"])
        lumo_err = mean_absolute_error(data_dict["lumo_gts"], data_dict["lumo_preds"])
        gap_err = mean_absolute_error(data_dict["gap_gts"], data_dict["gap_preds"])
        average = mean_absolute_error(data_dict["homo_gts"]+data_dict["lumo_gts"]+data_dict["gap_gts"], data_dict["homo_preds"]+data_dict["lumo_preds"]+data_dict["gap_preds"])
                
        print("HOMO MAE:", homo_err)
        print("LUMO MAE", lumo_err)
        print("GAP MAE", gap_err)
        print("Average:", average) 
        
        
if __name__ == "__main__":
    file_path = "/root/autodl-tmp/MolMoE/eval_result/fwd_pred-lora-phi-old-projector-bug-fixed-6ep-answer.json"
    calc_fingerprints(file_path, eos_token='<|endoftext|>')
    calc_mol_trans(file_path, eos_token='<|endoftext|>')
    # nltk.download('wordnet')
    # exit(0)
    # tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/MoleculeMoE/MolMoE/checkpoints/phi3-mini")
    # metrics = calc_mocap_metrics("/root/autodl-tmp/MoleculeMoE/MolMoE/eval_result/molcap_eval.json", '<|endoftext|>', tokenizer)
    
    # print(metrics)
    
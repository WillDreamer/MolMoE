"""
ref from https://github.com/UCSD-AI4H/drugchat/blob/main/dataset/smiles2graph.py
"""
from rdkit import Chem
import numpy as np
import json
import pickle
import os
from tqdm import tqdm
import random
from typing import Dict
from rdkit.Chem.rdchem import BondType, BondDir, ChiralType
import selfies as sf
import torch
from rdkit.Chem import BRICS

BOND_TYPE = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2, BondType.AROMATIC: 3}
BOND_DIR = {BondDir.NONE: 0, BondDir.ENDUPRIGHT: 1, BondDir.ENDDOWNRIGHT: 2}
CHI = {ChiralType.CHI_UNSPECIFIED: 0, ChiralType.CHI_TETRAHEDRAL_CW: 1, ChiralType.CHI_TETRAHEDRAL_CCW: 2, ChiralType.CHI_OTHER: 3}

def bond_dir(bond):
    d = bond.GetBondDir()
    return BOND_DIR[d]


def bond_type(bond):
    t = bond.GetBondType()
    return BOND_TYPE[t]


def atom_chiral(atom):
    c = atom.GetChiralTag()
    return CHI[c]


def atom_to_feature(atom):
    num = atom.GetAtomicNum() - 1
    if num == -1:
        # atom.GetAtomicNum() is 0, which is the generic wildcard atom *, may be used to symbolize an unknown atom of any element.
        # See https://biocyc.org/help.html?object=smiles
        num = 118  # normal num is [0, 117], so we use 118 to denote wildcard atom *
    return [num, atom_chiral(atom)]


def bond_to_feature(bond):
    return [bond_type(bond), bond_dir(bond)]


def smiles2graph(smiles_string)->Dict:
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    # 大改
    mol = Chem.MolFromSmiles(smiles_string)

    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 2
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph


def construct_instruct_question(selfies_str:str=None):
    """
    Construct instruct question for each graph
    """
    question_pools = [
        'Could you give me a brief overview of this molecule?',
        'Could you provide a description of this molecule?',
        'Describe this molecule.',
        'Please give me some details about this molecule.',
        'Provide a brief overview of this molecule.',
        'Provide a description of this molecule.',
        'What can you tell me about this molecule?'
    ]
    question = random.choice(question_pools)
    if selfies_str is not None:
        question += f" The compound SELFIES sequence is: {selfies_str}."
    if random.random() < 0.5:
        question = "<image>\n" + question
    else:
        question = question + "\n<image>"
    return question


def convert_chembl(qa_json, out_dir=None):
    assert os.path.exists(qa_json), f"{qa_json} not exists"
    qa_name = os.path.basename(qa_json).split(".")[0]
    with open(qa_json, "rt") as f:
        js = json.load(f)
    out = []
    for smi, rec in tqdm(js.items()):
        if len(rec) == 0:
            continue
        graph = smiles2graph(smi)
        for question, answer in rec:
            out.append({
                "graph": graph,
                "conversations": [
                    {"from": "human", "value": construct_instruct_question() },
                    {"from": "gpt", "value": answer}
                ],
            })
    print(f"Successfully convert {len(out)} samples.")

    if out_dir is None:
        out_dir = os.path.dirname(qa_json)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    with open(os.path.join(out_dir, qa_name+'.pkl'), "wb") as f:
        pickle.dump(out, f)


def convert_chebi20(txt, out_dir=None, add_selfies=False):
    assert os.path.exists(txt), f"{txt} not exists"
    qa_name = os.path.basename(txt).split(".")[0]
    out = []
    with open(txt, "rt") as f:
        f.readline()
        for i, line in enumerate(f.readlines()):
            cid, smi, desc = line.strip().split("\t")
            selfies_str = None
            if add_selfies:
                try:
                    selfies_str = sf.encoder(smi)
                except:
                    selfies_str = ""
            graph = smiles2graph(smi)
            out.append({
                "graph": graph,
                "conversations": [
                    {"from": "human", "value": construct_instruct_question(selfies_str) },
                    {"from": "gpt", "value": desc}
                ],
            })
    print(f"Successfully convert {len(out)} samples.")
    if out_dir is None:
        out_dir = os.path.dirname(txt)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if add_selfies:
        qa_name += "+selfies"
    with open(os.path.join(out_dir, qa_name+'.pkl'), "wb") as f:
        pickle.dump(out, f)


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol):
    try:
        smiles = get_smiles(mol)
        mol = get_mol(smiles)
    except Exception as e:
        return None
    return mol


def copy_atom(atom):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))  # 创建一个可操作的空分子
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
    return new_mol


def get_clique_mol(mol, atoms):
    try:
        # 确保分子的结构已经Kekulized，使得所有原子和键的类型都明确。
        Chem.Kekulize(mol, clearAromaticFlags=True)
        # 从分子中获取特定原子集的SMILES表示。
        smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
        # 从SMILES字符串创建分子对象，此处关闭自动校正，因为我们希望先手动处理。
        new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # 创建一个可编辑的分子副本。
        new_mol = copy_edit_mol(new_mol).GetMol()
        # 尝试校正分子结构，如果失败则返回None。
        new_mol = sanitize(new_mol)
        if new_mol is None:
            raise Exception("Sanitization failed.")

        # 返回处理后的分子。
        return new_mol
    except Exception as e:
        # 在任何错误的情况下返回空的2D分子。
        return Chem.MolFromSmiles('')


def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []
    breaks = []
    # 获取所有分子间的连接关系
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  # 获取

    # 即: 断掉BRICS算法中获取的键
    res = list(BRICS.FindBRICSBonds(mol))
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]])

    # merge cliques
    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0:
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    # 一个cliques对应一个mol
    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)  # 若为空, 则进入到下一个循环
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr) > 1:
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
                    cliques[i] = list(set(cliques[i]) - set(list(ring)))

    cliques = [c for c in cliques if n_atoms > len(c) > 0]

    return cliques


allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)),  # 元素周期表序号
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list': [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],  # 原子的手性
    'possible_hybridization_list': [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds': [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC,
    ],
    'possible_bond_dirs': [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_inring': [None, False, True]
}


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # Chem.Kekulize(mol)
    return mol


# mol.x, mol.edge_index, mol.edge_attr, mol.num_part
class MolGraph(object):

    def __init__(self, smiles):
        self.smiles = smiles
        self.mol = get_mol(smiles)  # 获取2D分子

        '''
        #Stereo Generation
        mol = Chem.MolFromSmiles(smiles)
        self.smiles3D = Chem.MolToSmiles(mol, isomericSmiles=True)
        self.smiles2D = Chem.MolToSmiles(mol)
        self.stereo_cands = decode_stereo(self.smiles2D)
        '''

        # 有的会导致atom.GetAtomicNum()为0
        atom_features_list = []

        # 出现的问题: index中的值有时为0
        for atom in self.mol.GetAtoms():
            # get atom features
            atom_feature = [allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [allowable_features[
                                             'possible_degree_list'].index(atom.GetDegree())]

            atom_features_list.append(atom_feature)
        self.x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

        # bonds
        num_bond_features = 2  # bond type, bond direction
        if len(self.mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in self.mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = [allowable_features['possible_bonds'].index(
                    bond.GetBondType())] + [allowable_features['possible_bond_inring'].index(
                    bond.IsInRing())]

                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            self.edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            self.edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
                                                  dtype=torch.long)
        else:
            self.edge_index_nosuper = torch.empty((2, 0), dtype=torch.long)  # edgeCOO索引，[[row索引],[col索引]]
            self.edge_attr_nosuper = torch.empty((0, num_bond_features), dtype=torch.long)

        # add super node
        num_atoms = self.x_nosuper.size(0)

        # 仅用于建立super_node
        super_x = torch.tensor([[119, 0]]).to(self.x_nosuper.device)

        # add motif
        cliques = motif_decomp(self.mol)  # 用于添加motif
        num_motif = len(cliques)
        if num_motif > 0:
            # 初始化motif, 以及
            motif_x = torch.tensor([[120, 0]]).repeat_interleave(num_motif, dim=0).to(self.x_nosuper.device)
            self.x = torch.cat((self.x_nosuper, motif_x, super_x), dim=0)

            motif_edge_index = []
            for k, motif in enumerate(cliques):
                motif_edge_index = motif_edge_index + [[i, num_atoms + k] for i in motif]
            motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(
                self.edge_index_nosuper.device)

            super_edge_index = [[num_atoms + i, num_atoms + num_motif] for i in range(num_motif)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(
                self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

            motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 2)
            motif_edge_attr[:, 0] = 6  # bond type for self-loop edge
            motif_edge_attr = motif_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)

            super_edge_attr = torch.zeros(num_motif, 2)
            super_edge_attr[:, 0] = 5  # bond type for self-loop edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim=0)

            # num_atoms, num_motif, num_graph
            self.num_part = (num_atoms, num_motif, 1)

        else:
            self.x = torch.cat((self.x_nosuper, super_x), dim=0)

            super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
            super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(
                self.edge_index_nosuper.device)
            self.edge_index = torch.cat((self.edge_index_nosuper, super_edge_index), dim=1)

            super_edge_attr = torch.zeros(num_atoms, 2)
            super_edge_attr[:, 0] = 5  # bond type for self-loop edge
            super_edge_attr = super_edge_attr.to(self.edge_attr_nosuper.dtype).to(self.edge_attr_nosuper.device)
            self.edge_attr = torch.cat((self.edge_attr_nosuper, super_edge_attr), dim=0)

            self.num_part = (num_atoms, 0, 1)

    def size_node(self):
        return self.x.size()[0]

    def size_edge(self):
        return self.edge_attr.size()[0]

    def size_atom(self):
        return self.x_nosuper.size()[0]

    def size_bond(self):
        return self.edge_attr_nosuper.size()[0]


if __name__ == '__main__':
    # graph = smiles2graph('O1C=C[C@H]([C@H]1O2)c3c2cc(OC)c4c3OC(=O)C5=C4CCC(=O)5')
    # print(graph)
    # qa_json = '/comp_robot/rentianhe/caohe/AIDD/DATA/MolFM/pubcgraphemsft_desc/test.json'
    # convert_chembl(qa_json)

    for split in ['train', 'test', 'validation']:
        txt = f'/cto_labs/AIDD/DATA/MolT5/ChEBI-20_data/{split}.txt'
        convert_chebi20(txt, add_selfies=True)
# prepare_dataset & SubgraphGenerator & Heterogeneous augmented graph
import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from copy import copy, deepcopy
import multiprocessing as mp
import torch
from torch.utils.data import Dataset

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize.rdMolStandardize import StandardizeSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, ChemicalFeatures, Fragments

import logging
logger = logging.getLogger(__name__)

##### Featurizer #####
allowable_features = {
	'possible_atomic_num_list': list(range(1, 119)),
	'simple_atomic_num_list': [1, 5, 6, 7, 8, 9, 11, 13, 14, 15, 16, 17, 19, 30, 33, 34, 35, 53, 'other'], # 19
	'possible_chirality_list':        [
		Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
		Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
		Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
		Chem.rdchem.ChiralType.CHI_OTHER
	],  # 4
	'possible_hybridization_list':    [
		Chem.rdchem.HybridizationType.S,
		Chem.rdchem.HybridizationType.SP,
		Chem.rdchem.HybridizationType.SP2,
		Chem.rdchem.HybridizationType.SP3,
		Chem.rdchem.HybridizationType.SP3D,
		Chem.rdchem.HybridizationType.SP3D2,
		Chem.rdchem.HybridizationType.UNSPECIFIED
	],  # 7
	'possible_numH_list':             [0, 1, 2, 3, 4], # 1
	'possible_aromatic': [False, True],
	'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
	'possible_degree_list':           [0, 1, 2, 3, 4, 'other'],
	'possible_formal_charge': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
	'possible_bonds':                 [
		Chem.rdchem.BondType.SINGLE,
		Chem.rdchem.BondType.DOUBLE,
		Chem.rdchem.BondType.TRIPLE,
		Chem.rdchem.BondType.AROMATIC,
		None
	],
	'possible_bond_dirs':             [  # only for double bond stereo information
		Chem.rdchem.BondDir.NONE,
		Chem.rdchem.BondDir.ENDUPRIGHT,
		Chem.rdchem.BondDir.ENDDOWNRIGHT
	],
	'possible_is_conjugated': [False, True],
	'possible_is_in_ring': [False, True],
	'possible_motif_atom_num': list(range(1, 18)), # 0 ~ 17
	'possible_motif_ring_num': list(range(0, 6)), # 0 ~ 5
	'possible_motif_hetero_num': list(range(0, 8)), # 0 ~ 7
}

def one_hot_enc(x, allowable_set):
	if 'other' in allowable_set:
		return [float(x == s) for s in allowable_set[:-1]] + [float(x not in allowable_set[:-1])]
	else:
		return [float(x == s) for s in allowable_set]

# For Atom-level 
def atom_featurizer(atom, onehot=True):
	if onehot:
		atomic_num = one_hot_enc(atom.GetAtomicNum(), allowable_features['simple_atomic_num_list'])
		atom_feat = atomic_num + \
					one_hot_enc(atom.GetTotalDegree(), allowable_features['possible_degree_list']) + \
					one_hot_enc(atom.GetChiralTag(), allowable_features['possible_chirality_list']) + \
					[allowable_features['possible_formal_charge'].index(atom.GetFormalCharge())
					if atom.GetFormalCharge() in allowable_features['possible_formal_charge'] else len(allowable_features['possible_formal_charge'])] + \
					one_hot_enc(atom.GetHybridization(), allowable_features['possible_hybridization_list']) + \
					one_hot_enc(atom.GetTotalNumHs(), allowable_features['possible_numH_list']) + \
					[allowable_features['possible_aromatic'].index(atom.GetIsAromatic())]
	else:
		# Use safe index lookup with fallback to 'other' or last index
		atomic_num_val = atom.GetAtomicNum()
		if atomic_num_val in allowable_features['simple_atomic_num_list']:
			atomic_num = [allowable_features['simple_atomic_num_list'].index(atomic_num_val)]
		else:
			atomic_num = [len(allowable_features['simple_atomic_num_list']) - 1]  # 'other' index
		
		degree_val = atom.GetTotalDegree()
		if degree_val in allowable_features['possible_degree_list']:
			degree_idx = allowable_features['possible_degree_list'].index(degree_val)
		else:
			degree_idx = len(allowable_features['possible_degree_list']) - 1  # 'other' index
		
		chirality_val = atom.GetChiralTag()
		if chirality_val in allowable_features['possible_chirality_list']:
			chirality_idx = allowable_features['possible_chirality_list'].index(chirality_val)
		else:
			chirality_idx = 0  # Default to CHI_UNSPECIFIED
		
		charge_val = atom.GetFormalCharge()
		if charge_val in allowable_features['possible_formal_charge']:
			charge_idx = allowable_features['possible_formal_charge'].index(charge_val)
		else:
			charge_idx = len(allowable_features['possible_formal_charge'])  # Out of range index
		
		hybridization_val = atom.GetHybridization()
		if hybridization_val in allowable_features['possible_hybridization_list']:
			hybridization_idx = allowable_features['possible_hybridization_list'].index(hybridization_val)
		else:
			hybridization_idx = len(allowable_features['possible_hybridization_list']) - 1  # UNSPECIFIED
		
		numH_val = atom.GetTotalNumHs()
		if numH_val in allowable_features['possible_numH_list']:
			numH_idx = allowable_features['possible_numH_list'].index(numH_val)
		else:
			numH_idx = len(allowable_features['possible_numH_list']) - 1  # Max value
		
		aromatic_val = atom.GetIsAromatic()
		aromatic_idx = allowable_features['possible_aromatic'].index(aromatic_val)
		
		atom_feat = atomic_num + [degree_idx, chirality_idx, charge_idx, hybridization_idx, numH_idx, aromatic_idx]
	return atom_feat 


def bond_featurizer(bond, onehot=True):
	if onehot:
		bond_feat = one_hot_enc(bond.GetBondType(), allowable_features['possible_bonds']) + \
					one_hot_enc(bond.GetBondDir(), allowable_features['possible_bond_dirs']) + \
					[allowable_features['possible_is_conjugated'].index(bond.GetIsConjugated())] + \
					[allowable_features['possible_is_in_ring'].index(bond.IsInRing())]
	else:
		bond_feat = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
					[allowable_features['possible_bond_dirs'].index(bond.GetBondDir())] + \
					[allowable_features['possible_is_conjugated'].index(bond.GetIsConjugated())] + \
					[allowable_features['possible_is_in_ring'].index(bond.IsInRing())]
	return bond_feat 
	
	
### For Motif-level
declist = Descriptors.descList
calc = {}
for (i,j) in declist:
	calc[i] = j

def get_motif_element(mol):
	atom_symbol={'C':0,'H':0,'O':0,'N':0,'P':0,
				'S':0,'F':0,'CL':0,'Br':0,'other':0,}
	feat_element =[]
	for atom in mol.GetAtoms():
		if atom.GetSymbol() in atom_symbol.keys():
			atom_symbol[atom.GetSymbol()]+=1
		else:
			atom_symbol['other']+=1
	for key,value in atom_symbol.items():
		feat_element+=[value]
	return feat_element

def maccskeys_emb(mol):
	return list(MACCSkeys.GenMACCSKeys(mol))

def motif_featurizer(motif_smi, vocab=False):
	motif = Chem.MolFromSmiles(motif_smi)
	motif_smi = Chem.MolToSmiles(motif)
	MOTIF_FEATURE_SIZE = 182
	if vocab:
		if motif_smi in vocab:
			motif_idx = vocab[motif_smi]
		else:
			motif_idx = len(vocab)
		motif_idx = [motif_idx]
		MOTIF_FEATURE_SIZE += 1
	else:
		motif_idx = []
		
	try:
		motif_feat= motif_idx+\
					[calc['TPSA'](motif)*0.01]+[calc['MolLogP'](motif)]+\
					[calc['HeavyAtomMolWt'](motif)*0.01] +\
					[1 if motif.GetRingInfo().NumRings()>0 else 0]+\
					[motif.GetRingInfo().NumRings()]+\
					get_motif_element(motif) +\
					maccskeys_emb(motif)
	except:
		motif_feat = [0]*MOTIF_FEATURE_SIZE
	return motif_feat

def motif_bond_featurizer(features, onehot=True):
	if onehot:
		edge_feature = one_hot_enc(features[0], allowable_features['possible_bonds']) + \
					one_hot_enc(features[1], allowable_features['possible_bond_dirs']) + \
					[allowable_features['possible_is_conjugated'].index(features[2])] + \
					[allowable_features['possible_is_in_ring'].index(features[3])]
	else:
		edge_feature = [allowable_features['possible_bonds'].index(features[0])] + \
					[allowable_features['possible_bond_dirs'].index(features[1])] + \
					[allowable_features['possible_is_conjugated'].index(features[2])] + \
					[allowable_features['possible_is_in_ring'].index(features[3])]
	return edge_feature


# Chem_utils + Molecule 
##### Chem_utils #####
MAX_VALENCE = {'B': 3, 'Br':1, 'C':4, 'Cl':1, 'F':1, 'I':1, 'N':3, 'O':2, 'P':5, 'S':6, 'Se':4, 'Si':4}

# count atoms except hydrogens
def cnt_atom(smi, return_dict=False):
	atom_dict = { atom: 0 for atom in MAX_VALENCE }
	for i in range(len(smi)):
		symbol = smi[i].upper()
		next_char = smi[i+1] if i+1 < len(smi) else None
		if symbol == 'B' and next_char == 'r':
			symbol += next_char
		elif symbol == 'C' and next_char == 'l':
			symbol += next_char
		if symbol in atom_dict:
			atom_dict[symbol] += 1
	if return_dict:
		return atom_dict
	else:
		return sum(atom_dict.values())
	
def smi_to_mol(smiles: str, kekulize=False, sanitize=True):
	mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
	if mol is None:
		return None
	if kekulize:
		Chem.Kekulize(mol, True)
	return mol

def mol_to_smi(mol):
	if mol is None:
		return None
	return Chem.MolToSmiles(mol)

def get_clique_mol(mol, atoms):
	Chem.Kekulize(mol, clearAromaticFlags=True)
	smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
	new_mol = Chem.MolFromSmiles(smiles, sanitize=True)
	Chem.Kekulize(new_mol, clearAromaticFlags=True)
	return new_mol

organic_major_ish = {'C', 'O', 'N', 'F', 'Cl', 'Br', 'I', 'S', 'P', 'B', 'H'}
def get_submol(mol, atom_indices, kekulize=False):
	if len(atom_indices) == 1:
		atom_symbol = mol.GetAtomWithIdx(atom_indices[0]).GetSymbol()
		if atom_symbol == 'Si':
			atom_symbol = '[Si]'
		elif atom_symbol == 'Na':
			atom_symbol = '[Na+]'
		elif atom_symbol == 'H':
			atom_symbol = '[H]'
		elif atom_symbol not in organic_major_ish:
			atom_symbol = '[' + atom_symbol + ']'
		return smi_to_mol(atom_symbol, kekulize)
	aid_dict = { i: True for i in atom_indices }
	edge_indices = []
	for i in range(mol.GetNumBonds()):
		bond = mol.GetBondWithIdx(i)
		begin_aid = bond.GetBeginAtomIdx()
		end_aid = bond.GetEndAtomIdx()
		if begin_aid in aid_dict and end_aid in aid_dict:
			edge_indices.append(i)
	mol = Chem.PathToSubmol(mol, edge_indices)
	return mol

def get_submol_atom_map(mol, submol, group, kekulize=False):
	if len(group) == 1:
		return { group[0]: 0 }
	# special with N+ and N-
	for atom in submol.GetAtoms():
		if atom.GetSymbol() != 'N':
			continue
		if (atom.GetExplicitValence() == 3 and atom.GetFormalCharge() == 1) or atom.GetExplicitValence() < 3:
			atom.SetNumRadicalElectrons(0)
			atom.SetNumExplicitHs(2)
			
	matches = mol.GetSubstructMatches(submol)
	if len(matches) < 1:
		Chem.Kekulize(mol, clearAromaticFlags=True)
		Chem.Kekulize(submol, clearAromaticFlags=True)
		matches = mol.GetSubstructMatches(submol)
	old2new = { i: 0 for i in group }  # old atom idx to new atom idx
	found = False
	for m in matches:
		hit = True
		for i, atom_idx in enumerate(m):
			if atom_idx not in old2new:
				hit = False
				break
			old2new[atom_idx] = i
		if hit:
			found = True
			break
	assert found
	return old2new


##### Molecule #####
class SubgraphNode:
	'''
	The node representing a subgraph
	'''
	def __init__(self, smiles: str, pos: int, atom_mapping: dict, kekulize: bool):
		self.smiles = smiles
		self.pos = pos
		self.mol = smi_to_mol(smiles, kekulize)
		self.atom_mapping = copy(atom_mapping)
	
	def get_mol(self):
		'''return molecule in rdkit form'''
		return self.mol

	def get_atom_mapping(self):
		return copy(self.atom_mapping)

	def __str__(self):
		return f'''
					smiles: {self.smiles},
					position: {self.pos},
					atom map: {self.atom_mapping}
				'''

class SubgraphEdge:
	'''
	Edges between two subgraphs
	'''
	def __init__(self, src: int, dst: int, edges: list):
		self.edges = copy(edges)  # list of tuple (a, b, type) where the canonical order is used
		self.src = src
		self.dst = dst
		self.bond_type = edges[-1]
		self.dummy = False
		if len(self.edges) == 0:
			self.dummy = True
	
	def get_edges(self):
		return copy(self.edges)
	
	def get_num_edges(self):
		return len(self.edges)

	def __str__(self):
		return f'''
					src subgraph: {self.src}, dst subgraph: {self.dst},
					atom bonds: {self.edges}
				'''

class Molecule(nx.Graph):
	'''molecule represented in subgraph-level'''

	def __init__(self, smiles: str=None, groups: list=None, kekulize: bool=False):
		super().__init__()
		if smiles is None:
			return
		self.graph['smiles'] = smiles
		rdkit_mol = smi_to_mol(smiles, kekulize)
		# processing atoms
		aid2pos = {}
		for pos, group in enumerate(groups):
			subgraph_smi, subgraph = group
			subgraph_mol = get_clique_mol(rdkit_mol, subgraph)

			for aid in subgraph:
				aid2pos[aid] = pos 
			try:
				self.atom_mapping = get_submol_atom_map(rdkit_mol, subgraph_mol, subgraph, kekulize)
			except:
				subgraph_mol = get_submol(rdkit_mol, subgraph)
				self.atom_mapping = get_submol_atom_map(rdkit_mol, subgraph_mol, subgraph, kekulize)

			node = SubgraphNode(subgraph_smi, pos, self.atom_mapping, kekulize)
			self.add_node(node)
			
		# process edges
		edges_arr = [[[] for _ in groups] for _ in groups]  # adjacent
		for edge_idx in range(rdkit_mol.GetNumBonds()):
			bond = rdkit_mol.GetBondWithIdx(edge_idx)
			begin = bond.GetBeginAtomIdx()
			end = bond.GetEndAtomIdx()

			begin_subgraph_pos = aid2pos[begin]
			end_subgraph_pos = aid2pos[end]
			begin_mapped = self.nodes[begin_subgraph_pos]['subgraph'].atom_mapping[begin]
			end_mapped = self.nodes[end_subgraph_pos]['subgraph'].atom_mapping[end]

			bond_type = bond.GetBondType()
			bond_dir = bond.GetBondDir()
			bond_conj = bond.GetIsConjugated()
			bond_ring = bond.IsInRing()
			edges_arr[begin_subgraph_pos][end_subgraph_pos].append((begin_mapped, end_mapped, bond_type, bond_dir, bond_conj, bond_ring))
			edges_arr[end_subgraph_pos][begin_subgraph_pos].append((end_mapped, begin_mapped, bond_type, bond_dir, bond_conj, bond_ring))

		# add edges into the graph
		for i in range(len(groups)):
			for j in range(len(groups)):
				if not i < j or len(edges_arr[i][j]) == 0:
					continue
				edge = SubgraphEdge(i, j, edges_arr[i][j])
				self.add_edge(edge)
	
	@classmethod
	def from_nx_graph(cls, graph: nx.Graph, deepcopy=True):
		if deepcopy:
			graph = deepcopy(graph)
		graph.__class__ = Molecule
		return graph

	@classmethod
	def merge(cls, mol0, mol1, edge=None):
		# reorder
		node_mappings = [{}, {}]
		mols = [mol0, mol1]
		mol = Molecule.from_nx_graph(nx.Graph())
		for i in range(2):
			for n in mols[i].nodes:
				node_mappings[i][n] = len(node_mappings[i])
				node = deepcopy(mols[i].get_node(n))
				node.pos = node_mappings[i][n]
				mol.add_node(node)
			for src, dst in mols[i].edges:
				edge = deepcopy(mols[i].get_edge(src, dst))
				edge.src = node_mappings[i][src]
				edge.dst = node_mappings[i][dst]
				mol.add_edge(src, dst, connects=edge)
		# add new edge
		edge = deepcopy(edge)
		edge.src = node_mappings[0][edge.src]
		edge.dst = node_mappings[1][edge.dst]
		return mol

	def get_edge(self, i, j) -> SubgraphEdge:
		return self[i][j]['connects']
	
	def get_node(self, i) -> SubgraphNode:
		return self.nodes[i]['subgraph']

	def add_edge(self, edge: SubgraphEdge) -> None:
		src, dst = edge.src, edge.dst
		super().add_edge(src, dst, connects=edge)
	
	def add_node(self, node: SubgraphNode) -> None:
		n = node.pos
		super().add_node(n, subgraph=node)

	def subgraph(self, nodes: list):
		graph = super().subgraph(nodes)
		assert isinstance(graph, Molecule)
		return graph

	def __str__(self):
		desc = 'nodes: \n'
		for ni, node in enumerate(self.nodes):
			desc += f'{ni}:{self.get_node(node)}\n'
		desc += 'edges: \n'
		for src, dst in self.edges:
			desc += f'{src}-{dst}:{self.get_edge(src, dst)}\n'
		return desc

	
##### Pretrain dataset #####
# smi -> atom-level dgl.graph
def Smi2AtomGraph(smi, atom_vocab=False):
	mol = Chem.MolFromSmiles(smi)
	if mol == None: 
		return None
	
	# Define connection between edges 
	src = []
	dst = []
	for bond in mol.GetBonds():
		u = bond.GetBeginAtomIdx()
		v = bond.GetEndAtomIdx()
		src.extend([u, v])
		dst.extend([v, u])  # undirectional 
		
	g = dgl.graph((src, dst), num_nodes=mol.GetNumAtoms())
	g.smiles = smi

	# Node featurize 
	atom_feats = []
	for i in range(mol.GetNumAtoms()):
		atom = mol.GetAtomWithIdx(i)
		atom_feats.append(atom_featurizer(atom))
	g.ndata['f'] = torch.tensor(atom_feats, dtype=torch.float)

	# Edge featurize
	bond_feats = []
	for u, v in zip(src, dst):
		bond = mol.GetBondBetweenAtoms(u, v)
		bond_feats.append(bond_featurizer(bond))  
	g.edata['f'] = torch.tensor(bond_feats, dtype=torch.float)
	
	if atom_vocab:            
		atom_idx = []
		for i in range(mol.GetNumAtoms()):
			atom = mol.GetAtomWithIdx(i)
			atom_idx.append(atom_vocab.stoi.get(atom_to_vocab(mol, atom), atom_vocab.other_index))
		g.ndata['atom_idx'] = torch.tensor(atom_idx, dtype=torch.float)    
	return g

# smi & tree & vocab -> Motif tree dgl.graph
def Tree2MotifGraph(smi, tree, vocab_dict):
	# Define connection between edges 
	src, dst = [], []
	fe_feats = []
	for u, v in tree.edges:
		src.extend([u, v])
		dst.extend([v, u])
		fe_feat = motif_bond_featurizer(tree.get_edge(u, v).edges[0][2:])
		fe_feats.append(fe_feat)
		fe_feats.append(fe_feat)
		
	# Exception: only self-loop exists
	if (len(fe_feats) == 0) | (len(tree.nodes)):
		src.extend([0., 0.])
		dst.extend([0., 0.])
		fe_feats.append([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.])
		fe_feats.append([0., 0., 0., 0., 1., 1., 0., 0., 0., 0.])
	
	if len(tree.nodes) == 0.:
		mol = Chem.MolFromSmiles(smi)
		smi = Chem.MolToSmiles(mol)
		g = dgl.graph((src, dst), num_nodes=1)
		g.smiles = smi
		g.edata['f'] = torch.tensor(fe_feats)
		motif_feat = motif_featurizer(smi, vocab_dict)
		g.ndata['f'] = torch.tensor([motif_feat])
		g.ndata['f'] = torch.tensor([motif_feat[1:]])

		g.ndata['motif_vocab_idx'] = torch.tensor([motif_feat[0]])
		
		g.gdata= {}
		g.gdata['atom_map'], g.gdata['motif_map'] = torch.tensor(list(range(mol.GetNumAtoms()))), torch.tensor([0]*mol.GetNumAtoms())
		return g

	g = dgl.graph((src, dst), num_nodes=len(tree.nodes))
	g.smiles = smi
	g.edata['f'] = torch.tensor(fe_feats)
		
	atom_map, motif_map, motif_idx = [], [], []
	motif_idx, motif_feats = [], [] 
	for node_i in tree.nodes:
		node = tree.get_node(node_i)
		atom_map.extend(node.atom_mapping.keys())
		motif_map.extend([node_i]*len(node.atom_mapping.keys()))
		motif_feat = motif_featurizer(node.smiles, vocab_dict)
		motif_feats.append(motif_feat[1:])
		motif_idx.append(motif_feat[0])
		
	g.ndata['f'] = torch.tensor(motif_feats)
	g.ndata['motif_vocab_idx'] = torch.tensor(motif_idx) 
	
	g.gdata = {}
	g.gdata['atom_map'], g.gdata['motif_map'] = torch.tensor(atom_map), torch.tensor(motif_map)
	return g


##### Prepare dataset (Embedding 추출용 간소화 ver) #####
class MolGraphDataset(Dataset):
	def __init__(self, smiles_list, label_list, vocab_path):
		self.smiles_list = smiles_list
		self.label_list = label_list 
		self.tree_generator = SubgraphGenerator(vocab_path)
		
		self.vocab_dict = {smiles:i for i,smiles in enumerate(self.tree_generator.vocab_dict.keys())}
		self.vocab_size = len(self.vocab_dict)
		
		self.process()
		
	def process(self):
		self.atom_graphs, self.motif_graphs = [], []
		self.labels = []
		
		for i in range(len(self.smiles_list)):
			smi = self.smiles_list[i]
			mol = Chem.MolFromSmiles(smi)
			if mol != None: 
				# atom-level graph
				atom_g = Smi2AtomGraph(smi)

				if atom_g.num_nodes() > 1:
					tree = self.tree_generator.generate(mol)
					motif_g = Tree2MotifGraph(smi, tree, self.vocab_dict)

					self.atom_graphs.append(atom_g)
					self.motif_graphs.append(motif_g)
					self.labels.append(self.label_list[i])
				else:
					pass
			else:
				pass
		self.labels = torch.tensor(self.labels) 
		
	def __len__(self):
		return len(self.atom_graphs)
	
	def __getitem__(self, idx):
		if hasattr(idx, "__iter__"):
			return [(self.atom_graphs[i], self.motif_graphs[i]) for i in idx]
		return (self.atom_graphs[idx], self.motif_graphs[idx])

def init_decomp(mol, vocab):
	n_atoms = mol.GetNumAtoms()
	if n_atoms == 1:
		return [[0]], [mol_to_smi(get_clique_mol(mol, [0]))]
		
	# For ring
	raw_rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
	rings = raw_rings.copy()
	
	flag = True
	while flag:
		flag = False
		for i in range(len(rings)):
			if len(rings[i]) == 0:
				continue
			for j in range(i + 1, len(rings)):
				shared_atoms = rings[i] & rings[j]
				if len(shared_atoms) > 2:
					merged_candidate = list(rings[i] | rings[j])
					rings[i].update(rings[j])
					rings[j] = set()
					flag = True
	rings = [r for r in rings if len(r) > 0]
	ring_smis = [StandardizeSmiles(mol_to_smi(get_clique_mol(mol, ring))) for ring in rings]

	if ~all([sub_smi in vocab for sub_smi in ring_smis]):
		rings = raw_rings
		
	# For non-ring functional group
	PATT = {
		'HETEROATOM': '[!#6]',
		'DOUBLE_TRIPLE_BOND': '*=,#*',
		'ACETAL': '[CX4]([O,N,S])[O,N,S]'
	}
	PATT = {k: Chem.MolFromSmarts(v) for k, v in PATT.items()}
	
	fgs = []
	marks = set()
	for patt in PATT.values():
		for sub in mol.GetSubstructMatches(patt):
			marks.update(sub)
	atom2fg = [[] for _ in range(n_atoms)]
	for atom in marks:
		fgs.append({atom})
		atom2fg[atom] = [len(fgs) - 1]
	for bond in mol.GetBonds():
		if bond.IsInRing():
			continue
		a1, a2 = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
		if a1 in marks and a2 in marks:
			assert a1 != a2
			assert len(atom2fg[a1]) == 1 and len(atom2fg[a2]) == 1
			fgs[atom2fg[a1][0]].update(fgs[atom2fg[a2][0]])
			fgs[atom2fg[a2][0]] = set()
			atom2fg[a2] = atom2fg[a1]
		elif a1 in marks:
			fgs[atom2fg[a1][0]].add(a2)
			atom2fg[a2].extend(atom2fg[a1])
		elif a2 in marks:
			fgs[atom2fg[a2][0]].add(a1)
			atom2fg[a1].extend(atom2fg[a2])
		else:
			fgs.append({a1, a2})
			atom2fg[a1].append(len(fgs) - 1)
			atom2fg[a2].append(len(fgs) - 1)
	tmp = []
	for fg in fgs:
		if len(fg) == 0:
			continue
		if len(fg) == 1 and mol.GetAtomWithIdx(list(fg)[0]).IsInRing():
			continue
		tmp.append(fg)
	fgs = tmp
	fgs.extend(rings)

	fgs = [list(fg) for fg in fgs]
	fg_smis = [StandardizeSmiles(mol_to_smi(get_clique_mol(mol, fg))) for fg in fgs]
	return fgs, fg_smis


##### Subgraph Generator #####
class SubgraphGenerator:
	def __init__(self, vocab_path):
		self.kekulize = False
		self.idx2subgraph, self.subgraph2idx, self.vocab_dict = [], {}, {}
		with open(vocab_path, 'r') as fin:
			lines = fin.read().strip().split('\n')
			for line in lines:
				smi, atom_num, freq = line.strip().split('\t')
				self.vocab_dict[smi] = int(atom_num)
				self.subgraph2idx[smi] = len(self.idx2subgraph)
				self.idx2subgraph.append(smi)
			
	def generate(self, mol):
		smiles = mol_to_smi(mol)
		
		subgraphs, subgraph_smis = init_decomp(mol, self.vocab_dict)
		res = [(sub_smi, sub_g) for sub_g, sub_smi in zip(subgraphs, subgraph_smis)]

		# construct reversed index
		aid2pid = {}
		for pid, subgraph in enumerate(res):
			_, aids = subgraph
			for aid in aids:
				aid2pid[aid] = pid
		# construct adjacent matrix
		ad_mat = [[0 for _ in res] for _ in res]
		
		for aid in range(mol.GetNumAtoms()):
			atom = mol.GetAtomWithIdx(aid)
			for nei in atom.GetNeighbors():
				nei_id = nei.GetIdx()
				i, j = aid2pid[aid], aid2pid[nei_id]
				if i != j:
					ad_mat[i][j] = ad_mat[j][i] = 1
		return Molecule(smiles, res, self.kekulize)

	def idx_to_subgraph(self, idx):
		return self.idx2subgraph[idx]
	
	def subgraph_to_idx(self, subgraph):
		return self.subgraph2idx[subgraph]

	def num_subgraph_type(self):
		return len(self.idx2subgraph)

	def __call__(self, mol):
		return self.generate(mol)
	
	def __len__(self):
		return len(self.idx2subgraph)


##### Heterogeneous Augmented graph #####
def extract_descriptor(smi):
	generator = rdNormalizedDescriptors.RDKit2DNormalized()
	return np.nan_to_num(np.array(generator.process(smi)[1:], dtype=np.float32), nan=0.0)

class MolAugGraphDataset(Dataset):
	def __init__(self, smiles_list, label_list, vocab_path=None):
		self.dataset = MolGraphDataset(smiles_list, label_list, vocab_path)
		self.process()
		
	def process(self):
		self.aug_g_list, self.atom_g_list, self.motif_g_list = [], [], []
		self.label_list = self.dataset.labels 
	
		for i in range(len(self.dataset)):
			atom_g, motif_g = self.dataset[i][0], self.dataset[i][1]
			atom_map = motif_g.gdata['atom_map'].long()
			motif_map = motif_g.gdata['motif_map'].long()
			
			num_node_dict = {'a': atom_g.num_nodes(), 'p': motif_g.num_nodes()}
			edge_dict = {('a', 'b', 'a') : atom_g.edges(),
						('p', 'r', 'p') : motif_g.edges(),
						('a', 'j', 'p') : (atom_map, motif_map),
						('p', 'j', 'a') : (motif_map, atom_map)}
			
			hetero_g = dgl.heterograph(edge_dict,
									num_nodes_dict=num_node_dict)            
			
			hetero_g.edges[('a', 'b', 'a')].data['f'] = atom_g.edata['f']
			hetero_g.edges[('p', 'r', 'p')].data['f'] = motif_g.edata['f']

			self.aug_g_list.append(hetero_g)
			self.atom_g_list.append(atom_g)
			self.motif_g_list.append(motif_g)
				
	def __len__(self):
		return len(self.aug_g_list)
	
	def __getitem__(self, idx):
		if hasattr(idx, "__iter__"):
			return [((self.aug_g_list[idx], self.atom_g_list[idx], self.motif_g_list[idx]), self.label_list[idx]) for i in idx]
		return (self.aug_g_list[idx], self.atom_g_list[idx], self.motif_g_list[idx]), self.label_list[idx]

	
# Additional imports for preprocessing functions
import re
import json
import csv
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from rdkit.Chem import SaltRemover, MolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from scipy.stats import normaltest

# Unit conversion imports
try:
	import pint
	from pint import UnitRegistry
	PINT_AVAILABLE = True
except ImportError:
	PINT_AVAILABLE = False

def data_loader(raw_datas, configs):
	task, raw_df = list(raw_datas.keys())[0], list(raw_datas.values())[0]

	batch_size = configs['loader']['batch_size']
	test_size = configs['loader']['test_size']

	current_dir = os.path.dirname(os.path.abspath(__file__))
	vocab_path = os.path.join(current_dir, configs['vocab'])
	#### Preprocessing step  #####  
	preprocessed_df = preprocess_dataframe(
		df=raw_df,
		task_type='regression',  # 일단은 regression으로 진행 -> 이후 데이터 포맷에 threshold column 확인되면 태스크별 정의 예정 
		task=task,
		smiles_column='smiles_structure_parent',
		activity_column='measurement_value',
		convert_units=True,
		correct_pH=True
	)
	
	# GIST format으로 변환
	gist_df = preprocess_to_gist(
		input_data=preprocessed_df,
		skip_preprocessing=True, # preprocess_dataframe 이후에 사용 시. 단독 사용 시 False
		endpoint_mapper='manual'
	)
	####
	X, y = gist_df['smiles'], gist_df[DEFAULT_ENDPOINT_SYNONYMS[task]]

	# X, y 수정 
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
	train_smi_list, test_smi_list, train_label_list, test_label_list = [list(data) for data in [X_train, X_test, y_train, y_test]]
	
	train_dataset = MolAugGraphDataset(train_smi_list, train_label_list, vocab_path=vocab_path)
	train_loader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True)

	test_dataset = MolAugGraphDataset(test_smi_list, test_label_list, vocab_path=vocab_path)
	test_loader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

	data = {"train":train_loader,"test":test_loader}
	return data


##### Preprocessing functions from hits-preprocess.py (logging removed) #####

def recognize_task_type(each_df, activity_col='measurement_value'):
	"""Analyze the activity value column to determine task type (classification or regression)."""
	if activity_col not in each_df.columns:
		return "classification"
	
	if 'Not specified' in each_df[activity_col].values or \
	each_df[activity_col].astype(str).str.contains('^[a-zA-Z]').any():
		return "classification"
	
	try:
		numeric_values = pd.to_numeric(each_df[activity_col].astype(str).str.replace(r'^[<>]=?\s*', '', regex=True), errors='coerce')
		if numeric_values.isna().sum() > 0:
			return "classification"
		unique_values = numeric_values.dropna().unique()
		if set(unique_values) <= {0, 1}:
			return "classification"
	except:
		return "classification"
	
	return "regression"


class TrainingQuorumError(Exception):
	def __init__(self, task_type, message=None):
		self.task_type = task_type
		if message is None:
			if task_type.lower() == 'classification':
				message = "Training quorum is 25 actives and 25 inactives per task."
			elif task_type.lower() == 'regression':
				message = "Training quorum is 50 data points out of which 25 uncensored per task."
			else:
				message = "Training quorum requirement not met. At least 50 data points needed"
		super().__init__(message)


class UnitConverter:
	"""Class for converting units to SI units using pint package"""
	def __init__(self):
		if not PINT_AVAILABLE:
			raise ImportError("pint package is required for unit conversion. Please install it with: pip install pint")
		self.ureg = UnitRegistry()
		self.unit_mappings = {
			'ug/ml': 'ug/mL', 'ug/l': 'ug/L', 'mg/ml': 'mg/mL', 'mg/l': 'mg/L',
			'ng/ml': 'ng/mL', 'ng/l': 'ng/L', 'pg/ml': 'pg/mL', 'pg/l': 'pg/L',
			'um': 'uM', 'umol/l': 'uM', 'umol/liter': 'uM', 'mmol/l': 'mM',
			'mmol/liter': 'mM', 'nmol/l': 'nM', 'nmol/liter': 'nM', 'pmol/l': 'pM',
			'pmol/liter': 'pM', 'mol/l': 'M', 'mol/liter': 'M',
			'hr': 'hour', 'hrs': 'hour', 'hours': 'hour', 'min': 'minute',
			'mins': 'minute', 'minutes': 'minute', 'sec': 'second', 'secs': 'second',
			'seconds': 'second', 'day': 'day', 'days': 'day',
			'ml': 'mL', 'l': 'L', 'ul': 'uL', 'nl': 'nL', 'pl': 'pL',
			'ug': 'ug', 'mg': 'mg', 'ng': 'ng', 'pg': 'pg', 'g': 'g', 'kg': 'kg',
			'%': 'percent', 'percent': 'percent',
			'c': 'degC', 'celsius': 'degC', 'f': 'degF', 'fahrenheit': 'degF',
			'k': 'kelvin', 'kelvin': 'kelvin',
			'atm': 'atm', 'bar': 'bar', 'pa': 'Pa', 'pascal': 'Pa', 'psi': 'psi',
			'j': 'J', 'joule': 'J', 'kj': 'kJ', 'kcal': 'kcal', 'cal': 'cal',
			'm': 'm', 'cm': 'cm', 'mm': 'mm', 'um': 'um', 'nm': 'nm', 'pm': 'pm',
			'angstrom': 'angstrom', 'a': 'angstrom',
		}
		self.compound_unit_mappings = {
			'ng*h/ml': 'ng*hour/mL', 'ng*hr/ml': 'ng*hour/mL', 'ng*hour/ml': 'ng*hour/mL',
			'ug*h/ml': 'ug*hour/mL', 'mg*h/ml': 'mg*hour/mL',
			'l/hr/kg': 'L/hour/kg', 'l/h/kg': 'L/hour/kg', 'ml/min/kg': 'mL/minute/kg',
			'ul/min/mg': 'uL/minute/mg', 'ml/min/mg': 'mL/minute/mg',
			'min-1': '1/minute', 'hr-1': '1/hour', 'h-1': '1/hour', 's-1': '1/second',
		}
		self.keep_as_is_units = {'1e-6 cm/s', 'ratio', 'fold', 'percent', 'dimensionless', 'x'}

	def normalize_unit_string(self, unit_str):
		"""Normalize a unit string to a pint-parseable (or keep-as-is) form,
		preserving '.', '-', '*', '^', '/', '·' so composite/exponent units
		survive (e.g. '10-6 cm/s', 'ng·h/mL', 'min-1')."""
		if unit_str is None or (isinstance(unit_str, float) and pd.isna(unit_str)):
			return None
		u = str(unit_str).strip()
		if u in ('', '-', 'Not specified', 'nan', 'NaN', 'None'):
			return None
		# Molar concentrations are case-sensitive (uM/nM/mM/pM/M != length).
		molar = re.fullmatch(r'([munp]?)M', u.replace(' ', ''))
		if molar:
			return {'u': 'uM', 'm': 'mM', 'n': 'nM', 'p': 'pM', '': 'M'}[molar.group(1)]
		low = u.lower()
		low = low.replace('·', '*').replace('×', '*').replace('•', '*')
		low = re.sub(r'\s+', '', low)
		if re.fullmatch(r'(x)?10\^?-6cm/s(ec)?', low):
			return '1e-6 cm/s'
		if low in ('ratio', 'fold', 'x'):
			return low
		if low in ('%', 'percent'):
			return 'percent'
		m = re.fullmatch(r'([a-z]+)-1', low)
		if m and m.group(1) in ('min', 'hr', 'h', 's', 'sec'):
			base = {'min': 'minute', 'hr': 'hour', 'h': 'hour', 's': 'second', 'sec': 'second'}[m.group(1)]
			return f'1/{base}'
		if 'cell' in low:
			return low
		low = low.replace('^', '**')
		if low in self.compound_unit_mappings:
			return self.compound_unit_mappings[low]
		if low in self.unit_mappings:
			return self.unit_mappings[low]
		return low

	def convert_to_si(self, value, unit_str):
		"""Convert a value with unit to SI units"""
		if not PINT_AVAILABLE:
			return value, unit_str, unit_str
		try:
			normalized_unit = self.normalize_unit_string(unit_str)
			if not normalized_unit:
				return value, unit_str, unit_str
			if normalized_unit in self.keep_as_is_units or 'cell' in normalized_unit:
				return value, normalized_unit, unit_str
			try:
				quantity = value * self.ureg(normalized_unit)
			except Exception:
				logger.warning(
					f"Could not parse unit '{unit_str}' (normalized '{normalized_unit}'). "
					f"Keeping original value unconverted.")
				return value, unit_str, unit_str
			si_quantity = quantity.to_base_units()
			si_unit_str = str(si_quantity.units)
			return float(si_quantity.magnitude), si_unit_str, unit_str
		except Exception:
			return value, unit_str, unit_str
	
	def convert_column_to_si(self, df, value_col, unit_col, new_value_col=None, new_unit_col=None):
		"""Convert a column of values with units to SI units"""
		if not PINT_AVAILABLE:
			return df
		if new_value_col is None:
			new_value_col = f"{value_col}_si"
		if new_unit_col is None:
			new_unit_col = f"{unit_col}_si"
		# object dtype so per-row float/str assignment works under pandas 3.0.
		df[new_value_col] = df[value_col].astype(object)
		df[new_unit_col] = df[unit_col].astype(object)
		for idx, row in df.iterrows():
			try:
				value = row[value_col]
				unit = row[unit_col]
				if pd.isna(value) or pd.isna(unit) or unit == "Not specified":
					continue
				try:
					if isinstance(value, str):
						value = re.sub(r'^[<>]=?\s*', '', value)
						value = float(value)
				except:
					continue
				converted_value, si_unit, _ = self.convert_to_si(value, unit)
				df.at[idx, new_value_col] = converted_value
				df.at[idx, new_unit_col] = si_unit
			except Exception:
				continue
		return df


class pHCorrector:
	"""Class for pH correction of activity values using different methods"""
	def __init__(self, method='all', target_pH=7.4):
		self.method = method
		self.target_pH = target_pH
		try:
			from rdkit.Chem import Descriptors
			self.rdkit_available = True
			self.Descriptors = Descriptors
		except ImportError:
			self.rdkit_available = False
	
	def _predict_pKa(self, smiles):
		"""Predict pKa values for a molecule"""
		if not self.rdkit_available:
			return None, None
		try:
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				return None, None
			acidic_groups = 0
			basic_groups = 0
			for atom in mol.GetAtoms():
				if atom.GetSymbol() == 'O' and atom.GetDegree() == 1:
					for bond in atom.GetBonds():
						if bond.GetOtherAtom(atom).GetSymbol() == 'C':
							acidic_groups += 1
				if atom.GetSymbol() == 'N' and atom.GetDegree() <= 3:
					basic_groups += 1
			pKa_acidic = 4.5 if acidic_groups > 0 else None
			pKa_basic = 9.5 if basic_groups > 0 else None
			return pKa_acidic, pKa_basic
		except Exception:
			return None, None
	
	def _henderson_hasselbalch_correction(self, activity_value, pH_measured, pKa_acidic=None, pKa_basic=None):
		"""Correct activity using Henderson-Hasselbalch equation"""
		if pH_measured == self.target_pH:
			return activity_value
		is_acidic = False
		pKa = None
		if pKa_acidic is not None and pKa_basic is not None:
			if abs(pKa_acidic - 7.4) < abs(pKa_basic - 7.4):
				pKa = pKa_acidic
				is_acidic = True
			else:
				pKa = pKa_basic
				is_acidic = False
		elif pKa_acidic is not None:
			pKa = pKa_acidic
			is_acidic = True
		elif pKa_basic is not None:
			pKa = pKa_basic
			is_acidic = False
		else:
			return activity_value
		try:
			if is_acidic:
				f_ionized_measured = 1 / (1 + 10**(pKa - pH_measured))
				f_ionized_target = 1 / (1 + 10**(pKa - self.target_pH))
			else:
				f_ionized_measured = 1 / (1 + 10**(pH_measured - pKa))
				f_ionized_target = 1 / (1 + 10**(self.target_pH - pKa))
			f_unionized_measured = 1 - f_ionized_measured
			f_unionized_target = 1 - f_ionized_target
			if f_unionized_measured == 0:
				return activity_value
			correction_factor = f_unionized_target / f_unionized_measured
			return activity_value * correction_factor
		except Exception:
			return activity_value
	
	def _empirical_correction(self, activity_value, pH_measured):
		"""Correct activity using empirical pH-activity relationship"""
		if pH_measured == self.target_pH:
			return activity_value
		try:
			pH_diff = self.target_pH - pH_measured
			if abs(pH_diff) <= 1.0:
				correction_factor = 1.0 + (pH_diff * 0.05)
			elif abs(pH_diff) <= 2.0:
				correction_factor = 1.0 + (pH_diff * 0.1)
			else:
				correction_factor = 1.0 + (pH_diff * 0.15)
			correction_factor = max(0.1, min(10.0, correction_factor))
			return activity_value * correction_factor
		except Exception:
			return activity_value
	
	def _molecular_properties_correction(self, activity_value, pH_measured, smiles):
		"""Correct activity using molecular properties"""
		if not self.rdkit_available or pH_measured == self.target_pH:
			return activity_value
		try:
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				return activity_value
			mol_weight = self.Descriptors.ExactMolWt(mol)
			logp = self.Descriptors.MolLogP(mol)
			tpsa = self.Descriptors.TPSA(mol)
			hbd = self.Descriptors.NumHDonors(mol)
			hba = self.Descriptors.NumHAcceptors(mol)
			logp_factor = max(0.5, min(2.0, 1.0 - (logp / 10.0)))
			tpsa_factor = max(0.5, min(2.0, 1.0 + (tpsa / 200.0)))
			hbd_hba_factor = max(0.5, min(2.0, 1.0 + ((hbd + hba) / 20.0)))
			pH_sensitivity = (logp_factor + tpsa_factor + hbd_hba_factor) / 3.0
			pH_diff = self.target_pH - pH_measured
			correction_factor = 1.0 + (pH_diff * 0.1 * pH_sensitivity)
			correction_factor = max(0.1, min(10.0, correction_factor))
			return activity_value * correction_factor
		except Exception:
			return activity_value
	
	def correct_activity(self, activity_value, pH_measured, smiles=None):
		"""Correct activity value using the specified method"""
		if pd.isna(activity_value) or pd.isna(pH_measured):
			return {'original': activity_value, 'henderson_hasselbalch': activity_value,
					'empirical': activity_value, 'molecular_properties': activity_value}
		try:
			activity_value = float(activity_value)
			pH_measured = float(pH_measured)
		except (ValueError, TypeError):
			return {'original': activity_value, 'henderson_hasselbalch': activity_value,
					'empirical': activity_value, 'molecular_properties': activity_value}
		result = {'original': activity_value}
		if self.method in ['all', 'henderson_hasselbalch']:
			pKa_acidic, pKa_basic = self._predict_pKa(smiles) if smiles else (None, None)
			result['henderson_hasselbalch'] = self._henderson_hasselbalch_correction(
				activity_value, pH_measured, pKa_acidic, pKa_basic)
		if self.method in ['all', 'empirical']:
			result['empirical'] = self._empirical_correction(activity_value, pH_measured)
		if self.method in ['all', 'molecular_properties']:
			result['molecular_properties'] = self._molecular_properties_correction(
				activity_value, pH_measured, smiles)
		return result
	
	def correct_column(self, df, activity_col, pH_col, smiles_col=None):
		"""Correct activity values in a DataFrame column"""
		if activity_col not in df.columns or pH_col not in df.columns:
			return df
		if self.method == 'all':
			new_cols = [f"{activity_col}_pH_corrected_hh", f"{activity_col}_pH_corrected_emp",
					f"{activity_col}_pH_corrected_mp"]
		else:
			method_suffix = {'henderson_hasselbalch': 'hh', 'empirical': 'emp', 'molecular_properties': 'mp'}
			new_cols = [f"{activity_col}_pH_corrected_{method_suffix[self.method]}"]
		# object dtype so per-row float assignment works under pandas 3.0.
		for col in new_cols:
			df[col] = df[activity_col].astype(object)
		for idx, row in df.iterrows():
			try:
				activity = row[activity_col]
				pH = row[pH_col]
				smiles = row[smiles_col] if smiles_col else None
				if pd.isna(activity) or pd.isna(pH):
					continue
				corrected = self.correct_activity(activity, pH, smiles)
				if self.method == 'all':
					df.at[idx, f"{activity_col}_pH_corrected_hh"] = corrected['henderson_hasselbalch']
					df.at[idx, f"{activity_col}_pH_corrected_emp"] = corrected['empirical']
					df.at[idx, f"{activity_col}_pH_corrected_mp"] = corrected['molecular_properties']
				else:
					df.at[idx, new_cols[0]] = corrected[self.method]
			except Exception:
				continue
		return df


def process_smiles_batch_global(smiles_batch):
	"""Global function for processing SMILES batches in parallel"""
	from rdkit.Chem import MolStandardize
	from rdkit.Chem.Scaffolds import MurckoScaffold
	uncharger = MolStandardize.rdMolStandardize.Uncharger()
	enumerator = MolStandardize.rdMolStandardize.TautomerEnumerator()
	results = []
	for smiles in smiles_batch:
		try:
			mol = Chem.MolFromSmiles(smiles)
			if mol is None:
				raise ValueError(f"SMILES string is not valid: {smiles}")
			for atom in mol.GetAtoms():
				atom.SetIsotope(0)
			mol = uncharger.uncharge(mol)
			Chem.SetAromaticity(mol)
			mol = Chem.RemoveHs(mol)
			Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
			mol = enumerator.Canonicalize(mol)
			try:
				Chem.SanitizeMol(mol)
			except Chem.rdChem.KekulizeException:
				raise ValueError(f"Valence error detected in molecule: {smiles}")
			standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
			scaffold = MurckoScaffold.GetScaffoldForMol(mol)
			scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
			results.append((standardized_smiles, scaffold_smiles))
		except Exception:
			results.append((None, None))
	return results


def _parse_bracket_numbers(cell):
	"""Parse a Permeability cell into a list of floats/None.

	Accepts bracketed lists like '[6,2]', '[6,]', '[,2]', plain scalars '6.2',
	or empty/placeholder values. An empty slot becomes None.
	"""
	if cell is None:
		return []
	s = str(cell).strip()
	if s in ("", "-", "nan", "NaN", "None", "Not specified"):
		return []
	s = s.strip("[]")
	if s == "":
		return []
	out = []
	for part in s.split(","):
		part = part.strip()
		if part == "":
			out.append(None)
			continue
		part = re.sub(r'^[<>]=?\s*', '', part)
		try:
			out.append(float(part))
		except ValueError:
			out.append(None)
	return out


def parse_permeability_pair(atob_cell, btoa_cell):
	"""Return (AtoB, BtoA) floats from the two Permeability value columns.

	Handles the v4.6 '[Value, Value]=[AB, BA]' convention whether the pair is
	split across the AtoB/BtoA columns or carried as a full list in either one.
	"""
	ab = _parse_bracket_numbers(atob_cell)
	ba = _parse_bracket_numbers(btoa_cell)
	atob = ab[0] if len(ab) >= 1 else None
	btoa = ba[1] if len(ba) >= 2 else (ba[0] if len(ba) == 1 else None)
	if atob is None and len(ba) >= 1:
		atob = ba[0]
	if btoa is None and len(ab) >= 2:
		btoa = ab[1]
	return atob, btoa


def _expand_permeability_columns(df):
	"""Split Permeability's ``measurement_value(atob)``/``(btoa)`` list columns
	into a scalar ``measurement_value`` (AtoB) plus ``measurement_efflux_ratio``
	(BtoA/AtoB). Operates on lowercase-normalized column names.
	"""
	ab_col, ba_col = 'measurement_value(atob)', 'measurement_value(btoa)'
	if ab_col not in df.columns and ba_col not in df.columns:
		return df
	atob_vals, btoa_vals, ratio_vals = [], [], []
	for _, row in df.iterrows():
		atob, btoa = parse_permeability_pair(row.get(ab_col), row.get(ba_col))
		atob_vals.append(atob)
		btoa_vals.append(btoa)
		if atob not in (None, 0) and btoa is not None:
			ratio_vals.append(btoa / atob)
		else:
			ratio_vals.append(None)
	if 'measurement_value' not in df.columns:
		df['measurement_value'] = atob_vals
	else:
		existing = df['measurement_value']
		df['measurement_value'] = [
			a if (pd.isna(e) or str(e) in ("", "Not specified")) else e
			for e, a in zip(existing, atob_vals)
		]
	df['measurement_value_btoa'] = btoa_vals
	df['measurement_efflux_ratio'] = ratio_vals
	return df


class DataInspector:
	"""Data inspector for loading and validating data (supports CSV path or DataFrame)"""
	def __init__(self, input_path=None, df=None, smiles_column='smiles_structure_parent',
				activity_column='measurement_value',
				condition_columns=['test', 'test_type', 'test_subject', 'measurement_type',
								'measurement_conc', 'measurement_temp', 'measurement_class',
								'measurement_route', 'measurement_sex', 'measurement_formulation']):
		self.smiles_col = smiles_column
		self.activity_col = activity_column
		self.condition_columns = condition_columns
		self.is_human_pk = False
		if df is not None:
			self.df = self._process_dataframe(df)
		elif input_path is not None:
			self.df = self.load_data(input_path)
		else:
			raise ValueError("Either input_path or df must be provided")
	
	def normalize_column_names(self, df):
		"""Normalize column names to canonical lowercase K-MELLODDY names.

		The real value column is preserved; the legacy 'Unnamed: 10' remap is
		applied only as a fallback when no measurement_value survived (old
		merged-cell layout).
		"""
		df = self._normalize_column_names_before_concat(df)
		# Legacy fallback for the old merged-cell layout.
		if 'measurement_value' not in df.columns and 'Unnamed: 10' in df.columns:
			df = df.rename(columns={'Unnamed: 10': 'measurement_value'})
		df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
		return df
	
	def _process_dataframe(self, df):
		"""Process DataFrame input"""
		df = df.copy()
		if self.smiles_col not in df.columns:
			raise ValueError(f"SMILES column '{self.smiles_col}' not found in data.")
		if self.activity_col in df.columns:
			try:
				df[self.activity_col] = df[self.activity_col].astype(str)
			except Exception:
				pass
		for col in df.columns:
			if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col].dtype):
				df[col] = df[col].astype(str).str.replace('μ', 'u')
				df[col] = df[col].apply(lambda x: self._replace_special_chars(x) if isinstance(x, str) else x)
		df = self.normalize_column_names(df)
		df = _expand_permeability_columns(df)
		for col in self.condition_columns:
			if col not in df.columns:
				df[col] = "Not specified"
		return df
	
	# Canonical (lowercase) column names keyed by their normalized form
	# (lowercased, trailing '*' and whitespace stripped). Covers v4.6 spelling,
	# the legacy 'Measurment_*' typo, 'Test_Species' -> 'test_subject', and VIVO's
	# lowercase 'Measurement_value'. Mirrors hits-preprocess.py CANONICAL_COLUMNS.
	CANONICAL_COLUMNS = {
		'chemical id': 'chemical id',
		'chemical name': 'chemical name',
		'smiles_structure_parent': 'smiles_structure_parent',
		'smiles_salt': 'smiles_salt',
		'test': 'test',
		'test_type': 'test_type',
		'test_subject': 'test_subject',
		'test_species': 'test_subject',          # legacy rename
		'test_dose': 'test_dose',
		'measurement_type': 'measurement_type',
		'measurment_type': 'measurement_type',   # legacy typo
		'measurement_relation': 'measurement_relation',
		'measurment_relation': 'measurement_relation',
		'measurement_value': 'measurement_value',
		'measurment_value': 'measurement_value',
		'measurement_value(atob)': 'measurement_value(atob)',
		'measurement_value(btoa)': 'measurement_value(btoa)',
		'measurement_unit': 'measurement_unit',
		'measurment_unit': 'measurement_unit',
		'measurement_conc': 'measurement_conc',
		'measurement_temp': 'measurement_temp',
		'measurement_condition': 'measurement_condition',
		'measurment_condition': 'measurement_condition',
		'measurement_class': 'measurement_class',
		'measurement_route': 'measurement_route',
		'measurement_sex': 'measurement_sex',
		'measurement_formulation': 'measurement_formulation',
		'comment': 'comment',
	}

	def _normalize_column_names_before_concat(self, df):
		"""Normalize column names to canonical lowercase names before concat."""
		column_mapping = {}
		for col in df.columns:
			key = str(col).strip().rstrip('*').strip().lower()
			canonical = self.CANONICAL_COLUMNS.get(key)
			if canonical and canonical != col:
				column_mapping[col] = canonical
		if column_mapping:
			df = df.rename(columns=column_mapping)
		df = df.loc[:, ~df.columns.duplicated()]
		return df
	
	def _replace_special_chars(self, text):
		"""Replace special characters that might cause issues"""
		replacements = {
			'μ': 'u', '°': 'deg', 'α': 'alpha', 'β': 'beta', 'γ': 'gamma', 'δ': 'delta',
			'±': '+/-', '≤': '<=', '≥': '>=', '×': 'x', '÷': '/',
		}
		for char, replacement in replacements.items():
			text = text.replace(char, replacement)
		return text
	
	def load_data(self, input_path):
		"""Load data from CSV or Excel file"""
		# Convert relative path to absolute path based on data.py location
		path = Path(input_path).expanduser()
		base_dir = Path(os.path.dirname(os.path.abspath(__file__)))
		if not path.is_absolute():
			path = base_dir / path
		# Resolve path to prevent path traversal attacks
		resolved_path = path.resolve()
		base_resolved = base_dir.resolve()
		# Ensure resolved path is within base directory
		try:
			resolved_path.relative_to(base_resolved)
		except ValueError:
			raise ValueError(f"Path traversal detected: {input_path} resolves outside allowed directory")
		input_path = str(resolved_path)
		if not os.path.exists(input_path):
			raise FileNotFoundError(f"Input file not found: {input_path}")
		file_extension = os.path.splitext(input_path)[1].lower()
		if file_extension == '.csv':
			df = pd.read_csv(input_path).fillna("Not specified")
			# Remove duplicate columns if any (prevent groupby errors)
			df = df.loc[:, ~df.columns.duplicated()]
		elif file_extension in ['.xlsx', '.xls']:
			# Use context manager to ensure Excel file is properly closed
			with pd.ExcelFile(input_path) as excel_file:
				sheets = excel_file.sheet_names
				# Human PK: wide-format, 3-row header, non-GIST schema.
				human_pk_sheets = [s for s in sheets
								if str(s).startswith('데이터') and 'human pk' in str(s).lower()]
				if human_pk_sheets:
					self.is_human_pk = True
					hpk_dfs = [pd.read_excel(excel_file, sheet_name=s, header=2).fillna("Not specified")
							for s in human_pk_sheets]
					df = pd.concat(hpk_dfs, ignore_index=True)
					df = df.loc[:, ~df.columns.duplicated()]
					if self.smiles_col not in df.columns:
						for col in df.columns:
							if 'smiles' in str(col).lower():
								df = df.rename(columns={col: self.smiles_col})
								break
					return df
				dfs = []
				if 'ADMET' in sheets:
					admet_df = pd.read_excel(excel_file, sheet_name='ADMET').fillna("Not specified")
					# Normalize column names before concatenation
					admet_df = self._normalize_column_names_before_concat(admet_df)
					dfs.append(admet_df)
				if 'PK' in sheets:
					pk_df = pd.read_excel(excel_file, sheet_name='PK').fillna("Not specified")
					# Normalize column names before concatenation
					pk_df = self._normalize_column_names_before_concat(pk_df)
					dfs.append(pk_df)
				if '데이터' in sheets:
					data_df = pd.read_excel(excel_file, sheet_name='데이터', header=1).fillna("Not specified")
					# Normalize column names before concatenation
					data_df = self._normalize_column_names_before_concat(data_df)
					dfs.append(data_df)
				if not dfs:
					df = pd.read_excel(excel_file, sheet_name=0).fillna("Not specified")
				else:
					df = pd.concat(dfs, ignore_index=True)
				# Remove duplicate columns if any (prevent groupby errors)
				df = df.loc[:, ~df.columns.duplicated()]
		else:
			raise ValueError(f"Unsupported file format: {file_extension}")
		
		# Normalize column names to lowercase after loading
		df = self._normalize_column_names_before_concat(df)
		
		if self.smiles_col not in df.columns:
			# Try to find SMILES column with case-insensitive match
			smiles_candidates = [col for col in df.columns if col.lower() == self.smiles_col.lower()]
			if smiles_candidates:
				self.smiles_col = smiles_candidates[0]
			else:
				raise ValueError(f"SMILES column '{self.smiles_col}' not found in data. Available columns: {list(df.columns)}")
		if self.activity_col in df.columns:
			try:
				df[self.activity_col] = df[self.activity_col].astype(str)
			except Exception:
				pass
		for col in df.columns:
			if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col].dtype):
				df[col] = df[col].astype(str).str.replace('μ', 'u')
				df[col] = df[col].apply(lambda x: self._replace_special_chars(x) if isinstance(x, str) else x)
		df = self.normalize_column_names(df)
		df = _expand_permeability_columns(df)
		for col in self.condition_columns:
			if col not in df.columns:
				df[col] = "Not specified"
		return df
	
	def satisfy_training_quorum(self, each_df, task_type):
		"""Check if data satisfies training quorum requirements"""
		if task_type.lower() == 'classification':
			counts = each_df[self.activity_col].value_counts()
			active_count = counts.get('active', 0)
			inactive_count = counts.get('inactive', 0)
			return active_count >= 25 and inactive_count >= 25
		else:
			total_count = len(each_df)
			rel_col = 'measurement_relation' if 'measurement_relation' in each_df.columns else (
				'Measurement_Relation' if 'Measurement_Relation' in each_df.columns else None)
			if rel_col is not None:
				# Censored = {>, >=, <, <=}; equality is stored as quoted '"="'.
				rel_norm = (each_df[rel_col].astype(str)
							.str.strip().str.strip('"').str.strip("'").str.strip())
				censored_mask = rel_norm.isin(['>', '>=', '<', '<='])
				uncensored_count = int((~censored_mask).sum())
			else:
				has_relation_mask = each_df[self.activity_col].astype(str).str.contains(r'^[<>]=?')
				uncensored_count = len(each_df[~has_relation_mask])
			return total_count >= 50 and uncensored_count >= 25


class Preprocessor:
	"""Main preprocessing pipeline for K-MELLODDY data"""
	def __init__(self, df, task_type, task, smiles_column='smiles_structure_parent',
				activity_column='measurement_value', remove_salt=True, keep_stereo=False,
				keep_duplicates=False, detect_outliers=False, scale_activity=True,
				convert_units=True, correct_pH=True, pH_method='all', target_pH=7.4, threshold=None):
		self.df = df.copy()
		self.activity_col = activity_column
		self.smiles_col = smiles_column
		self.check_task(task_type, task)
		self.threshold = threshold
		self.inspect_label()
		self.remove_salt = remove_salt
		self.keep_stereo = keep_stereo
		self.keep_duplicates = keep_duplicates
		self.scale_activity = scale_activity
		self.detect_outliers = detect_outliers
		self.convert_units = convert_units
		self.correct_pH = correct_pH
		self.pH_method = pH_method
		self.target_pH = target_pH
		self.active_is_high = True
		self.final_cols = []
		self.smiles_column = smiles_column
		self.remover = SaltRemover.SaltRemover()
		self.uncharger = MolStandardize.rdMolStandardize.Uncharger()
		self.enumerator = MolStandardize.rdMolStandardize.TautomerEnumerator()
		
		if self.convert_units and PINT_AVAILABLE:
			try:
				self.unit_converter = UnitConverter()
			except Exception:
				self.convert_units = False
		elif self.convert_units and not PINT_AVAILABLE:
			self.convert_units = False
		
		if self.correct_pH:
			try:
				self.pH_corrector = pHCorrector(method=self.pH_method, target_pH=self.target_pH)
			except Exception:
				self.correct_pH = False
		
		if 'test' in self.df.columns and 'Pharmacokinetics' in self.df['test'].values:
			if 'Test_Dose' in self.df.columns:
				self.final_cols.append('Test_Dose')
			else:
				self.df['Test_Dose'] = "Unknown"
				self.final_cols.append('Test_Dose')
		
		if 'Chemical ID' in self.df.columns:
			self.final_cols.append('Chemical ID')
	
	def check_task(self, task_type, task=None):
		"""Check if task type and task information are valid"""
		if task_type.lower() not in ['classification', 'regression']:
			raise ValueError(f"Task should be specified (Classification or Regression). Current task: {task_type}")
		self.task_type = task_type.lower()
		self.task = task
	
	def inspect_label(self):
		"""Inspect label type"""
		# Label type inspection is simplified - always set to regression
		# Original complex logic was commented out as it's not needed for current use case
		self.label_type = 'regression'
	
	def preprocess_compound(self, smiles):
		"""Preprocess a single SMILES string"""
		mol = Chem.MolFromSmiles(smiles)
		if mol is None:
			raise ValueError(f"SMILES string is not valid: {smiles}")
		if self.remove_salt:
			mol = self.remover.StripMol(mol)
		for atom in mol.GetAtoms():
			atom.SetIsotope(0)
		mol = self.uncharger.uncharge(mol)
		Chem.SetAromaticity(mol)
		mol = Chem.RemoveHs(mol)
		if not self.keep_stereo:
			Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
		mol = self.enumerator.Canonicalize(mol)
		try:
			Chem.SanitizeMol(mol)
		except Chem.rdChem.KekulizeException:
			raise ValueError(f"Valence error detected in molecule: {smiles}")
		standardized_smiles = Chem.MolToSmiles(mol, canonical=True)
		scaffold = MurckoScaffold.GetScaffoldForMol(mol)
		scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
		return standardized_smiles, scaffold_smiles
	
	def preprocess_compounds(self, smiles_list):
		"""Preprocess SMILES list"""
		standardized_smiles_list = []
		scaffolds = []
		if len(smiles_list) < 1000 or multiprocessing.cpu_count() <= 2:
			for smiles in smiles_list:
				try:
					standardized_smiles, scaffold_smiles = self.preprocess_compound(smiles)
					standardized_smiles_list.append(standardized_smiles)
					scaffolds.append(scaffold_smiles)
				except ValueError:
					standardized_smiles_list.append(None)
					scaffolds.append(None)
		else:
			smiles_list = list(smiles_list)
			batch_size = max(100, len(smiles_list) // (multiprocessing.cpu_count() * 2))
			smiles_batches = [smiles_list[i:i+batch_size] for i in range(0, len(smiles_list), batch_size)]
			with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
				results = pool.map(process_smiles_batch_global, smiles_batches)
			flat_results = [item for sublist in results for item in sublist]
			standardized_smiles_list = [item[0] for item in flat_results]
			scaffolds = [item[1] for item in flat_results]
		self.df["Standardized_SMILES"] = standardized_smiles_list
		self.df["Scaffold"] = scaffolds
		# Add columns to final_cols only if not already present (prevent duplicates)
		for col in ["Standardized_SMILES", "Scaffold"]:
			if col not in self.final_cols:
				self.final_cols.append(col)
	
	def detect_outliers_statistical(self):
		"""Detect outliers using IQR method"""
		q1 = self.df[self.activity_col].quantile(0.25)
		q3 = self.df[self.activity_col].quantile(0.75)
		iqr = q3 - q1
		lower_bound = q1 - 1.5 * iqr
		upper_bound = q3 + 1.5 * iqr
		return self.df[(self.df[self.activity_col] < lower_bound) | (self.df[self.activity_col] > upper_bound)]
	
	def detect_outliers_density_based(self):
		"""Detect outliers using LOF"""
		activity_data = self.df[[self.activity_col]].dropna()
		lof = LocalOutlierFactor(n_neighbors=20)
		labels = lof.fit_predict(activity_data)
		return self.df[labels == -1]
	
	def detect_outliers_classification_based(self):
		"""Detect outliers using OneClassSVM"""
		activity_data = self.df[[self.activity_col]].dropna()
		svm = OneClassSVM(kernel='rbf', gamma='auto')
		labels = svm.fit_predict(activity_data)
		return self.df[labels == -1]
	
	def detect_outlier_from_distribution(self):
		"""Detect outliers based on distribution"""
		activity_data = self.df[self.activity_col].dropna()
		stat, p_value = normaltest(activity_data)
		if p_value >= 0.05:
			self.outliers = self.detect_outliers_statistical()
		else:
			if self.task_type == "regression":
				self.outliers = self.detect_outliers_density_based()
			else:
				self.outliers = self.detect_outliers_classification_based()
	
	def scale_experiment_values(self, labels):
		"""Scale experimental values"""
		try:
			numeric_mask = pd.to_numeric(labels, errors='coerce').notna()
			if numeric_mask.sum() > 0:
				numeric_values = pd.to_numeric(labels[numeric_mask])
				if self.task_type == "regression" and self.scale_activity:
					scaler = StandardScaler()
					scaled_values = scaler.fit_transform(numeric_values.values.reshape(-1, 1))
					scaled_col = self.activity_col + "_scaled"
					self.df[scaled_col] = np.nan
					self.df.loc[numeric_mask, scaled_col] = scaled_values.flatten()
					# Add column to final_cols only if not already present (prevent duplicates)
					if scaled_col not in self.final_cols:
						self.final_cols.append(scaled_col)
		except Exception:
			if self.activity_col not in self.final_cols:
				self.final_cols.append(self.activity_col)
	
	def create_classification_label(self, labels):
		"""Convert numeric data to classification labels"""
		try:
			if set(labels.unique()) <= {'active', 'inactive', 'Active', 'Inactive'}:
				self.df["Classification_label"] = labels.str.lower()
				self.final_cols.append("Classification_label")
				return
			numeric_mask = pd.to_numeric(labels, errors='coerce').notna()
			if numeric_mask.sum() > 0 and self.threshold is not None:
				numeric_values = pd.to_numeric(labels[numeric_mask])
				self.df["Classification_label"] = "Not classified"
				if self.active_is_high:
					self.df.loc[numeric_mask, "Classification_label"] = np.where(
						numeric_values >= self.threshold, 'active', 'inactive')
				else:
					self.df.loc[numeric_mask, "Classification_label"] = np.where(
						numeric_values <= self.threshold, 'active', 'inactive')
				self.final_cols.append("Classification_label")
		except Exception:
			pass
	
	def preprocess_labels(self, labels):
		"""Preprocess labels"""
		try:
			if self.label_type == 'Categorical' and self.task_type == 'classification':
				if 'Classification_label' not in self.df.columns:
					self.create_classification_label(labels)
			if self.detect_outliers:
				self.detect_outlier_from_distribution()
			# pandas 3.0: np.issubdtype(StringDtype, np.number) raises; use pandas API.
			is_numeric = pd.api.types.is_numeric_dtype(labels)
			has_numeric_strings = False
			if not is_numeric:
				try:
					has_numeric_strings = labels.astype(str).str.match(r'^[<>]?=?\s*\d+\.?\d*$').any()
				except Exception:
					has_numeric_strings = False
			if is_numeric or has_numeric_strings:
				self.scale_experiment_values(labels)
			if self.label_type == 'Continuous' and self.task_type == "classification" and self.threshold is not None:
				self.create_classification_label(labels)
		except Exception:
			if self.activity_col not in self.final_cols:
				self.final_cols.append(self.activity_col)
	
	def _infer_ph_column(self):
		"""Infer a per-row pH value from v4.6 locations in priority order:
		dedicated pH columns, test_subject ('pH7.4'), measurement_condition (bare
		number or 'pH x'), then test_type. Returns a float Series (NaN if none)."""
		ph_pattern = re.compile(r"pH\s*([0-9]+(?:\.[0-9]+)?)", flags=re.IGNORECASE)

		def _ph_from_text(text):
			m = ph_pattern.search(str(text))
			if m:
				try:
					return float(m.group(1))
				except ValueError:
					return None
			return None

		def _bare_ph(text):
			m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*", str(text))
			if m:
				try:
					v = float(m.group(1))
					if 0.0 <= v <= 14.0:
						return v
				except ValueError:
					return None
			return None

		inferred = pd.Series(np.nan, index=self.df.index, dtype=float)
		for col in ['pH', 'pH_Value', 'Measurement_pH', 'Test_pH', 'ph']:
			if col in self.df.columns:
				inferred = inferred.combine_first(pd.to_numeric(self.df[col], errors='coerce'))
		if 'test_subject' in self.df.columns:
			inferred = inferred.combine_first(
				pd.to_numeric(self.df['test_subject'].apply(_ph_from_text), errors='coerce'))
		if 'measurement_condition' in self.df.columns:
			mc = self.df['measurement_condition'].apply(
				lambda x: _ph_from_text(x) if _ph_from_text(x) is not None else _bare_ph(x))
			inferred = inferred.combine_first(pd.to_numeric(mc, errors='coerce'))
		if 'test_type' in self.df.columns:
			inferred = inferred.combine_first(
				pd.to_numeric(self.df['test_type'].apply(_ph_from_text), errors='coerce'))
		return inferred

	def preprocess(self):
		"""Main preprocessing pipeline"""
		compounds = self.df[self.smiles_col]
		labels = self.df[self.activity_col]

		if self.convert_units and 'measurement_unit' in self.df.columns:
			try:
				self.df = self.unit_converter.convert_column_to_si(
					self.df, self.activity_col, 'measurement_unit',
					f"{self.activity_col}_si", 'measurement_unit_si')
				if f"{self.activity_col}_si" in self.df.columns:
					self.final_cols.append(f"{self.activity_col}_si")
				if 'measurement_unit_si' in self.df.columns:
					self.final_cols.append('measurement_unit_si')
			except Exception:
				pass
		
		if self.correct_pH:
			inferred_pH = self._infer_ph_column()
			pH_data_mask = inferred_pH.notna()
			if pH_data_mask.any():
				try:
					self.df['inferred_pH'] = inferred_pH
					pH_col = 'inferred_pH'
					if pH_col is not None:
						pH_df = self.df[pH_data_mask].copy()
						corrected_pH_df = self.pH_corrector.correct_column(
							pH_df, self.activity_col, pH_col, self.smiles_col)
						for idx in corrected_pH_df.index:
							for col in corrected_pH_df.columns:
								if col.startswith(f"{self.activity_col}_pH_corrected"):
									self.df.at[idx, col] = corrected_pH_df.at[idx, col]
						if self.pH_method == 'all':
							pH_corrected_cols = [
								f"{self.activity_col}_pH_corrected_hh",
								f"{self.activity_col}_pH_corrected_emp",
								f"{self.activity_col}_pH_corrected_mp"]
						else:
							method_suffix = {'henderson_hasselbalch': 'hh', 'empirical': 'emp', 'molecular_properties': 'mp'}
							pH_corrected_cols = [f"{self.activity_col}_pH_corrected_{method_suffix[self.pH_method]}"]
						for col in pH_corrected_cols:
							if col in self.df.columns and col not in self.final_cols:
								self.final_cols.append(col)
				except Exception:
					pass
		
		self.preprocess_compounds(compounds)
		self.preprocess_labels(labels)
		if self.detect_outliers:
			self.detect_outlier_from_distribution()
			self.df.drop(self.outliers.index, inplace=True)
		
		is_pk_data = False
		if 'test' in self.df.columns and 'Pharmacokinetics' in self.df['test'].values:
			is_pk_data = True
		
		if not self.keep_duplicates and not is_pk_data:
			before_count = len(self.df)
			# Snapshot with all enriched columns (SI, pH_corrected, standardized)
			# so recovery does not lose them if the strict dedup empties the set.
			df_before_dedup = self.df.copy()
			self.df.drop_duplicates(subset=["Standardized_SMILES"], keep=False, inplace=True, ignore_index=True)
			after_count = len(self.df)
			if after_count == 0 and before_count > 0:
				# Recover from the fully-processed frame, keeping the first row per
				# SMILES so all columns (SI/pH/standardized) survive.
				self.df = df_before_dedup.drop_duplicates(
					subset=["Standardized_SMILES"], keep='first', ignore_index=True)
			else:
				self.df.drop_duplicates(subset=["Standardized_SMILES"], keep='first', inplace=True, ignore_index=True)
		
		for col in [self.activity_col, 'test', 'test_type', 'test_subject', 'measurement_type']:
			if col in self.df.columns and col not in self.final_cols:
				self.final_cols.append(col)
		
		if len(self.df) == 0:
			raise ValueError("Preprocessing resulted in empty dataset!")
		
		# Remove duplicate columns from final_cols (preserve order)
		seen = set()
		unique_final_cols = []
		for col in self.final_cols:
			if col not in seen:
				seen.add(col)
				unique_final_cols.append(col)
		
		# Filter to only columns that exist in dataframe
		existing_cols = [col for col in unique_final_cols if col in self.df.columns]
		
		return self.df[existing_cols]


##### Manual endpoint converter (integrated from manual_converter.py) #####

# GIST endpoint list (hardcoded to avoid file dependency)
GIST_ENDPOINTS = [
	'Caco2', 'pgp_inhibitor', 'pgp_substrate', 'F20', 'F50', 'skin_permeability', 'HIA',
	'PAMPA_pH5(bc)', 'PAMPA_pH5(mc)', 'PAMPA_pH7.4(bc)', 'PAMPA_pH7.4(mc)', 'hydrationE',
	'Lipophilicity', 'Solubility', 'BBB_cns(reg)', 'BBB_logbb(cls)', 'PPBR', 'VDss', 'BCRP',
	'OATP1B1_Inhibitor', 'OATP1B3_Inhibitor', 'OATP2B1_Inhibitor', 'MATE1_Inhibitor',
	'fu_df', 'fu_deeppk', 't1/2(reg)', 't0.5(cls)', 'Efflux_ratio', 'CYP1A2_Inhibitor',
	'CYP2B6_Inhibitor', 'CYP2C9_Inhibitor', 'CYP2C19_Inhibitor', 'CYP2D6_Inhibitor',
	'CYP3A4_Inhibitor', 'CYP1A2_Substrate', 'CYP2B6_Substrate', 'CYP2C9_Substrate',
	'CYP2C19_Substrate', 'CYP2D6_Substrate', 'CYP3A4_Substrate', 'HLM', 'RLM',
	'HLC_Stability', 'Clearance_Hepatocyte_AZ', 'Clearance_Microsome_AZ', 'CLp(r)', 'CLp(bc)',
	'MRT', 'UGT_substrate', 'Clearance_total', 'OCT2_Inhibitor', 'NR-AhR', 'NR-AR', 'NR-AR-LBD',
	'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma', 'SR-ARE', 'SR-ATAD5', 'SR-HSE',
	'SR-MMP', 'SR-p53', 'GR', 'TR', 'Skin Reaction', 'Respir_tox', 'Liver_tox_hepato', 'DILI',
	'hERG', 'Micronucleus', 'Eye_irritation', 'Eye_corrosion', 'Carcinogen', 'AMES', 'ClinTox',
	'Nephro_tox', 'Mito_tox', 'Hemolytic', 'Reprotox', 'FDAMDD(reg)', 'FDAMDD(bc)', 'LD50',
	'Neuro_tox', 'Mouse_pTD50', 'Rat_pTD50'
]

DEFAULT_ENDPOINT_SYNONYMS = {
	# Permeability assays
	"caco2": "Caco2", "caco 2": "Caco2", "caco-2": "Caco2",
	"pampa": "PAMPA_pH7.4(bc)", "pampa ph74": "PAMPA_pH7.4(bc)", "pampa ph7.4": "PAMPA_pH7.4(bc)",
	"pampa ph75": "PAMPA_pH7.4(bc)", "pampa apical": "PAMPA_pH7.4(mc)", "pampa basolateral": "PAMPA_pH7.4(bc)",
	"pampa ph5": "PAMPA_pH5(bc)", "pampa ph 5": "PAMPA_pH5(bc)",
	"pampa ph5 apical": "PAMPA_pH5(mc)", "pampa ph5 basolateral": "PAMPA_pH5(bc)",
	"mdck": "Efflux_ratio", "skin permeability": "skin_permeability", "bia skin permeability": "skin_permeability",
	"hia": "HIA", "human intestinal absorption": "HIA",
	"bbb": "BBB_logbb(cls)", "blood brain barrier": "BBB_logbb(cls)", "brain penetration": "BBB_logbb(cls)",
	"bbb cns": "BBB_cns(reg)", "brain cns": "BBB_cns(reg)",
	"vdss": "VDss", "volume of distribution": "VDss", "vd": "VDss",
	# Transporters and pumps
	"pgp inhibitor": "pgp_inhibitor", "pgp substrate": "pgp_substrate", "p-gp substrate": "pgp_substrate",
	"p-gp inhibitor": "pgp_inhibitor", "p-gp": "pgp_inhibitor", "p_gp": "pgp_inhibitor",
	"bcrp inhibitor": "BCRP", "bcrp": "BCRP", "breast cancer resistance protein": "BCRP",
	"efflux ratio": "Efflux_ratio", "efflux_ratio": "Efflux_ratio",
	"oatp1b1": "OATP1B1_Inhibitor", "oatp1b3": "OATP1B3_Inhibitor", "oatp2b1": "OATP2B1_Inhibitor",
	"mate1": "MATE1_Inhibitor", "oct2": "OCT2_Inhibitor",
	# CYPs - only map CYPs that exist in GIST format
	"cyp1a2": "CYP1A2_Inhibitor", "cyp inhibition cyp1a2": "CYP1A2_Inhibitor", "cyp1a2 inhibition": "CYP1A2_Inhibitor",
	"cyp2b6": "CYP2B6_Inhibitor", "cyp inhibition cyp2b6": "CYP2B6_Inhibitor", "cyp2b6 inhibition": "CYP2B6_Inhibitor",
	"cyp2c9": "CYP2C9_Inhibitor", "cyp inhibition cyp2c9": "CYP2C9_Inhibitor", "cyp2c9 inhibition": "CYP2C9_Inhibitor",
	"cyp2c19": "CYP2C19_Inhibitor", "cyp inhibition cyp2c19": "CYP2C19_Inhibitor", "cyp2c19 inhibition": "CYP2C19_Inhibitor",
	"cyp2d6": "CYP2D6_Inhibitor", "cyp inhibition cyp2d6": "CYP2D6_Inhibitor", "cyp2d6 inhibition": "CYP2D6_Inhibitor",
	"cyp3a4": "CYP3A4_Inhibitor", "cyp inhibition cyp3a4": "CYP3A4_Inhibitor", "cyp3a4 inhibition": "CYP3A4_Inhibitor",
	"cyp3a4 mdz": "CYP3A4_Inhibitor", "cyp3a4 tst": "CYP3A4_Inhibitor",
	"cyp1a2 substrate": "CYP1A2_Substrate", "cyp2b6 substrate": "CYP2B6_Substrate",
	"cyp2c9 substrate": "CYP2C9_Substrate", "cyp2c19 substrate": "CYP2C19_Substrate",
	"cyp2d6 substrate": "CYP2D6_Substrate", "cyp3a4 substrate": "CYP3A4_Substrate",
	# Note: CYP1A1 and CYP2C8 are not in GIST format, so they won't be mapped
	# Solubility and lipophilicity
	"solubility": "Solubility", "aqueous solubility": "Solubility",
	"lipophilicity": "Lipophilicity", "logp": "Lipophilicity", "log p": "Lipophilicity",
 	"alogp": "Lipophilicity", "clogp": "Lipophilicity",
	"hydration energy": "hydrationE", "hydration": "hydrationE",
	# Fraction unbound
	"fu": "fu_df", "fraction unbound": "fu_df", "unbound fraction": "fu_df",
	"fu deeppk": "fu_deeppk",
	# Half-life
	"half life": "t1/2(reg)", "t1/2": "t1/2(reg)", "t12": "t1/2(reg)",
	"t0.5": "t0.5(cls)", "t05": "t0.5(cls)",
	# Clearance
	"plasma protein binding": "PPBR", "ppb": "PPBR", "clearance": "Clearance_total",
	"intrinsic clearance": "Clearance_Hepatocyte_AZ", "hepatocyte clearance": "Clearance_Hepatocyte_AZ",
	"microsome clearance": "Clearance_Microsome_AZ", "microsomal clearance": "Clearance_Microsome_AZ",
	"clp": "CLp(r)", "clp bc": "CLp(bc)", "clearance plasma": "CLp(r)",
	"mrt": "MRT", "mean residence time": "MRT",
	"ugt": "UGT_substrate", "ugt substrate": "UGT_substrate",
	# Liver microsomes
	"hlm": "HLM", "human liver microsome": "HLM", "human liver microsomes": "HLM",
	"rlm": "RLM", "rat liver microsome": "RLM", "rat liver microsomes": "RLM",
	"liver microsome": "HLM", "liver microsomes": "HLM", "metabolic stability liver microsomes": "HLM",
	"metabolic stability liver microsomes phase ii": "HLM",
	"hlc stability": "HLC_Stability", "hepatocyte stability": "HLC_Stability",
	"metabolic stability hepatocytes": "HLC_Stability", "hepatocytes": "HLC_Stability",
	"metabolic stability hepatocytes phase ii": "HLC_Stability",
	# Nuclear receptors
	"ahr": "NR-AhR", "aryl hydrocarbon receptor": "NR-AhR",
	"ar": "NR-AR", "androgen receptor": "NR-AR",
	"ar lbd": "NR-AR-LBD", "androgen receptor lbd": "NR-AR-LBD",
	"aromatase": "NR-Aromatase",
	"er": "NR-ER", "estrogen receptor": "NR-ER",
	"er lbd": "NR-ER-LBD", "estrogen receptor lbd": "NR-ER-LBD",
	"ppar gamma": "NR-PPAR-gamma", "ppar-gamma": "NR-PPAR-gamma",
	# Stress response
	"are": "SR-ARE", "antioxidant response element": "SR-ARE",
	"atad5": "SR-ATAD5",
	"hse": "SR-HSE", "heat shock element": "SR-HSE",
	"mmp": "SR-MMP", "mitochondrial membrane potential": "SR-MMP",
	"p53": "SR-p53",
	"gr": "GR", "glucocorticoid receptor": "GR",
	"tr": "TR", "thyroid receptor": "TR",
	# Toxicity & safety
	"skin reaction": "Skin Reaction", "respiratory toxicity": "Respir_tox", "liver toxicity": "Liver_tox_hepato",
	"dili": "DILI", "drug induced liver injury": "DILI",
	"herg": "hERG", "herg channel": "hERG", "toxicity herg": "hERG", "toxicity | herg": "hERG",
	"ames": "AMES", "ames test": "AMES", "toxicity ames": "AMES", "toxicity | ames": "AMES",
	"cytotoxicity": "ClinTox", "toxicity cytotoxicity": "ClinTox", "toxicity | cytotoxicity": "ClinTox",
	"genetoxicity": "Micronucleus", "toxicity genetoxicity": "Micronucleus", "toxicity | genetoxicity": "Micronucleus",
	"carcinogenicity": "Carcinogen", "carcinogen": "Carcinogen",
	"neurotoxicity": "Neuro_tox", "nephrotoxicity": "Nephro_tox", "mitochondrial toxicity": "Mito_tox",
	"hemolytic": "Hemolytic", "hemolysis": "Hemolytic",
	"reproductive toxicity": "Reprotox", "reprotox": "Reprotox",
	"eye irritation": "Eye_irritation", "eye corrosion": "Eye_corrosion",
	"skin irritation": "Skin Reaction", "micronucleus": "Micronucleus",
	"fdamdd": "FDAMDD(reg)", "fdamdd bc": "FDAMDD(bc)",
	"ld50": "LD50", "lethal dose 50": "LD50",
	"clintox": "ClinTox", "clinical toxicity": "ClinTox",
	"mouse ptd50": "Mouse_pTD50", "mouse ptd 50": "Mouse_pTD50",
	"rat ptd50": "Rat_pTD50", "rat ptd 50": "Rat_pTD50",
	# F20, F50 (less common, but add basic mappings)
	"f20": "F20", "f 20": "F20",
	"f50": "F50", "f 50": "F50",
}


@dataclass
class ManualConversionConfig:
	"""Configuration for ManualFormatConverter"""
	mapping_path: Optional[str] = None
	min_similarity: float = 0.55
	prefer_exact: bool = True


class ManualFormatConverter:
	"""Endpoint converter using deterministic heuristics (integrated from manual_converter.py)"""
	# CYP isoforms that actually have a GIST column; CYP1A1/CYP2C8 have none.
	CYP_GIST_ISOFORMS = frozenset({"cyp1a2", "cyp2b6", "cyp2c9", "cyp2c19", "cyp2d6", "cyp3a4"})
	# Endpoints present in v4.6 but with NO GIST column: never force-map them.
	UNMAPPABLE_TOKENS = frozenset({"cytotoxicity", "genetoxicity"})

	def __init__(self, config=None):
		self.config = config or ManualConversionConfig()
		self.gist_endpoints = self._load_gist_endpoints()
		self._gist_set = set(self.gist_endpoints)
		self._normalized_gist = {endpoint: self._normalize(endpoint) for endpoint in self.gist_endpoints}
		self._gist_tokens = {endpoint: self._tokenize(norm) for endpoint, norm in self._normalized_gist.items()}
		self.synonym_map = self._load_synonym_map()
	
	def _load_gist_endpoints(self):
		"""Load GIST endpoints (hardcoded to avoid file dependency)"""
		return GIST_ENDPOINTS.copy()
	
	def _load_synonym_map(self):
		"""Load synonym mapping"""
		mapping = {}
		for key, value in DEFAULT_ENDPOINT_SYNONYMS.items():
			mapping[self._normalize(key)] = value
		if self.config.mapping_path:
			# Convert relative path to absolute path based on data.py location
			mapping_path = Path(self.config.mapping_path).expanduser()
			if not mapping_path.is_absolute():
				base_dir = os.path.dirname(os.path.abspath(__file__))
				mapping_path = Path(base_dir) / mapping_path
			path = mapping_path
			if not path.exists():
				raise FileNotFoundError(f"Manual mapping file not found: {path}")
			if path.suffix.lower() == ".json":
				with open(path, "r", encoding="utf-8") as handler:
					file_map = json.load(handler)
				for key, value in file_map.items():
					mapping[self._normalize(key)] = value
			else:
				with open(path, "r", encoding="utf-8") as handler:
					reader = csv.DictReader(handler)
					if "source" not in reader.fieldnames or "target" not in reader.fieldnames:
						raise ValueError(f"Mapping CSV must contain 'source' and 'target' columns: {path}")
					for row in reader:
						mapping[self._normalize(row["source"])] = row["target"]
		return mapping
	
	@staticmethod
	def _normalize(value):
		"""Normalize endpoint string"""
		text = str(value).lower()
		text = text.replace("μ", "u").replace("µ", "u")
		text = re.sub(r"[^a-z0-9]+", " ", text)
		return re.sub(r"\s+", " ", text).strip()
	
	@staticmethod
	def _tokenize(normalized):
		"""Tokenize normalized string"""
		if not normalized:
			return tuple()
		return tuple(token for token in normalized.split(" ") if token)
	
	@staticmethod
	def _sequence_score(a, b):
		"""Calculate sequence similarity using Dice coefficient on bigrams"""
		if not a or not b:
			return 0.0
		def bigrams(text):
			return [text[i:i + 2] for i in range(len(text) - 1)] or [text]
		bigrams_a = bigrams(a)
		bigrams_b = bigrams(b)
		overlap = len(set(bigrams_a) & set(bigrams_b))
		score = (2.0 * overlap) / (len(bigrams_a) + len(bigrams_b))
		return max(0.0, min(score, 1.0))
	
	@staticmethod
	def _token_overlap(tokens_a, tokens_b):
		"""Calculate token overlap score"""
		if not tokens_a or not tokens_b:
			return 0.0
		set_a = set(tokens_a)
		set_b = set(tokens_b)
		intersection = len(set_a & set_b)
		union = len(set_a | set_b)
		return intersection / union if union else 0.0
	
	def _match_with_similarity(self, normalized, tokens):
		"""Match endpoint using similarity"""
		# Special handling for CYP endpoints: require exact CYP match
		# CYP2C9 and CYP2D6 are completely different, so similarity matching can cause errors
		if 'cyp' in normalized:
			# Extract CYP number from normalized string (e.g., "cyp2c9" -> "2c9")
			cyp_match = None
			for token in tokens:
				if token.startswith('cyp'):
					cyp_match = token
					break
			# If CYP is found, only match with exact CYP endpoint
			if cyp_match:
				for endpoint, gist_tokens in self._gist_tokens.items():
					# Check if endpoint is a CYP endpoint
					if 'cyp' in self._normalized_gist[endpoint].lower():
						# Extract CYP number from GIST endpoint
						gist_cyp_match = None
						for gist_token in gist_tokens:
							if gist_token.startswith('cyp'):
								gist_cyp_match = gist_token
								break
						# Only match if CYP numbers are exactly the same
						if gist_cyp_match and cyp_match == gist_cyp_match:
							# Additional similarity check to ensure it's a good match
							seq_score = self._sequence_score(normalized, self._normalized_gist[endpoint])
							overlap_score = self._token_overlap(tokens, gist_tokens)
							combined = max(seq_score, overlap_score)
							if combined >= self.config.min_similarity:
								return endpoint
				# If no exact CYP match found, return None (don't use similarity for CYP)
				return None
		
		# For non-CYP endpoints, use normal similarity matching
		best_score = 0.0
		best_endpoint = None
		for endpoint, gist_tokens in self._gist_tokens.items():
			seq_score = self._sequence_score(normalized, self._normalized_gist[endpoint])
			overlap_score = self._token_overlap(tokens, gist_tokens)
			combined = max(seq_score, overlap_score)
			if tokens and gist_tokens and tokens[0] == gist_tokens[0]:
				combined += 0.1
			if combined > best_score:
				best_score = combined
				best_endpoint = endpoint
		if best_score >= self.config.min_similarity:
			return best_endpoint
		return None
	
	@staticmethod
	def _contiguous_index(key_tokens, tokens):
		"""Start index of key_tokens as a contiguous sublist of tokens, else -1.
		Whole-token matching so short keys ('tr'/'gr') don't substring-match
		inside 'transporter'."""
		n, m = len(tokens), len(key_tokens)
		if m == 0 or m > n:
			return -1
		for i in range(n - m + 1):
			if tokens[i:i + m] == key_tokens:
				return i
		return -1

	def _cyp_guard(self, normalized):
		"""Map CYP endpoints to their exact GIST isoform column, never a neighbour.
		Returns a GIST column, '' (CYP with no GIST slot -> unmapped), or None."""
		isoforms = re.findall(r"cyp\d+[a-z]\d*", normalized)
		if not isoforms:
			return None
		iso = isoforms[0]
		role = "Substrate" if "substrate" in normalized else "Inhibitor"
		if iso in self.CYP_GIST_ISOFORMS:
			target = f"{iso.upper()}_{role}"
			if target in self._gist_set:
				return target
		return ""

	def match_endpoint(self, endpoint):
		"""Match single endpoint"""
		normalized = self._normalize(endpoint)
		if not normalized:
			return str(endpoint)
		tokens = self._tokenize(normalized)

		# CYP guard first: exact isoform only, no cross-isoform similarity bleed.
		cyp = self._cyp_guard(normalized)
		if cyp is not None:
			return cyp if cyp else str(endpoint)

		# Priority 1: Exact match
		if normalized in self.synonym_map:
			return self.synonym_map[normalized]

		# Endpoints with no GIST slot: leave unmapped (do not force via similarity).
		if any(tok in self.UNMAPPABLE_TOKENS for tok in tokens):
			return str(endpoint)

		# Priority 2: whole-token subsequence match; prefer longest key, then the
		# latest start (Measurement_Type is the last, most-specific component).
		best = None  # (num_tokens, start_index, mapped)
		for synonym_key, mapped in self.synonym_map.items():
			key_tokens = tuple(synonym_key.split())
			idx = self._contiguous_index(key_tokens, tokens)
			if idx < 0:
				continue
			cand = (len(key_tokens), idx, mapped)
			if best is None or cand[:2] > best[:2]:
				best = cand
		if best is not None:
			return best[2]

		# Priority 3: Similarity-based matching
		matched = self._match_with_similarity(normalized, tokens)
		if matched:
			return matched
		return str(endpoint)
	
	def match_endpoints(self, df, endpoint_column="endpoint"):
		"""Match endpoints in DataFrame"""
		if endpoint_column not in df.columns:
			raise ValueError(f"Endpoint column '{endpoint_column}' not found in dataframe.")
		unique_endpoints = df[endpoint_column].dropna().unique()
		mapping = {}
		for endpoint in unique_endpoints:
			mapping[str(endpoint)] = self.match_endpoint(endpoint)
		return mapping


##### Wrapper functions for preprocessing #####

def _ensure_bool(value):
	"""Convert value to bool"""
	if isinstance(value, bool):
		return value
	if isinstance(value, str):
		return value.strip().lower() in {"1", "true", "yes", "y", "on"}
	return bool(value)


def preprocess_to_gist(input_data, config=None, smiles_col='smiles_structure_parent',
					activity_col='measurement_value', endpoint_mapper='manual',
					manual_mapping_path=None, manual_min_similarity=0.55,
					manual_prefer_exact=True, gist_format_path=None,
					skip_preprocessing=False, fill_missing=False, **kwargs):
	"""
	Convert input data to GIST format matrix.
	
	Parameters:
	-----------
	input_data : str or pd.DataFrame
		CSV file path or DataFrame (can be preprocessed by preprocess_dataframe)
	config : dict, optional
		Configuration dictionary
	smiles_col : str, optional
		SMILES column name (auto-detected if None and skip_preprocessing=True)
	activity_col : str, optional
		Activity column name (auto-detected if None and skip_preprocessing=True)
	endpoint_mapper : str
		'llm' or 'manual' (default: 'manual', LLM not available)
	manual_mapping_path : str, optional
		Path to custom endpoint mapping CSV/JSON
	manual_min_similarity : float
		Minimum similarity for manual matching (default: 0.55)
	manual_prefer_exact : bool
		Prefer exact matches in manual mode (default: True)
	gist_format_path : str, optional
		Deprecated: GIST endpoints are now hardcoded in the code
	skip_preprocessing : bool
		If True, skip DataInspector normalization and assume data is already preprocessed.
		Auto-detects Standardized_SMILES and converted activity columns.
	fill_missing : bool
		If True, fill missing values with 0. If False (default), use NaN to distinguish
		between actual 0 measurements and missing data.
	**kwargs : dict
		Additional configuration options
	
	Returns:
	--------
	pd.DataFrame : GIST format matrix
	"""
	# Load data
	if isinstance(input_data, str):
		# Convert relative path to absolute path based on data.py location
		input_path = Path(input_data).expanduser()
		if not input_path.is_absolute():
			base_dir = os.path.dirname(os.path.abspath(__file__))
			input_path = Path(base_dir) / input_path
		inspector = DataInspector(input_path=str(input_path), smiles_column=smiles_col, activity_column=activity_col)
		df = inspector.df.copy()
		skip_preprocessing = False  # Force preprocessing for raw CSV files
		# Human PK is a wide-format, non-GIST schema: return the raw table instead
		# of forcing endpoint mapping (which yields an empty/garbage matrix).
		if getattr(inspector, "is_human_pk", False):
			import warnings as _warnings
			_warnings.warn("Human PK detected: returning raw wide-format table; "
						"not a GIST-mappable schema.")
			return df
	elif isinstance(input_data, pd.DataFrame):
		if skip_preprocessing:
			# Use DataFrame directly without DataInspector normalization
			df = input_data.copy()
			# Remove duplicate columns if any (prevent groupby errors)
			df = df.loc[:, ~df.columns.duplicated()]
		else:
			inspector = DataInspector(df=input_data, smiles_column=smiles_col, activity_column=activity_col)
			df = inspector.df.copy()
			# Remove duplicate columns if any (prevent groupby errors)
			df = df.loc[:, ~df.columns.duplicated()]
	else:
		raise ValueError("input_data must be a CSV path (str) or DataFrame")
	
	# Merge config
	if config:
		smiles_col = config.get('smiles_col', smiles_col)
		activity_col = config.get('activity_col', activity_col)
		endpoint_mapper = config.get('endpoint_mapper', endpoint_mapper)
		manual_mapping_path = config.get('manual_mapping_path', manual_mapping_path)
		manual_min_similarity = config.get('manual_min_similarity', manual_min_similarity)
		manual_prefer_exact = config.get('manual_prefer_exact', manual_prefer_exact)
		skip_preprocessing = config.get('skip_preprocessing', skip_preprocessing)
		fill_missing = config.get('fill_missing', fill_missing)
	
	# Auto-detect columns for preprocessed data
	if skip_preprocessing:
		# Auto-detect SMILES column (prefer Standardized_SMILES)
		if smiles_col is None or smiles_col not in df.columns:
			if 'Standardized_SMILES' in df.columns:
				smiles_col = 'Standardized_SMILES'
			elif 'smiles_structure_parent' in df.columns:
				smiles_col = 'smiles_structure_parent'
			elif 'smiles' in df.columns:
				smiles_col = 'smiles'
			else:
				# Try to find any column with 'smiles' in name (case-insensitive)
				smiles_candidates = [col for col in df.columns if 'smiles' in col.lower()]
				if smiles_candidates:
					smiles_col = smiles_candidates[0]
				else:
					raise ValueError("Could not detect SMILES column in preprocessed data")
		
		# Auto-detect activity column (prefer converted versions)
		if activity_col is None or activity_col not in df.columns:
			# Check for converted versions first
			if 'measurement_value_si' in df.columns:
				activity_col = 'measurement_value_si'
			elif 'measurement_value_scaled' in df.columns:
				activity_col = 'measurement_value_scaled'
			elif 'measurement_value' in df.columns:
				activity_col = 'measurement_value'
			else:
				# Try to find any column with 'measurement' or 'activity' in name
				activity_candidates = [col for col in df.columns 
									if 'measurement' in col.lower() or 'activity' in col.lower()]
				if activity_candidates:
					activity_col = activity_candidates[0]
				else:
					raise ValueError("Could not detect activity column in preprocessed data")
	else:
		# Normalize column names for raw data
		# Auto-detect SMILES column
		if smiles_col not in df.columns:
			if 'smiles_structure_parent' in df.columns:
				smiles_col = 'smiles_structure_parent'
			else:
				# Try to find any column with 'smiles' in name (case-insensitive)
				smiles_candidates = [col for col in df.columns if 'smiles' in col.lower()]
				if smiles_candidates:
					smiles_col = smiles_candidates[0]
				else:
					raise ValueError(f"SMILES column '{smiles_col}' not found in data. Available columns: {list(df.columns)}")
		
		# Auto-detect activity column (handle case variations)
		if activity_col not in df.columns:
			# Try case-insensitive match first
			activity_candidates = [col for col in df.columns 
								if col.lower() == activity_col.lower()]
			if activity_candidates:
				activity_col = activity_candidates[0]
			else:
				# Try to find any column with 'measurement' and 'value' in name
				activity_candidates = [col for col in df.columns 
									if 'measurement' in col.lower() and 'value' in col.lower()]
				if activity_candidates:
					activity_col = activity_candidates[0]
				else:
					raise ValueError(f"Activity column '{activity_col}' not found in data. Available columns: {list(df.columns)}")
	
	# Auto-detect unit column (handle case variations)
	unit_col = None
	if "measurement_unit" in df.columns:
		unit_col = "measurement_unit"
	else:
		# Try case-insensitive match
		unit_candidates = [col for col in df.columns if col.lower() == 'measurement_unit']
		if unit_candidates:
			unit_col = unit_candidates[0]
	
	# Build canonical endpoint string
	def _safe_str(x):
		return str(x) if (x is not None and x != "Not specified") else ""
	c1 = df["test"].apply(_safe_str) if "test" in df.columns else pd.Series([""] * len(df))
	c2 = df["test_type"].apply(_safe_str) if "test_type" in df.columns else pd.Series([""] * len(df))
	c3 = df["measurement_type"].apply(_safe_str) if "measurement_type" in df.columns else (
		df["measurment_type"].apply(_safe_str) if "measurment_type" in df.columns else pd.Series([""] * len(df)))
	df["endpoint_canonical"] = (c1.astype(str) + " | " + c2.astype(str) + " | " + c3.astype(str)).str.strip(" |")
	df.loc[df["endpoint_canonical"] == "", "endpoint_canonical"] = df.get("test", "")
	
	# Setup converter (using integrated ManualFormatConverter)
	mapper_choice = endpoint_mapper.lower()
	converter = None
	
	# LLM converter is not integrated (requires external API)
	# Only manual converter is available
	if mapper_choice == "llm":
		# LLM converter not available, fallback to manual
		pass
	
	# Use integrated manual converter
	manual_config = ManualConversionConfig(
		mapping_path=manual_mapping_path,
		min_similarity=float(manual_min_similarity),
		prefer_exact=_ensure_bool(manual_prefer_exact),
	)
	converter = ManualFormatConverter(manual_config)
	
	# Match endpoints
	tmp_df = pd.DataFrame({"endpoint": df["endpoint_canonical"].unique().tolist()})
	endpoint_map = converter.match_endpoints(tmp_df)
	df["gist_endpoint"] = df["endpoint_canonical"].map(endpoint_map).fillna(df["endpoint_canonical"])
	
	# Unit conversion (skip if already converted or skip_preprocessing is True)
	if skip_preprocessing and ('_si' in activity_col or '_scaled' in activity_col):
		# Already converted, use as-is
		value_col = activity_col
	elif unit_col is not None and not skip_preprocessing:
		try:
			uc = UnitConverter()
			# Fix: convert_column_to_si uses value_col parameter name, not value_col=activity_col
			df = uc.convert_column_to_si(df, activity_col, unit_col,
										new_value_col="value_si", new_unit_col="unit_si")
			value_col = "value_si"
		except Exception:
			value_col = activity_col
	else:
		value_col = activity_col
	
	# H6: qualitative Ames/Genetoxicity results store Positive/Negative in the
	# unit column with a '-' value; encode as 1.0/0.0 before numeric coercion.
	if unit_col is not None and unit_col in df.columns:
		unit_lower = df[unit_col].astype(str).str.strip().str.lower()
		qual_mask = unit_lower.isin(["positive", "negative"])
		if qual_mask.any():
			df.loc[qual_mask, value_col] = unit_lower[qual_mask].map(
				{"positive": 1.0, "negative": 0.0})

	df[value_col] = pd.to_numeric(df[value_col], errors='coerce')

	# Permeability: derive dimensionless Efflux_ratio (BtoA/AtoB) rows.
	if 'measurement_efflux_ratio' in df.columns:
		ratio_num = pd.to_numeric(df['measurement_efflux_ratio'], errors='coerce')
		er_mask = ratio_num.notna()
		if er_mask.any():
			er = df.loc[er_mask].copy()
			er['gist_endpoint'] = 'Efflux_ratio'
			er[value_col] = ratio_num.loc[er_mask].values
			df = pd.concat([df, er], ignore_index=True)
	
	# Load GIST columns (use hardcoded list)
	gist_cols = GIST_ENDPOINTS.copy()
	if not gist_cols:
		raise RuntimeError("GIST endpoint list is empty.")

	# M1/M2: report endpoints with no GIST target rather than dropping silently.
	gist_set = set(gist_cols)
	unmapped_mask = ~df["gist_endpoint"].isin(gist_set)
	if unmapped_mask.any():
		counts = df.loc[unmapped_mask, "endpoint_canonical"].value_counts()
		logger.warning(
			"%d row(s) across %d endpoint(s) have no GIST target and are excluded "
			"from the matrix: %s",
			int(unmapped_mask.sum()), len(counts),
			", ".join(f"{ep!r}x{int(n)}" for ep, n in counts.head(20).items()))

	# Detect PK rows
	def is_pk_row(row):
		txt = " ".join([str(row.get(col, "")) for col in ["test", "test_type", "measurement_type", "measurment_type", "test_subject"]]).lower()
		return ("patient" in txt) or ("pk" in txt)
	
	pk_mask = df.apply(is_pk_row, axis=1)
	
	# ADMET: aggregate mean per SMILES x endpoint
	admet_df = df.loc[~pk_mask].copy()
	admet_matrix = None
	if not admet_df.empty:
		# Ensure smiles_col is a string, not a list
		if isinstance(smiles_col, list):
			smiles_col = smiles_col[0] if smiles_col else 'smiles'
		# Remove duplicate columns from DataFrame if any
		admet_df = admet_df.loc[:, ~admet_df.columns.duplicated()]
		# Verify columns exist and are 1-dimensional
		if smiles_col not in admet_df.columns:
			raise ValueError(f"SMILES column '{smiles_col}' not found in data. Available columns: {list(admet_df.columns)}")
		if "gist_endpoint" not in admet_df.columns:
			raise ValueError(f"gist_endpoint column not found in data. Available columns: {list(admet_df.columns)}")
		if value_col not in admet_df.columns:
			raise ValueError(f"Value column '{value_col}' not found in data. Available columns: {list(admet_df.columns)}")
		agg = admet_df.groupby([smiles_col, "gist_endpoint"], dropna=False)[value_col].mean().reset_index()
		# Use NaN for missing values to distinguish from actual 0 measurements
		fill_val = 0 if fill_missing else np.nan
		admet_matrix = agg.pivot_table(index=smiles_col, columns="gist_endpoint", values=value_col, fill_value=fill_val).reset_index()
		for col in gist_cols:
			if col not in admet_matrix.columns:
				# Initialize missing columns with NaN (or 0 if fill_missing=True)
				admet_matrix[col] = fill_val
		admet_matrix = admet_matrix[[smiles_col] + gist_cols]
		admet_matrix = admet_matrix.rename(columns={smiles_col: "smiles"})
	
	# PK: keep duplicates by expanding rows
	pk_df = df.loc[pk_mask].copy()
	pk_matrix = None
	if not pk_df.empty:
		fill_val = 0 if fill_missing else np.nan
		# Use vectorized operations instead of iterrows() for better performance
		def process_pk_row(row):
			"""Process a single PK row into a dictionary"""
			rec = {"smiles": row.get(smiles_col, "")}
			# Initialize all GIST columns with fill_val
			for col in gist_cols:
				rec[col] = fill_val
			# Set the specific endpoint value if valid
			ge = row.get("gist_endpoint", None)
			if pd.notna(ge) and ge in gist_cols:
				val = row.get(value_col, None)
				if pd.notna(val):
					rec[ge] = val
				else:
					rec[ge] = fill_val
			return rec
		# Use apply() instead of iterrows() for better performance
		records = pk_df.apply(process_pk_row, axis=1).tolist()
		pk_matrix = pd.DataFrame.from_records(records)
	
	# Concatenate
	if admet_matrix is not None and pk_matrix is not None:
		out_df = pd.concat([admet_matrix, pk_matrix], ignore_index=True)
	elif admet_matrix is not None:
		out_df = admet_matrix
	elif pk_matrix is not None:
		out_df = pk_matrix
	else:
		out_df = pd.DataFrame(columns=["smiles"] + gist_cols)
	
	# Fill NaNs with 0 only if fill_missing=True
	if fill_missing:
		for col in out_df.columns:
			if col != "smiles":
				out_df[col] = out_df[col].fillna(0)
	else:
		# Keep NaN values to distinguish missing data from actual 0 measurements
		# Convert to float to properly handle NaN
		for col in out_df.columns:
			if col != "smiles":
				out_df[col] = pd.to_numeric(out_df[col], errors='coerce')
	
	return out_df


def preprocess_dataframe(df, task_type, task, config=None, smiles_column='smiles_structure_parent',
						activity_column='measurement_value', **kwargs):
	"""
	Preprocess DataFrame (long format) using Preprocessor.
	
	Note: This function expects LONG format data (one measurement per row),
	NOT GIST format (wide format with endpoints as columns).
	
	Parameters:
	-----------
	df : pd.DataFrame
		Input DataFrame in long format (one measurement per row).
		Expected columns:
		- smiles_column: SMILES strings
		- activity_column: Activity/measurement values
		- measurement_unit (optional): For unit conversion
		- test_type (optional): For pH correction
		- Test, test_subject, measurement_type (optional): Metadata
	task_type : str
		'classification' or 'regression'
	task : str or tuple
		Task identifier
	config : dict, optional
		Configuration dictionary
	smiles_column : str
		SMILES column name (default: 'smiles_structure_parent')
	activity_column : str
		Activity column name (default: 'measurement_value')
	**kwargs : dict
		Additional Preprocessor parameters
	
	Returns:
	--------
	pd.DataFrame : Preprocessed DataFrame with standardized SMILES and processed labels
	"""
	# Merge config with kwargs
	if config:
		kwargs.update(config)
	
	preprocessor = Preprocessor(
		df=df,
		task_type=task_type,
		task=task,
		smiles_column=smiles_column,
		activity_column=activity_column,
		**kwargs
	)
	return preprocessor.preprocess()
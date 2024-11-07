import numpy as np
from torch_geometric.datasets import QM9
from rdkit.Chem import Draw
import os
import torch
from collections import Counter
from openbabel import pybel
from rdkit import Chem
from xyz2mol import xyz2mol
import io
from rdkit import Chem
from contextlib import redirect_stderr
import re
from rdkit import RDLogger
import logging
from rdkit import rdBase
import matplotlib.pyplot as plt
import pandas as pd
import copy

import sys
sys.path.append('/Users/tommy/Documents/atdl/assignment3/DiGress/src/analysis')
from rdkit_functions import build_molecule

sys.path.append('/Users/tommy/Documents/atdl/assignment3/e3_diffusion_for_molecules/qm9')
sys.path.append('/Users/tommy/Documents/atdl/assignment3/e3_diffusion_for_molecules')
# from rdkit_functions import build_xae_molecule
import importlib
spec = importlib.util.spec_from_file_location("rdkit_functions_edm", "/Users/tommy/Documents/atdl/assignment3/e3_diffusion_for_molecules/qm9/rdkit_functions.py")
rdkit_functions_edm = importlib.util.module_from_spec(spec)
rdkit_functions_edm.__package__ = "qm9"
spec.loader.exec_module(rdkit_functions_edm)
build_xae_molecule = rdkit_functions_edm.build_xae_molecule
build_molecule_edm = rdkit_functions_edm.build_molecule


# Dictionary of maximum valences by atomic number (example values, expand as needed)
max_valences = {
    'H': 1,  # Hydrogen
    'C': 4,  # Carbon
    'N': 3,  # Nitrogen
    'O': 2,  # Oxygen
    'F': 1,   # Fluor
}

atom_decoder_no_hydrogen = {0: 6, 1: 7, 2: 8, 3: 9}
atom_decoder_hydrogen = {0: 1, 1: 6, 2: 7, 3: 8, 4: 9}
atom_decoder_edm = {'H': 0, 'C': 1, 'O': 2, 'N': 3, 'F': 4}
atom_decoder_edm = {'H': 1, 'C': 6, 'O': 7, 'N': 8, 'F': 9}

def check_valence_errors(X, E):
    errors = []
    num_atoms = len(X)
    
    for i in range(num_atoms):
        atom_type = X[i]
        atom_type_decoded = {0: 'H', 1: 'C', 2: 'N', 3: 'O', 4: 'F'}[atom_type]
        max_valence = max_valences[atom_type_decoded]
        
        # Sum of bond orders for this atom
        bond_count = sum(E[i])
        
        # Check if bond count exceeds the maximum allowed valence
        if bond_count > max_valence:
            errors.append((i, atom_type_decoded, bond_count, max_valence))
    
    return errors


def parse_digress_file(file_path):
    molecules = []
    with open(file_path, 'r') as file:
        molecule = {}
        matrix_lines = []
        for line in file:
            line = line.strip()
            
            if line.startswith('N='):
                # Save the previous molecule if any
                if molecule:
                    molecule['E'] = np.array(matrix_lines)
                    molecules.append(molecule)
                
                # Start a new molecule
                molecule = {'N': int(line.split('=')[1])}
                matrix_lines = []
            
            elif line.startswith('X:'):
                molecule['X'] = list(map(int, next(file).strip().split()))
            
            elif line.startswith('E:'):
                # Read the adjacency matrix lines
                for _ in range(molecule['N']):
                    matrix_lines.append(list(map(int, next(file).strip().split())))
        
        # Append the last molecule
        if molecule:
            molecule['E'] = np.array(matrix_lines)
            molecules.append(molecule)
    
    return molecules


# def check_valency_errors_qm9(x, edge_index, edge_attr, z):
#     errors = []
#     max_valences = {1: 1, 6: 4, 7: 3, 8: 2, 9: 1}  # Define max valences for common elements
    
#     # Define bond orders for each one-hot encoded bond type in `edge_attr`
#     bond_order_map = [1, 2, 3, 1.5]  # single, double, triple, aromatic

#     num_atoms = x.shape[0]
#     for i in range(num_atoms):
#         atom_type = int(z[i].item())  # Get atomic number
#         max_valence = max_valences.get(atom_type, None)
        
#         if max_valence is None:
#             print(f"Warning: Unknown atom type with atomic number {atom_type}")
#             continue
        
#         # Sum bond orders connected to this atom
#         bond_count = 0
#         for j in range(edge_index.shape[1]):
#             if edge_index[0, j] == i or edge_index[1, j] == i:  # Check if atom i is in this edge
#                 # Find the index of the one-hot encoded bond type
#                 bond_type_idx = int(edge_attr[j].argmax().item())
#                 bond_order = bond_order_map[bond_type_idx]
#                 bond_count += bond_order  # Accumulate bond order based on type
        
#         # Check if bond count exceeds max allowed valence
#         if bond_count > max_valence:
#             errors.append((i, atom_type, bond_count, max_valence))
    
#     return errors

def validate_valency_qm9(data):
    # Define standard valency
    valency_rules = {1: 1, 6: 4, 7: 3, 8: 2}  # H:1, C:4, N:3, O:2
    
    # Initialize a dictionary to store each atom's bond count
    bond_counts = {i: 0 for i in range(data.x.shape[0])}

    # Iterate over each edge to calculate bond orders
    for i, (src, dest) in enumerate(data.edge_index.t()):
        bond_order = int(data.edge_attr[i].argmax())  # Assuming the bond order corresponds to the max index
        bond_counts[src.item()] += bond_order
        bond_counts[dest.item()] += bond_order

    # Check for valency errors
    errors = []
    for atom_idx, atom_type in enumerate(data.z):
        expected_valency = valency_rules.get(atom_type.item(), None)
        if expected_valency and bond_counts[atom_idx] != expected_valency:
            errors.append((atom_idx, atom_type.item(), bond_counts[atom_idx], expected_valency))

    return errors


def check_valency_qm9_rdkit(smiles):
    try:
        # Convert SMILES to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "Error: Invalid SMILES"

        # Iterate over atoms to check valency
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            atomic_num = atom.GetAtomicNum()
            explicit_valence = atom.GetExplicitValence()
            expected_valence = Chem.GetPeriodicTable().GetDefaultValence(atomic_num)
            if symbol == 'N':
                expected_valence += 1

            # Check if explicit valence exceeds expected valence
            if explicit_valence > expected_valence:
                return f"Valency error in {symbol} with valence {explicit_valence} (expected max {expected_valence})"
        
    except Exception as e:
        return f"Error during valency check: {str(e)}"

def check_valency_qm9_graph(data):

    # Define standard valency limits
    valency_limits = {1: 1, 6: 4, 7: 3, 8: 2}  # H:1, C:4, N:3, O:2
    
    # Initialize dictionary to count bond orders for each atom
    bond_counts = {i: 0 for i in range(data.x.shape[0])}

    # Iterate through each edge and add bond orders
    for idx, (src, dest) in enumerate(data.edge_index.t()):
        # Determine bond order (assuming one-hot encoded bond type)
        bond_order = data.edge_attr[idx].argmax().item() + 1

        # Increment bond counts only for one direction to avoid double-counting
        if data.z[src].item() != 1:  # Skip hydrogen as neighbor for src
            bond_counts[src.item()] += bond_order
        if data.z[dest].item() != 1:  # Skip hydrogen as neighbor for dest
            bond_counts[dest.item()] += bond_order

    # Check for valency errors
    errors = []
    for atom_idx, atomic_number in enumerate(data.z):
        expected_valency = valency_limits.get(atomic_number.item(), None)
        if expected_valency is not None and bond_counts[atom_idx] > expected_valency:
            errors.append((atom_idx, atomic_number.item(), bond_counts[atom_idx], expected_valency))

    return errors


def parse_xyz(filepath):

    with open(filepath, 'r') as file:
        lines = file.readlines()

    num_atoms = int(lines[0].strip())

    atom_types = []
    positions = []
    for line in lines[2:num_atoms + 2]:
        atom_type, x, y, z = line.split()
        atom_types.append(atom_decoder_edm[atom_type])
        # atom_types.append(atom_type)
        positions.append([float(x), float(y), float(z)])

    # Convert to torch tensors
    # atom_types = torch.tensor(atom_types, dtype=torch.int)
    positions = torch.tensor(positions, dtype=torch.float)

    return positions, atom_types



def check_valence_errors_mol(mol):
    """
    Checks for valence errors in a molecule provided as a SMILES string,
    RDKit Mol object, or Pybel Molecule object.

    Parameters:
    - input_mol: str, Chem.Mol, or pybel.Molecule

    Returns:
    - Nested dictionary valence_error_counts[atom_symbol][valence]: count of errors.
    """

    # Perform valence check and collect error counts
    valence_error_counts = Counter()
    for atom in mol.GetAtoms():
        atom.UpdatePropertyCache(strict=False)  # Update valence information
        atomic_num = atom.GetAtomicNum()
        symbol = atom.GetSymbol()
        total_valence = atom.GetTotalValence()
        formal_charge = atom.GetFormalCharge()

        # Adjust total valence by adding formal charge if desired
        adjusted_valence = total_valence + formal_charge  # Use this if you want total_valence + formal_charge
        # adjusted_valence = total_valence  # Use this if you want total_valence only

        allowed_valences = Chem.GetPeriodicTable().GetValenceList(atomic_num)
        # Adjust allowed valences based on formal charge if desired
        adjusted_allowed_valences = [val - formal_charge for val in allowed_valences]  # Adjusted for formal charge
        # adjusted_allowed_valences = allowed_valences  # Use this if not adjusting for formal charge

        if total_valence not in adjusted_allowed_valences:
            # valence_error_counts[symbol][valence_key] = valence_error_counts[symbol].get(valence_key, 0) + 1
            valence_error_counts[symbol + ':' + str(adjusted_valence)] += 1

    return valence_error_counts


def count_bond_types(mol, add_hydrogen=True):
    counter = Counter()

    if add_hydrogen is True:
        mol = Chem.AddHs(mol)

    for bond in mol.GetBonds():
        bond_type = bond.GetBondType()
        counter['total'] += 1
        # atom1 = bond.GetBeginAtom()
        # atom2 = bond.GetEndAtom()
        # atom1_type = atom1.GetSymbol()
        # atom2_type = atom2.GetSymbol()
        if bond_type == Chem.rdchem.BondType.SINGLE:
            counter['single'] += 1
        elif bond_type == Chem.rdchem.BondType.DOUBLE:
            counter['double'] += 1
        elif bond_type == Chem.rdchem.BondType.TRIPLE:
            counter['triple'] += 1
        elif bond_type == Chem.rdchem.BondType.AROMATIC:
            counter['aromatic'] += 1

    return counter


def process_edm():

    # EDM
    dataset_info = {
        'atom_encoder': {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4},
        'atom_decoder': ['H', 'C', 'N', 'O', 'F'],
        'name': 'qm9',
        }
    counter = Counter()
    valence_error_counts = Counter()
    pattern = re.compile(r"Explicit valence for atom # \d+ (\w), (\d+), is greater than permitted")
    counter_bond_types = Counter()
    for i in range(10000):
        # counter_molecule = Counter()  # not necessary, because of assert len(matches) == 1
        if i % 100 == 0:
            print(i)

        xyz_file = f'e3_diffusion_for_molecules/generated_samples/molecule_{i:03}.txt'

        # Chem.MolFromXYZFile also fails
        # mol_rdkit = Chem.MolFromXYZFile(xyz_file)  # MolFromXYZFile seems corrupted, so revert to pybel
        # Chem.SanitizeMol(mol_rdkit)

        # xyz2mol also fails
        # positions, atom_types = parse_xyz(xyz_file)
        # print('atom_types', atom_types)
        # print(positions.tolist())
        # mol_xyz2mol = xyz2mol(atom_types, positions.tolist())
        # print(mol_xyz2mol)
        # smiles_xyz2mol = Chem.MolToSmiles(mol_xyz2mol)
        # print(smiles_xyz2mol)

        # Chem.SDMolSupplier also fails
        # sdf_file = 'tmp.sdf'
        # mol_pybel.write("sdf", sdf_file, overwrite=True)
        # supplier = Chem.SDMolSupplier(sdf_file)
        # mol = supplier[0]

        # build_xae_molecule also fails
        # positions, atom_types = parse_xyz(xyz_file)
        # X, A, E = build_xae_molecule(positions, atom_types, dataset_info)
        # print(X)
        # print(E)
        # errors = check_valence_errors(X, E)
        # mol = build_molecule(
        #     torch.Tensor(X).int(),
        #     torch.Tensor(E).int(),
        #     atom_decoder_hydrogen,
        # )

        mol_pybel = next(pybel.readfile("xyz", xyz_file))
        smiles_pybel = mol_pybel.write('can')
        mol_rdkit = Chem.MolFromSmiles(smiles_pybel)
        if mol_rdkit is not None:
            Chem.SanitizeMol(mol_rdkit, Chem.SanitizeFlags.SANITIZE_ALL)  # tmp!!!
            counter_bond_types += count_bond_types(mol_rdkit, add_hydrogen=False)

        if mol_rdkit is None:
            print(smiles_pybel)

            stderr_buffer = io.StringIO()
            logger = logging.getLogger('rdkit')
            logger.addHandler(logging.StreamHandler(stderr_buffer))
            rdBase.LogToPythonLogger()
            mol_rdkit = Chem.MolFromSmiles(smiles_pybel)
            stderr = stderr_buffer.getvalue().strip()
            pattern = r"\[\d{2}:\d{2}:\d{2}\]\s+Explicit valence for atom # \d+ (\w), (\d+), is greater than permitted"
            matches = re.findall(pattern, stderr)
            assert len(matches) == 1, len(matches)
            match = matches[0]
            counter[match] += 1
            counter[('Total', '')] += 1
            path = f'atdl3/output/graph2png/edm_pybel/{match[0]}_{match[1]}/{i}.png'
            mol_pybel.removeh()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            mol_pybel.draw(show=False, filename=path, update=True, usecoords=False)
            continue

    return counter, counter_bond_types


def process_digress_graph():

    file_path = 'DiGress/generated_samples/generated_samples_qm9withH.txt'
    atom_decoder = atom_decoder_hydrogen
    molecules = parse_digress_file(file_path)
    print(f"Parsed {len(molecules)} molecules.")
    counter = Counter()
    counter_bond_types = Counter()
    for i, molecule in enumerate(molecules):
        counter_molecule = Counter()
        errors = check_valence_errors(molecule['X'], molecule['E'])
        if errors:
            print(f"Molecule with N={molecule['N']} has valence errors: {errors}")
            # counter[errors[0][1]] += 1
            print('molecule', molecule)

            # Do mol to draw it and count bonds.
            mol = build_molecule(
                torch.Tensor(molecule['X']).int(),
                torch.Tensor(molecule['E']).int(),
                atom_decoder,
            )

            for error in errors:
                counter_molecule[tuple(error[1:3])] = 1  # count only once per molecule
                path = f'atdl3/output/graph2png/digress/{error[1]}_{error[2]}/{i}.png'
                if os.path.isfile(path):
                    continue
                img = Draw.MolToImage(mol, size=(300, 300))
                os.makedirs(os.path.dirname(path), exist_ok=True)
                img.save(path)
            counter_molecule[('Total', '')] = 1
        else:
            # build molecule no matter what for the purpose of counting aromatic bonds
            mol = build_molecule(
                torch.Tensor(molecule['X']).int(),
                torch.Tensor(molecule['E']).int(),
                atom_decoder,
            )
        counter += counter_molecule
        try:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)  # Enforce aromaticity!
        except Chem.rdchem.AtomValenceException:
            continue
        except Chem.rdchem.KekulizeException:
            continue
        assert mol is not None
        if mol is None:
            continue
        # mol.UpdatePropertyCache(strict=False)  # suggested by Lu
        counter_bond_types += count_bond_types(mol, add_hydrogen=False)

    return counter, counter_bond_types


def process_digress_smiles():

    file_path = 'DiGress/generated_samples/final_smiles_qm9withH.txt'
    atom_decoder = atom_decoder_hydrogen
    molecules = parse_digress_file(file_path)
    print(f"Parsed {len(molecules)} molecules.")
    counter_bond_types = Counter()
    with open(file_path) as file:
        for line in file:
            smiles = line.strip()
            if smiles == 'None':
                continue
            mol = Chem.MolFromSmiles(smiles)
            assert mol is not None
            counter_bond_types += count_bond_types(mol, add_hydrogen=False)

    return counter_bond_types


def process_qm9():

    # placeholder for function from Lu
    # counter = Counter()
    # _ = {'C': {2: 899, 5: 510}, 'N': {5: 21826, 1: 9082, 4: 8}, 'O': {0: 334, 4: 1}}
    # for k1 in _.keys():
    #     for k2, v in _[k1].items():
    #         counter[(k1, k2)] = v
    # n = 134000

    root = 'data/QM9'
    dataset = QM9(root=root)
    suppl = Chem.SDMolSupplier(f'{root}/raw/gdb9.sdf')
    mol_dict = {mol.GetProp('_Name'): mol for mol in suppl if mol is not None}
    mol_list = [mol for mol in mol_dict.values() if mol is not None]

    valence_error_counts = {}

    counter = Counter()
    counter_bond_types = Counter()
    for (i, mol) in enumerate(mol_list):
        counter_bond_types += count_bond_types(mol, add_hydrogen=True)
        # mol = Chem.AddHs(mol)  # tmp!!!
        counter_molecule = Counter()
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_ALL)  # tmp!!!
        mol_copy = copy.deepcopy(mol)
        for atom in mol_copy.GetAtoms():
            atom.UpdatePropertyCache(strict=False)

            atom_symbol = atom.GetSymbol()
            atomic_num = atom.GetAtomicNum()
            total_valence = atom.GetExplicitValence() + atom.GetImplicitValence()

            allowed_valences = Chem.GetPeriodicTable().GetValenceList(atomic_num)

            formal_charge = atom.GetFormalCharge()
            if formal_charge != 0:
                allowed_valences = [valence - formal_charge for valence in allowed_valences]

            if total_valence not in allowed_valences:
                num_bonds = atom.GetDegree()

            # if(atom_symbol == "C" and total_valence + formal_charge == 5):
            #     print(i)

                if atom_symbol not in valence_error_counts:
                    valence_error_counts[atom_symbol] = {}

                valence_error_counts[atom_symbol][total_valence + formal_charge] = valence_error_counts[atom_symbol].get(total_valence + formal_charge, 0) + 1
                counter_molecule[(atom_symbol, total_valence + formal_charge)] = 1
                # counter_molecule[(atom_symbol, total_valence)] = 1
                counter_molecule[('Total', '')] = 1
        counter += counter_molecule

    n = i + 1

    return counter, n, counter_bond_types




def plot_bond_type_percentages(counter_qm9, counter_edm, counter_digress):
    # Prepare data
    datasets = ["QM9", "EDM", "DiGress"]
    counters = [counter_qm9, counter_edm, counter_digress]

    bond_type_order = ['Single', 'Double', 'Triple', 'Aromatic']

    # Transform data to a DataFrame for easy plotting
    data = []
    for dataset, counter in zip(datasets, counters):
        total = counter['total']  # Total bonds for percentage calculation
        for bond_type in map(str.lower, bond_type_order):
            count = counter[bond_type]
            if bond_type != 'total':  # Skip the 'total' key
                percentage = (count / total) * 100  # Calculate percentage
                data.append({
                    "Dataset": dataset,
                    "Bond Type": bond_type.replace('_', ' ').title(),
                    "Percentage": percentage,
                })
    
    df = pd.DataFrame(data)
    print(df)

    # Pivot the data for grouped bar plot
    df_pivot = df.pivot(index="Bond Type", columns="Dataset", values="Percentage").fillna(0)
    print(df_pivot)
    df_pivot = df_pivot.reindex(bond_type_order)  # Ensure bond types are in specified order to make Lu happy :-)
    print(df_pivot)

    # Plot grouped bar chart with customized bar width and offsets
    bar_width = 0.25
    x = np.arange(len(df_pivot))  # the label locations
    offsets = [-bar_width, 0, bar_width]  # Offset each dataset's bar position

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot each dataset separately with a different offset
    for idx, (dataset, offset) in enumerate(zip(datasets, offsets)):
        ax.bar(
            x + offset,
            df_pivot[dataset],
            width=bar_width,
            label=dataset,
            color=f"C{idx}",
        )

    plt.title("Bond Type Percentages Across Datasets")
    plt.xlabel("Bond Type")
    plt.ylabel("Percentage of All Bonds")
    plt.ylim(0, 100)  # Set y-axis limit for percentage

    plt.legend(title="Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(df_pivot.index, rotation=45, ha="right")

    # Save or display the plot
    plt.tight_layout()

    plt.savefig('bond_type_dist.png')


def plot_valence_errors(counter_edm, counter_digress, counter_qm9, size_edm, size_digress, size_qm9):
    # Prepare data
    datasets = ["QM9", "DiGress", "EDM"]
    sizes = [size_qm9, size_digress, size_edm]
    counters = [counter_qm9, counter_digress, counter_edm]

    # Transform data to a DataFrame for easy plotting
    data = []
    for dataset, size, counter in zip(datasets, sizes, counters):
        for (atom, valence), count in counter.items():
            percentage = (count / size) * 100  # Calculate percentage
            data.append({
                "Dataset": dataset,
                "Atom-Valence": f"{atom}-{valence}",
                "Percentage": percentage,
                "Count": count,
                "Size": size,
            })

    df = pd.DataFrame(data)
    print('df', df)

    # Pivot the data for grouped bar plot
    df_pivot = df.pivot(
        index="Atom-Valence",
        columns="Dataset",
        values="Percentage",
    ).fillna(0)
    print('df_pivot', df_pivot)

    # # Plot grouped bar chart
    # ax = df_pivot.plot(kind="bar", figsize=(12, 6), width=0.7)

    # Make it a bit easier to associate the labels and bars by changing bar width
    bar_width = 0.1
    x = np.arange(len(df_pivot))  # the label locations
    offsets = [-bar_width, 0, bar_width]  # Offset each dataset's bar position

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot each dataset separately with a different offset to increase spacing between groups
    for idx, (dataset, offset) in enumerate(zip(datasets, offsets)):
        ax.bar(
            x + offset,
            df_pivot[dataset],
            width=bar_width,
            label=dataset,
            color=f"C{idx}",
        )

    plt.title("Percentage of Valence Errors by Atom-Valence Type Across Datasets")
    plt.xlabel("Atom-Valence")
    plt.ylabel("Percentage of Errors")
    ymax = 4.5
    plt.ylim(0, ymax)
    plt.legend(title="Dataset")
    # plt.xticks(rotation=45, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(df_pivot.index, rotation=45, ha="right")

    # Indicate values that exceed the y-axis range by displaying the actual value
    for idx, dataset in enumerate(datasets):
        for i, value in enumerate(df_pivot[dataset]):
            if value > ymax:  # If the value exceeds the y-axis limit
                ax.annotate(
                    f'{value:.2f}',  # Display the actual value formatted to 2 decimal places
                    xy=(i + (idx - 1) * 0.25, ymax),  # Position just above the y-limit
                    xytext=(0, -5),  # Offset to avoid overlap with the bar
                    textcoords="offset points",
                    ha='center',
                    va='bottom',
                    color='red',  # Text color
                    fontsize=10,
                    fontweight='bold'
                )

    plt.savefig('valence_errors.png')


# Execute the comparison
if __name__ == "__main__":

    '''
    Kan vi lave et plot ligesom functional groups for valence?
    Have atom typen på x aksen og y aksen være den relative procentdel af alle valence fejl for alle dataset?
    '''

            # for atom in rdkit_mol.GetAtoms():
            #     if atom.GetExplicitValence() > Chem.GetPeriodicTable().GetDefaultValence(atom.GetAtomicNum()):


#     suppl = Chem.SDMolSupplier('data/QM9/raw/gdb9.sdf')
#     # for mol in suppl:
#     #     if mol.GetProp('_Name') is None:
#     #         continue
#     #     print(Chem.MolToSmiles(mol))
#     mol_dict = {mol.GetProp('_Name'): mol for mol in suppl if mol is not None}
#     mol_list = [mol for mol in mol_dict.values() if mol is not None]
#     idx = 32801
#     for mol in mol_list:
#         for atom in mol.GetAtoms():
#             formal_charge = atom.GetFormalCharge()
#             if formal_charge:
#                 smiles = Chem.MolToSmiles(mol)
#                 print(smiles)
#                 break
#     for idx in map(int, '''
# 6243

# 28999
# 30433
# 30437
# 124959
# 127317
# 127330
# 127370
# 129498'''.split()):
#         mol = mol_list[idx]
#         smiles = Chem.MolToSmiles(mol)
#         print(smiles)
#         # img = Draw.MolToImage(mol, size=(300, 300))
#         # img.save(f'{idx}.png')
#         mol_pybel = pybel.readstring("smi", smiles)
#         filename = f'{idx}.png'
#         mol_pybel.draw(show=False, filename=filename, update=True, usecoords=False)
#         print(filename)

#     stop66666


    # dataset = QM9(root='data/QM9')
    # cnt = 0 
    # for data in dataset:
    #     # errors = validate_valency_qm9(data)
    #     # errors = check_valency_qm9_rdkit(data.smiles)
    #     errors = check_valency_qm9_graph(data)
    #     atoms = list(map(int, data.z))
    #     coordinates = data.pos.tolist()
    #     ac, raw_mol = xyz2mol(atoms, coordinates)
    #     Chem.MolToSmiles(raw_mol)
    #     print(raw_mol)
    #     stop1
    #     mol = Chem.Mol(raw_mol)
    #     # mol = Chem.MolFromSmiles(data.smiles)
    #     smiles = Chem.MolToSmiles(mol)
    #     if errors:
    #         print(errors)
    #         print(data.smiles)
    #         mol = Chem.MolFromSmiles(data.smiles)
    #         d = vars(data)
    #         for k, v in d['_store'].items():
    #             print(k)
    #             print(v)
    #         stop2
    #     continue
    #     stop2
    #     molecule = data
    #     errors = check_valency_errors_qm9(
    #         molecule.x, molecule.edge_index, molecule.edge_attr, molecule.z)
    #     if errors:
    #         cnt += 1
    #         print(data.smiles)
    #         print(errors)
    #         stop1
    # stop999999

    print('Processing QM9')
    counter_qm9, size_qm9, counter_bond_types_qm9 = process_qm9()

    print('Processing DiGress')
    counter_digress, counter_bond_types_digress = process_digress_graph()
    print(counter_bond_types_digress)
    # counter_bond_types_digress = process_digress_smiles()
    print(counter_bond_types_digress)
    size_digress = 10000

    print('Processing EDM')
    counter_edm, counter_bond_types_edm = process_edm()
    size_edm = 10000


    print('counter_edm', counter_edm)
    print('counter_digress', counter_digress)
    print('counter_qm9', counter_qm9)

    plot_valence_errors(
        counter_edm, counter_digress, counter_qm9,
        size_edm, size_digress, size_qm9,
    )

    plot_bond_type_percentages(
        counter_bond_types_qm9,
        counter_bond_types_edm,
        counter_bond_types_digress,
    )

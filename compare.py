import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter
from rdkit.Chem import Draw
import json

# Define additional functional groups
with open('atdl3/functional_groups.json') as file:
    functional_groups_smarts = {k: Chem.MolFromSmarts(v) for k, v in json.load(file).items()}

os.makedirs('mols', exist_ok=True)


def calculate_properties_from_mol(mol):
    if mol is None:
        return None, None
    mol_wt = Descriptors.MolWt(mol)
    heavy_atom_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return mol_wt, heavy_atom_count

# Calculate functional groups
def count_functional_groups_from_mol(mol, functional_groups_smarts):
    functional_groups = {group_name: 0 for group_name in functional_groups_smarts.keys()}
    for group_name, smarts in functional_groups_smarts.items():
        if mol.HasSubstructMatch(smarts):
            functional_groups[group_name] += 1
    return functional_groups


# Process generated samples with validation
def process_generated_samples_edm():
    generated_properties = {
        'mol_wts': [],
        'heavy_atom_counts': [],
        'functional_group_counts': Counter(),
    }

    for i in range(10000):
        print(i)
        xyz_file = f"e3_diffusion_for_molecules/generated_samples/molecule_{i:03}.txt"
        mol_pybel = next(pybel.readfile("xyz", xyz_file))

        # mol, smiles, inchi = xyz_to_mol_smiles_inchi(xyz_file)
        # if not (smiles and inchi):
        #     print(smiles, inchi)
        #     stop1
        smiles = mol_pybel.write('smi')
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue

        mol_wt, heavy_atom_count = calculate_properties_from_mol(mol)

        generated_properties['mol_wts'].append(mol_wt)
        generated_properties['heavy_atom_counts'].append(heavy_atom_count)

        func_groups = count_functional_groups_from_mol(mol, functional_groups_smarts)
        for group_name, count in func_groups.items():
            generated_properties['functional_group_counts'][group_name] += count

    return generated_properties


# Load QM9 dataset
def process_qm9_dataset_old():
    dataset = QM9(root='data/QM9')
    qm9_properties = {
        'mol_wts': [],
        'heavy_atom_counts': [],
        'functional_group_counts': Counter(),
    }

    for functional_group in functional_groups_smarts.keys():
        qm9_properties['functional_group_counts'][functional_group] = 0

    skipped = set()
    for data in dataset:
        mol_babel = pybel.readstring("smi", data.smiles)
        mol_rdkit = Chem.MolFromSmiles(data.smiles)
        if mol_babel is None and mol_rdkit is not None:
            print(data.smiles)
            stop1
        # if mol_babel is not None and mol_rdkit is None:
        #     print(data.smiles)
        #     stop2
        mol = mol_rdkit
        if mol is None:
            print('skipping qm9', data.idx, data.smiles)
            skipped.add(data.smiles)
            continue
        mol_to_png(mol, f'png/qm9/{int(data.idx)}.png')

        mol_wt, heavy_atom_count = calculate_properties_from_mol(mol)
        qm9_properties['mol_wts'].append(mol_wt)
        qm9_properties['heavy_atom_counts'].append(heavy_atom_count)
        
        func_groups = count_functional_groups_from_mol(mol, functional_groups_smarts)
        for group_name, count in func_groups.items():
            qm9_properties['functional_group_counts'][group_name] += count

    with open('qm9_skip.txt', 'w') as file:
        for smiles in skipped:
            print(smiles, file=file)

    return qm9_properties


def process_qm9_dataset():

    qm9_properties = {
        'mol_wts': [],
        'heavy_atom_counts': [],
        'functional_group_counts': Counter(),
    }

    # Initialize the dictionary. Should probably just use counter instead. Whatever.
    for functional_group in functional_groups_smarts.keys():
        qm9_properties['functional_group_counts'][functional_group] = 0

    root = 'data/QM9'
    dataset = QM9(root=root)
    suppl = Chem.SDMolSupplier(f'{root}/raw/gdb9.sdf')
    mol_dict = {mol.GetProp('_Name'): mol for mol in suppl if mol is not None}
    mol_list = [mol for mol in mol_dict.values() if mol is not None]

    for mol in mol_list:
        mol_wt, heavy_atom_count = calculate_properties_from_mol(mol)
        qm9_properties['mol_wts'].append(mol_wt)
        qm9_properties['heavy_atom_counts'].append(heavy_atom_count)
        
        func_groups = count_functional_groups_from_mol(mol, functional_groups_smarts)
        for group_name, count in func_groups.items():
            qm9_properties['functional_group_counts'][group_name] += count

    return qm9_properties


# Plot histograms for comparison with dual y-axis and save as image
def compare_histograms(data1, data2, label1, label2, title, xlabel, filename):
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # Plot for QM9 dataset on the left y-axis
    ax1.hist(data2, bins=30, alpha=0.5, color='blue', label=label2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(f'Frequency - {label2}', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Plot for generated samples on the right y-axis
    ax2 = ax1.twinx()
    ax2.hist(data1, bins=30, alpha=0.5, color='green', label=label1)
    ax2.set_ylabel(f'Frequency - {label1}', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    # Title and layout
    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory


# Helper function to save histograms with separate y-axes
def save_dual_axis_histogram(data1, data2, label1, label2, title, xlabel, filename):

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.hist(data2, bins=30, alpha=0.5, color='blue', label=label2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(f'{label2} Frequency', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.hist(data1, bins=30, alpha=0.5, color='green', label=label1)
    ax2.set_ylabel(f'{label1} Frequency', color='green')
    ax2.tick_params(axis='y', labelcolor='green')

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()


# Helper function to save functional group comparison with dual percentage bars
def save_functional_group_comparison(
        func_qm9, func_digress_noH, func_digress_withH, func_edm, filename):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    labels = list(func_edm.keys())
    counts_edm = np.array([func_edm[group] for group in labels])
    counts_qm9 = np.array([func_qm9[group] for group in labels])
    counts_digress_noH = np.array([func_digress_noH[group] for group in labels])
    counts_digress_withH = np.array([func_digress_withH[group] for group in labels])
    
    # Convert counts to percentages for each dataset
    percentages_edm = (counts_edm / counts_edm.sum()) * 100
    percentages_qm9 = (counts_qm9 / counts_qm9.sum()) * 100
    percentages_digress_noH = (counts_digress_noH / counts_digress_noH.sum()) * 100
    percentages_digress_withH = (counts_digress_withH / counts_digress_withH.sum()) * 100
    
    x = np.arange(len(labels))
    width = 0.2

    # Plot percentages
    ax1.bar(x - width, percentages_edm, width, color='#a6cee3', label='EDM %')
    ax1.bar(x, percentages_qm9, width, color='#1f78b4', label='QM9 %')
    ax1.bar(x + width, percentages_digress_withH, width, color='#33a02c', label='DiGress withH %')
    ax1.bar(x + 2*width, percentages_digress_noH, width, color='#b2df8a', label='DiGress noH %')

    # Labeling and aesthetics
    ax1.set_xlabel('Functional Group')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Functional Group Percentages')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    
    # Add legend for clarity
    ax1.legend()

    # Adjust layout and save the plot
    fig.tight_layout()
    plt.savefig(filename)
    plt.close()



# Helper function to save scatter plot for functional group comparison with selective labeling
def save_functional_group_scatter(generated_func, qm9_func, filename, threshold=500):
    labels = list(generated_func.keys())
    generated_counts = np.array([generated_func[group] for group in labels])
    qm9_counts = np.array([qm9_func[group] for group in labels])

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(qm9_counts, generated_counts, color='purple', alpha=0.6)
    
    # Add a diagonal line for reference (if perfectly correlated, points lie on this line)
    max_val = max(max(generated_counts), max(qm9_counts))
    plt.plot([0, max_val], [0, max_val], 'k--', linewidth=1)
    
    # Label points far from the diagonal based on threshold
    for i, label in enumerate(labels):
        # Calculate distance from the diagonal
        distance = abs(generated_counts[i] - qm9_counts[i])
        if distance > threshold:
            plt.annotate(label, (qm9_counts[i], generated_counts[i]), fontsize=9, ha='right')

    # Set plot labels and title
    plt.xlabel('QM9 Count')
    plt.ylabel('Generated Count')
    plt.title('Functional Group Frequency Comparison: Generated vs QM9')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename)
    plt.close()  # Close the plot to free up memory


def atom_to_symbol(atomic_num):
    """Convert atomic number to element symbol."""
    periodic_table = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"} 
    return periodic_table[atomic_num]


def build_rdkit_mol_with_distances(atom_types, positions):
    mol = Chem.RWMol()
    atom_map = {}  # Map for tracking added atom indices
    
    # Add atoms to RDKit molecule
    for i, atomic_num in enumerate(atom_types):
        atom = Chem.Atom(int(atomic_num))
        atom_idx = mol.AddAtom(atom)
        atom_map[i] = atom_idx

    # Determine bonds based on distances
    for i in range(len(atom_types)):
        for j in range(i+1, len(atom_types)):
            dist = np.linalg.norm(positions[i] - positions[j])
            # Set distance thresholds for single/double bonds (example values)
            if dist < 1.5:  # Threshold for C-N single bond
                mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.SINGLE)
            elif dist < 1.35:  # Threshold for C=N double bond
                mol.AddBond(atom_map[i], atom_map[j], Chem.BondType.DOUBLE)

    Chem.SanitizeMol(mol)
    return mol


class FunctionalGroupCounter:
    def __init__(self, functional_groups_smarts=None):
        """Initializes the functional group counter with SMARTS patterns."""
        self.compiled_smarts = functional_groups_smarts
        # self.compiled_smarts = {name: Chem.MolFromSmarts(smarts) for name, smarts in self.functional_groups_smarts.items()}

    def count_from_molecule(self, mol):
        """Counts functional groups in a given molecule."""
        counts = {group_name: 0 for group_name in self.compiled_smarts.keys()}
        for group_name, pattern in self.compiled_smarts.items():
            if mol.HasSubstructMatch(pattern):
                counts[group_name] += 1
        return counts


def process_generated_samples_digress(hydrogen=False):

    if hydrogen is True:
        path = 'DiGress/generated_samples/final_smiles_qm9withH.txt'
    else:
        path = 'DiGress/generated_samples/final_smiles_qm9_noH.txt'

    dataset = MolecularDataset()
    with open(path) as file:
        for line in file:
            if line == 'None\n':
                continue
            mol = Chem.MolFromSmiles(line)
            dataset.add(mol)
    return dataset.get_summary()


class MolecularDataset:
    def __init__(self):
        self.properties = {
            'mol_wts': [],
            'heavy_atom_counts': [],
            'functional_group_counts': Counter(),
        }
        self.fg_counter = FunctionalGroupCounter(functional_groups_smarts)

    def add(self, mol):
        """Adds a molecule to the dataset and updates properties."""
        mol_weight = Descriptors.MolWt(mol)
        self.properties['mol_wts'].append(mol_weight)
        
        heavy_atom_count = mol.GetNumHeavyAtoms()
        self.properties['heavy_atom_counts'].append(heavy_atom_count)
        
        # Count and update functional groups
        func_groups = self.fg_counter.count_from_molecule(mol)
        for group, count in func_groups.items():
            self.properties['functional_group_counts'][group] += count

    def get_summary(self):
        """Returns a summary of the molecular dataset properties."""
        summary = {
            'average_mol_weight': sum(self.properties['mol_wts']) / len(self.properties['mol_wts']) if self.properties['mol_wts'] else 0,
            'average_heavy_atom_count': sum(self.properties['heavy_atom_counts']) / len(self.properties['heavy_atom_counts']) if self.properties['heavy_atom_counts'] else 0,
            'functional_group_counts': dict(self.properties['functional_group_counts']),
        }
        return summary

# Main comparison function with saving enabled
def compare_qm9_generated():

    properties_qm9 = process_qm9_dataset()
    properties_edm = process_generated_samples_edm()
    properties_digress_noH = process_generated_samples_digress(hydrogen=False)
    properties_digress_withH = process_generated_samples_digress(hydrogen=True)

    # Compare and save molecular weight histogram
    save_dual_axis_histogram(
        properties_edm['mol_wts'], properties_qm9['mol_wts'],
        'Generated Samples', 'QM9 Dataset',
        'Molecular Weight Comparison', 'Molecular Weight',
        'molecular_weight_comparison.png'
    )

    # Compare and save heavy atom count histogram
    save_dual_axis_histogram(
        properties_edm['heavy_atom_counts'], properties_qm9['heavy_atom_counts'],
        'Generated Samples', 'QM9 Dataset',
        'Heavy Atom Count Comparison', 'Number of Heavy Atoms',
        'heavy_atom_count_comparison.png'
    )

    # for functional_group in list(qm9_properties['functional_group_counts']):
    for functional_group in list(properties_qm9['functional_group_counts']):
        if properties_edm['functional_group_counts'][functional_group] == 0 and properties_qm9['functional_group_counts'][functional_group] == 0 and properties_digress_withH['functional_group_counts'][functional_group] == 0:
            print('Deleting zero occurence:', functional_group)
            del properties_edm['functional_group_counts'][functional_group]
            del properties_qm9['functional_group_counts'][functional_group]
            del properties_digress_noH['functional_group_counts'][functional_group]
            del properties_digress_withH['functional_group_counts'][functional_group]

    # Compare and save functional group counts
    save_functional_group_comparison(
        properties_qm9['functional_group_counts'],
        properties_digress_noH['functional_group_counts'],
        properties_digress_withH['functional_group_counts'],
        properties_edm['functional_group_counts'],
        'functional_group_comparison.png'
    )

    save_functional_group_scatter(
        properties_edm['functional_group_counts'], properties_qm9['functional_group_counts'],
        'functional_group_scatter.png'
    )



def pos_to_png(positions, atomic_numbers, filename):
    """
    Converts atomic positions and atomic numbers to a 3D molecular PNG image.

    Args:
        positions (np.ndarray): Array of shape (n_atoms, 3) representing the 3D coordinates of each atom.
        atomic_numbers (list or np.ndarray): Array of atomic numbers (e.g., 1 for H, 6 for C) corresponding to each atom.
        filename (str): The path to save the PNG image.
    """
    # Create a color map for different elements
    color_map = {1: 'white', 6: 'black', 7: 'blue', 8: 'red', 16: 'yellow'}  # Add more elements as needed
    atom_colors = [color_map.get(z, 'gray') for z in atomic_numbers]  # Default color is gray

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot each atom
    for pos, color in zip(positions, atom_colors):
        ax.scatter(pos[0], pos[1], pos[2], color=color, s=100)  # Adjust `s` for atom size

    # Add lines for bonds if needed
    # This can be done using a distance threshold or edge_index if bond information is available
    # Example (uncomment and replace `bond_indices` with actual indices if available):
    # for i, j in bond_indices:
    #     ax.plot([positions[i][0], positions[j][0]],
    #             [positions[i][1], positions[j][1]],
    #             [positions[i][2], positions[j][2]], color='gray')

    # Set plot limits and style
    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()  # Remove axes for a cleaner image

    # Save to file
    plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    plt.close()

def mol_to_png(mol, filename):
    """Converts an RDKit Mol object to a PNG image and saves it."""
    if not os.path.isfile(filename):
        img = Draw.MolToImage(mol, size=(300, 300))  # Set desired size
        img.save(filename)



# Execute the comparison
if __name__ == "__main__":

    compare_qm9_generated()

    if False:
        cnt3 = 0
        dataset = QM9(root='data/QM9')
        for data in dataset:
            mol_babel = pybel.readstring("smi", data.smiles)
            mol_rdkit = Chem.MolFromSmiles(data.smiles)
            # if mol_babel is None and mol_rdkit is not None:
            if mol_rdkit is None:
                cnt3 += 1
                continue

        cnt1 = 0
        with open('DiGress/generated_samples/final_smiles_qm9withH.txt') as file:
        # with open('DiGress/generated_samples/final_smiles_qm9_noH.txt') as file:
            for line in file:
                if line == 'None\n':
                    cnt1 += 1
                    continue
                mol = Chem.MolFromSmiles(line)
                if mol is None:
                    cnt1 += 1
                    print('tmp1', line)
                    stop1

        cnt2 = 0
        for i in range(10000):
            xyz_file = f"e3_diffusion_for_molecules/generated_samples/molecule_{i:03}.txt"
            mol = next(pybel.readfile("xyz", xyz_file))
            smiles = mol.write('smi')
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                cnt2 += 1
                print('tmp2', smiles)

        print(cnt1, cnt2, cnt3)
        stop2


            



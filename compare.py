import os
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import QM9
from openbabel import pybel
from rdkit import Chem
from rdkit.Chem import Descriptors
from collections import Counter


# Define additional functional groups
functional_groups_smarts = {
    'Alcohol': Chem.MolFromSmarts('[OX2H]'),
    'Aldehyde': Chem.MolFromSmarts('[CX3H1](=O)[#6]'),
    'Ketone': Chem.MolFromSmarts('[CX3](=O)[#6]'),
    'Amine': Chem.MolFromSmarts('[$([NX3H2]),$([NX3H1][#6]),$([NX3]([#6])[#6])]'),
    'Ether': Chem.MolFromSmarts('COC'),
    'Phenyl': Chem.MolFromSmarts('c1ccccc1'),
    'Carboxyl': Chem.MolFromSmarts('C(=O)O'),
    'Ester': Chem.MolFromSmarts('C(=O)O[#6]'),
    'Amide': Chem.MolFromSmarts('C(=O)N'),
    'Alkyne': Chem.MolFromSmarts('C#C'),
    'Alkene': Chem.MolFromSmarts('C=C'),
    'Halide': Chem.MolFromSmarts('[F,Cl,Br,I]'),

    # Additional functional groups from Wikipedia list
    'Imine': Chem.MolFromSmarts('C=N'),  # Imine group
    'Nitrile': Chem.MolFromSmarts('C#N'),  # Nitrile group (-C≡N)
    'Nitro': Chem.MolFromSmarts('[$([NX3](=O)=O)]'),  # Nitro group (R-NO2)
    'Isocyanate': Chem.MolFromSmarts('N=C=O'),  # Isocyanate group (R-N=C=O)
    'Isothiocyanate': Chem.MolFromSmarts('N=C=S'),  # Isothiocyanate group (R-N=C=S)
    'Azide': Chem.MolFromSmarts('[N-]=[N+]=[N]'),  # Azide group (R-N3)
    'Dihydropyrimidine ring': Chem.MolFromSmarts('C1=CNC(=O)NC1'),  # Basic pattern for a 1,4-dihydropyrimidine

    # Three-membered rings
    'Borirene': Chem.MolFromSmarts('[B]1=CC1'),
    'Cyclopropenone': Chem.MolFromSmarts('C1=CC1=O'),

    # Five-membered rings
    'Furan': Chem.MolFromSmarts('c1ccoc1'),
    'Pyrrole': Chem.MolFromSmarts('c1cc[nH]c1'),
    'Imidazole': Chem.MolFromSmarts('c1cncn1'),
    'Thiophene': Chem.MolFromSmarts('c1ccsc1'),
    'Phosphole': Chem.MolFromSmarts('[P]1=CC=CC1'),
    'Pyrazole': Chem.MolFromSmarts('c1cnnc1'),
    'Oxazole': Chem.MolFromSmarts('c1cocn1'),
    'Isoxazole': Chem.MolFromSmarts('c1cno[nH]1'),
    'Thiazole': Chem.MolFromSmarts('c1cscn1'),
    'Isothiazole': Chem.MolFromSmarts('c1cnsn1'),
    'Triazole': Chem.MolFromSmarts('c1nncn1'),
    'Tetrazole': Chem.MolFromSmarts('c1nnnn1'),
    'Pentazole': Chem.MolFromSmarts('c1nnnnn1'),

    # Six-membered rings
    'Benzene': Chem.MolFromSmarts('c1ccccc1'),
    'Pyridine': Chem.MolFromSmarts('c1ccncc1'),
    'Pyrazine': Chem.MolFromSmarts('c1cnccn1'),
    'Pyrimidine': Chem.MolFromSmarts('c1cncnc1'),
    'Pyridazine': Chem.MolFromSmarts('c1cncnn1'),
    'Triazine': Chem.MolFromSmarts('c1ncncn1'),
    'Tetrazine': Chem.MolFromSmarts('c1nnnnc1'),
    'Pentazine': Chem.MolFromSmarts('c1nnnnn1'),
    'Hexazine': Chem.MolFromSmarts('c1nnnnnn1'),
}


# Convert XYZ to SMILES and InChI, skipping invalid files
def xyz_to_smiles_inchi(file_path):
    try:
        mol = next(pybel.readfile("xyz", file_path))
        smiles = mol.write("smi").strip()
        inchi = mol.write("inchi").strip()
        return smiles, inchi
    except Exception as e:
        print(f"Skipping invalid file {file_path}: {e}")
        return None, None

# Calculate properties and skip invalid SMILES
def calculate_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Skipping invalid SMILES: {smiles}")
        return None, None
    mol_wt = Descriptors.MolWt(mol)
    heavy_atom_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1)
    return mol_wt, heavy_atom_count


# Calculate functional groups
def count_functional_groups(smiles, functional_groups_smarts):
    mol = Chem.MolFromSmiles(smiles)
    functional_groups = {group_name: 0 for group_name in functional_groups_smarts.keys()}
    for group_name, smarts in functional_groups_smarts.items():
        if mol.HasSubstructMatch(smarts):
            functional_groups[group_name] += 1
    return functional_groups


# Process generated samples with validation
def process_generated_samples():
    generated_properties = {
        'mol_wts': [],
        'heavy_atom_counts': [],
        'functional_groups': Counter(),
    }

    for i in range(10000):
        print(i)
        xyz_file = f"e3_diffusion_for_molecules/generated_samples/molecule_{i}.txt"
        smiles, inchi = xyz_to_smiles_inchi(xyz_file)

        if smiles and inchi:  # Only process valid molecules
            mol_wt, heavy_atom_count = calculate_properties(smiles)
            if mol_wt is not None:  # Ensure properties calculation succeeded
                generated_properties['mol_wts'].append(mol_wt)
                generated_properties['heavy_atom_counts'].append(heavy_atom_count)

                func_groups = count_functional_groups(smiles, functional_groups_smarts)
                for group_name, count in func_groups.items():
                    generated_properties['functional_groups'][group_name] += count

    return generated_properties


# Load QM9 dataset
def process_qm9_dataset():
    dataset = QM9(root='data/QM9')
    qm9_properties = {
        'mol_wts': [],
        'heavy_atom_counts': [],
        'functional_groups': Counter(),
    }

    for functional_group in functional_groups_smarts.keys():
        qm9_properties['functional_groups'][functional_group] = 0

    for data in dataset:
        mol = Chem.MolFromSmiles(data.smiles)
        if mol is not None:
            mol_wt, heavy_atom_count = calculate_properties(data.smiles)
            qm9_properties['mol_wts'].append(mol_wt)
            qm9_properties['heavy_atom_counts'].append(heavy_atom_count)
            
            func_groups = count_functional_groups(data.smiles, functional_groups_smarts)
            for group_name, count in func_groups.items():
                qm9_properties['functional_groups'][group_name] += count

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
def save_functional_group_comparison(generated_func, qm9_func, filename):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    labels = list(generated_func.keys())
    generated_counts = np.array([generated_func[group] for group in labels])
    qm9_counts = np.array([qm9_func[group] for group in labels])
    
    # Convert counts to percentages for each dataset
    generated_percentages = (generated_counts / generated_counts.sum()) * 100
    qm9_percentages = (qm9_counts / qm9_counts.sum()) * 100
    
    x = np.arange(len(labels))
    width = 0.35

    # Plot generated dataset percentages
    ax1.bar(x - width/2, generated_percentages, width, color='green', label='Generated %')
    
    # Plot QM9 dataset percentages
    ax1.bar(x + width/2, qm9_percentages, width, color='blue', label='QM9 %')

    # Labeling and aesthetics
    ax1.set_xlabel('Functional Group')
    ax1.set_ylabel('Percentage (%)')
    ax1.set_title('Functional Group Percentages in Generated vs QM9')
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
    

# Main comparison function with saving enabled
def compare_qm9_generated():
    # Process both datasets
    generated_properties = process_generated_samples()
    qm9_properties = process_qm9_dataset()

    # Compare and save molecular weight histogram
    save_dual_axis_histogram(
        generated_properties['mol_wts'], qm9_properties['mol_wts'],
        'Generated Samples', 'QM9 Dataset',
        'Molecular Weight Comparison', 'Molecular Weight',
        'molecular_weight_comparison.png'
    )

    # Compare and save heavy atom count histogram
    save_dual_axis_histogram(
        generated_properties['heavy_atom_counts'], qm9_properties['heavy_atom_counts'],
        'Generated Samples', 'QM9 Dataset',
        'Heavy Atom Count Comparison', 'Number of Heavy Atoms',
        'heavy_atom_count_comparison.png'
    )

    for functional_group in list(generated_properties['functional_groups']):
        if generated_properties['functional_groups'][functional_group] == 0 and qm9_properties['functional_groups'][functional_group] == 0:
            print('Deleting zero occurence:', functional_group)
            del generated_properties['functional_groups'][functional_group]
            del qm9_properties['functional_groups'][functional_group]

    # Compare and save functional group counts
    save_functional_group_comparison(
        generated_properties['functional_groups'], qm9_properties['functional_groups'],
        'functional_group_comparison.png'
    )

    save_functional_group_scatter(
        generated_properties['functional_groups'], qm9_properties['functional_groups'],
        'functional_group_scatter.png'
    )

# Execute the comparison
if __name__ == "__main__":
    compare_qm9_generated()


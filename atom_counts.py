import re
from collections import Counter
import matplotlib.pyplot as plt

# Function to count atoms directly from SMILES strings
def count_atoms_from_smiles(smiles):
    atom_counts = Counter()
    # Regex patterns for each atom type (ensures only full atoms are counted)
    patterns = {
        'C': r'\bC\b',
        'N': r'\bN\b',
        'O': r'\bO\b',
        'H': r'\bH\b',
        'F': r'\bF\b'
    }
    for atom, pattern in patterns.items():
        atom_counts[atom] = len(re.findall(pattern, smiles))
    return atom_counts

# Counting atoms in QM9 dataset
def count_atoms_qm9():
    from torch_geometric.datasets import QM9
    dataset = QM9(root='data/QM9')
    atom_counts = Counter()

    for data in dataset:
        atom_counts.update(count_atoms_from_smiles(data.smiles))
    
    return atom_counts

# Counting atoms in DiGress dataset (with hydrogens)
def count_atoms_digress(file_path):
    atom_counts = Counter()
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if line == 'None' or not line:
                continue
            atom_counts.update(count_atoms_from_smiles(line))
    
    return atom_counts

# Counting atoms in EDM dataset from XYZ files (convert to SMILES with Pybel)
def count_atoms_edm(num_files=10000):
    from openbabel import pybel
    atom_counts = Counter()
    for i in range(num_files):
        xyz_file = f"e3_diffusion_for_molecules/generated_samples/molecule_{i:03}.txt"
        with open(xyz_file) as file:
            file.readline()
            file.readline()
            for line in file:
                atom = line.split(' ', 1)[0]
                atom_counts[atom] += 1
    
    return atom_counts

# Running the counts for each dataset
print('qm9')
qm9_atom_counts = count_atoms_qm9()
print('digress')
digress_atom_counts = count_atoms_digress('DiGress/generated_samples/final_smiles_qm9withH.txt')
print('edm')
edm_atom_counts = count_atoms_edm()

# Combine results into a dictionary for plotting
datasets = ['QM9', 'DiGress', 'EDM']

atom_types = qm9_atom_counts.keys()

atom_data = {
    atom: [qm9_atom_counts[atom], digress_atom_counts[atom], edm_atom_counts[atom]]
    for atom in atom_types
}

total_atoms_qm9 = sum(qm9_atom_counts.values())
total_atoms_digress = sum(digress_atom_counts.values())
total_atoms_edm = sum(edm_atom_counts.values())

atom_data_percentage = {
    atom: [
        qm9_atom_counts[atom] / total_atoms_qm9 * 100,
        digress_atom_counts[atom] / total_atoms_digress * 100,
        edm_atom_counts[atom] / total_atoms_edm * 100
    ]
    for atom in atom_types
}

# Plotting atom percentages with atom types as main categories and datasets grouped within
fig, ax = plt.subplots(figsize=(12, 6))

# Bar plot for each dataset grouped within each atom type
width = 0.25  # Width of each bar within an atom type category
x = range(len(atom_data_percentage))  # Positions for each atom type

# Extracting dataset labels for legend
dataset_labels = ['QM9', 'DiGress', 'EDM']

# Iterate over each dataset and plot for each atom type
for j, dataset_label in enumerate(dataset_labels):
    ax.bar(
        [p + width * j for p in x],  # Offset each dataset within each atom type category
        [percentage[j] for percentage in atom_data_percentage.values()],  # Data for each dataset
        width,
        label=dataset_label
    )

# Adding labels and legend
ax.set_xlabel('Atom Types')
ax.set_ylabel('Atom %')
ax.set_title('Atom % for Each Type in QM9, DiGress, and EDM Datasets')
ax.set_xticks([p + width for p in x])  # Center the x-tick labels
ax.set_xticklabels(atom_data_percentage.keys())  # Atom types as main categories
ax.legend(title="Datasets")

plt.tight_layout()
plt.savefig('atom_counts_grouped_by_atom.png')

import os


CURDIR = os.path.dirname(os.path.abspath(__file__))
DATADIR = os.path.join(CURDIR, 'data')

name = '1ps3'
ligand_path = os.path.join(DATADIR, name, name + '_ligand.mol2')
generate_ligand_path = os.path.join(DATADIR, name, name + '_generate_ligand.pdb')
protein_path = os.path.join(DATADIR, name, name + '_pocket.pdb')
box_path = os.path.join(DATADIR, name, 'box.txt')
import os
import shutil
import tempfile

import numpy as np
import dgl
from dgl.data.utils import load_graphs
from torch.utils.data import Dataset

from openbabel import pybel
from joblib import Parallel, delayed

from .mol2graph_rdmda_ob_res_large import mol_to_graph, prot_to_graph
from .extract_pocket_prody_ob import extract_pocket_with_ligand
from .protein import from_pdb_path
from .all_atom import atom37_to_torsion_angles


class VSDataset(Dataset):
    def __init__(self,
                 ids=None,
                 ligs=None,
                 prot=None,
                 gen_pocket=False,
                 cutoff=None,
                 reflig=None,
                 explicit_H=False,
                 use_chirality=True,
                 parallel=True
                 ):
        self.graphp = None
        self.graphsl = None
        self.pocketdir = None
        self.prot = None
        self.ligs = None
        self.ligswords = None
        self.cutoff = cutoff
        self.explicit_H = explicit_H
        self.use_chirality = use_chirality
        self.parallel = parallel

        if isinstance(prot, pybel.Molecule):
            assert !gen_pocket
            self.prot = prot
            self.graphp = prot_to_graph(self.prot, cutoff)
        else:
            if gen_pocket:
                if cutoff is None or reflig is None:
                    raise ValueError('If you want to generate the pocket, the cutoff and the reflig should be given')
                try:
                    self.pocketdir = tempfile.mkdtemp()
                    if reflig.endswith(".mol2"):
                        lig_blocks = self._mol2_split(ligs)
                        if len(lig_blocks) > 0:
                            reflig_mol = pybel.readstring("mol2", lig_blocks[0])
                            reflig_mol.removeh()
                            reflig_mol.write("pdb", os.path.join(self.pocketdir, "reflig.pdb"))
                            reflig = os.path.join(self.pocketdir, "reflig.pdb")
                        else:
                            reflig_mol = next(pybel.readfile("mol2", reflig))
                            reflig_mol.removeh()
                            reflig_mol.write("pdb", os.path.join(self.pocketdir, "reflig.pdb"))
                            reflig = os.path.join(self.pocketdir, "reflig.pdb")
                    elif reflig.endswith(".sdf"):
                        lig_blocks = self._sdf_split(ligs)
                        if len(lig_blocks) > 0:
                            reflig_mol = pybel.readstring("sdf", lig_blocks[0])
                            reflig_mol.removeh()
                            reflig_mol.write("pdb", os.path.join(self.pocketdir, "reflig.pdb"))
                            reflig = os.path.join(self.pocketdir, "reflig.pdb")
                        else:
                            reflig_mol = next(pybel.readfile("sdf", reflig))
                            reflig_mol.removeh()
                            reflig_mol.write("pdb", os.path.join(self.pocketdir, "reflig.pdb"))
                            reflig = os.path.join(self.pocketdir, "reflig.pdb")

                    extract_pocket_with_ligand(prot, reflig, cutoff,
                                   protname="temp",
                                   workdir=self.pocketdir)

                    pocket_path = "%s/temp_pocket_%s.pdb" % (self.pocketdir, cutoff)
                    prot = from_pdb_path(pocket_path)
                    prot_torsions_feats = atom37_to_torsion_angles(prot.aatype[np.newaxis, :],
                                                                   prot.atom_positions[np.newaxis, :],
                                                                   prot.atom_mask[np.newaxis, :])

                    prot_torsion_sin_cos = np.array(prot_torsions_feats['torsion_angles_sin_cos'])
                    torsions = np.angle(prot_torsion_sin_cos[:, :, :, 1] + 1j*prot_torsion_sin_cos[:, :, :, 0]).reshape(-1, 7)
                    self.graphp = prot_to_graph(pocket_path, cutoff,
                                                prot_torsions_sin_cos=torsions,
                                                rigidgroups_gt_frames=None,
                                                protein=prot)
                except Exception as e:
                    print(e)
                    raise ValueError('The graph of pocket cannot be generated')
            else:
                try:
                    pocket_path = prot
                    prot = from_pdb_path(pocket_path)
                    prot_torsions_feats = atom37_to_torsion_angles(prot.aatype[np.newaxis, :],
                                                                   prot.atom_positions[np.newaxis, :],
                                                                   prot.atom_mask[np.newaxis, :])

                    prot_torsion_sin_cos = np.array(prot_torsions_feats['torsion_angles_sin_cos'])
                    torsions = np.angle(prot_torsion_sin_cos[:, :, :, 1] + 1j*prot_torsion_sin_cos[:, :, :, 0]).reshape(-1, 7)
                    self.graphp = prot_to_graph(pocket_path, cutoff,
                                                prot_torsions_sin_cos=torsions,
                                                rigidgroups_gt_frames=None,
                                                protein=prot)
                except Exception as e:
                    import traceback
                    traceback.print_exc()

        if isinstance(ligs, np.ndarray) or isinstance(ligs, list):
            if isinstance(ligs[0], pybel.Molecule):
                self.ligs = ligs
                self.graphsl = self._mol_to_graph()
            elif isinstance(ligs[0], dgl.DGLGraph):
                self.graphsl = ligs
            else:
                raise ValueError('Ligands should be a list of rdkit.Chem.rdchem.Mol objects')
        else:
            if ligs.endswith(".mol2"):
                lig_blocks = self._mol2_split(ligs)
                decoys_ligs = []
                for lig_block in lig_blocks:
                    oblig = pybel.readstring("mol2", lig_block)
                    oblig.removeh()
                    decoys_ligs.append(oblig)
                    break
                self.ligs = decoys_ligs
                self.graphsl = self._mol_to_graph()
            elif ligs.endswith(".sdf"):
                lig_blocks = self._sdf_split(ligs)
                decoys_ligs = []
                for lig_block in lig_blocks:
                    oblig = pybel.readstring("sdf", lig_block)
                    oblig.removeh()
                    decoys_ligs.append(oblig)
                    break
                self.ligs = decoys_ligs
                self.graphsl = self._mol_to_graph()
            elif ligs.endswith(".pdb"):
                oblig = next(pybel.readfile("pdb", ligs))
                oblig.removeh()
                self.ligs = [oblig]
                self.graphsl = self._mol_to_graph()
            else:
                try:
                    self.graphsl, _ = load_graphs(ligs)
                except:
                    raise ValueError(
                        'Only the ligands with .sdf or .mol2 or a file to genrate DGLGraphs will be supported')

        if ids is None:
            if self.ligs is not None:
                self.idsx = ["%s" % (self.get_ligname(lig)) for i, lig in enumerate(self.ligs)]
            else:
                self.idsx = ["lig%s" % i for i in range(len(self.graphsl))]
        else:
            self.idsx = ids

        try:
            self.ids, self.graphsl = zip(*filter(lambda x: x[1] != None, zip(self.idsx, self.graphsl)))
        except:
            self.ids, self.graphsl = [], []
        self.ids = list(self.ids)
        self.graphsl = list(self.graphsl)
        assert len(self.ids) == len(self.graphsl)
        if self.pocketdir is not None:
            shutil.rmtree(self.pocketdir)

    def __getitem__(self, idx):
        return self.ids[idx], self.graphsl[idx], self.graphp

    def __len__(self):
        return len(self.ids)

    def _mol2_split(self, infile):
        contents = open(infile, 'r').read()
        return ["@<TRIPOS>MOLECULE\n" + c for c in contents.split("@<TRIPOS>MOLECULE\n")[1:]]

    def _sdf_split(self, infile):
        contents = open(infile, 'r').read()
        return [c + "$$$$\n" for c in contents.split("$$$$\n")[:-1]]

    def _mol_to_graph0(self, lig):
        try:
            gx = mol_to_graph(
                lig,
                explicit_H=self.explicit_H, use_chirality=self.use_chirality)
        except:
            print("failed to scoring for {} and {}".format(self.graphp, lig))
            return None
        return gx

    def _mol_to_graph(self):
        if self.parallel:
            return Parallel(n_jobs=-1, backend="threading")(
                delayed(self._mol_to_graph0)(self.ligs[mmm])
                for mmm in range(len(self.ligs)))
        else:
            graphs = []
            for mmm in range(len(self.ligs)):
                graphs.append(self._mol_to_graph0(self.ligs[mmm]))
            return graphs

    def get_ligname(self, m):
        if m is None:
            return None
        else:
            return m.title

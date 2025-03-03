# This file modified from https://github.com/sc8668/RTMScore
# MIT License Copyright (c) 2023 sc8668

import os
import re

import prody as pr
from openbabel import openbabel as ob
from openbabel import pybel

from deepmice.tools import ob_read_file

from .pdb_selaltloc import remove_altloc


ob_log_handler = pybel.ob.OBMessageHandler()
ob_log_handler.SetOutputLevel(0)
pybel.ob.cvar.obErrorLog.SetOutputLevel(0)


def write_file(output_file, outline):
    buffer = open(output_file, 'w')
    buffer.write(outline)
    buffer.close()


def lig_rename(infile, outfile):
    lines = open(infile, 'r').readlines()
    newlines = []
    for line in lines:
        if re.search(r'^HETATM|^ATOM', line):
            newlines.append(line[:17] + "LIG" + line[20:])
        else:
            newlines.append(line)
    write_file(outfile, ''.join(newlines))


def check_mol(infile, outfile):
    os.system("cat %s | sed '/LIG/d' > %s" % (infile, outfile))


def extract_pocket_with_ligand(protpath,
                               ligpath,
                               cutoff=10.0,
                               workdir='.'):
    protname = 'PROT'
    ligname = 'LIG'

    wk_ligand_path = os.path.join(workdir, '{}.pdb'.format(ligname))
    wk_rename_ligand_path = os.path.join(workdir, '{}_rename.pdb'.format(ligname))

    ligand = ob_read_file(ligpath)
    ligand.write('pdb', wk_ligand_path)

    xprot = pr.parsePDB(protpath)
    lig_rename(wk_ligand_path, wk_rename_ligand_path)
    os.remove(wk_ligand_path)
    os.rename(wk_rename_ligand_path, wk_ligand_path)
    xlig = pr.parsePDB(wk_ligand_path)
    lresname = xlig.getResnames()[0]
    xcom = xlig + xprot

    pocket_tmp_path = os.path.join(workdir,
                                   '{}_pocket_{}_temp.pdb'.format(protname, cutoff))
    pocket_path = os.path.join(workdir,
                               '{}_pocket_{}.pdb'.format(protname, cutoff))

    ret = xcom.select('same residue as exwithin %s of resname %s' % (cutoff, lresname))

    pr.writePDB(pocket_tmp_path, ret)

    check_mol(pocket_tmp_path, pocket_path)
    os.remove(pocket_tmp_path)

    obConversion2 = ob.OBConversion()
    obConversion2.SetInAndOutFormats("pdb", "pdb")
    pocket = ob.OBMol()
    obConversion2.ReadFile(pocket, pocket_path)
    pocket.DeleteHydrogens()
    obConversion2.WriteFile(pocket, pocket_path)

    outstr = remove_altloc(pocket_path)
    with open(pocket_path, 'w') as f:
        f.write(outstr)
    return pocket_path


def extract_pocket_with_box(protpath, box, workdir='.'):
    from deepmice.tools import cut_protein_with_box

    protname = os.path.basename(protpath).split('.')[0]
    cutoff = 10
    tmp_pdb_path = "%s/%s_pocket_%s_temp.pdb" % (workdir, protname, cutoff)
    res_pdb_path = "%s/%s_pocket_%s.pdb" % (workdir, protname, cutoff)
    cut_protein_with_box(protpath, box, tmp_pdb_path)

    check_mol(tmp_pdb_path, res_pdb_path)
    os.remove(tmp_pdb_path)
    obConversion2 = ob.OBConversion()
    obConversion2.SetInAndOutFormats("pdb", "pdb")
    pocket = ob.OBMol()
    obConversion2.ReadFile(pocket, res_pdb_path)
    pocket.DeleteHydrogens()
    obConversion2.WriteFile(pocket, res_pdb_path)
    outstr = remove_altloc(res_pdb_path)
    with open(res_pdb_path, 'w') as f:
        f.write(outstr)
    return res_pdb_path

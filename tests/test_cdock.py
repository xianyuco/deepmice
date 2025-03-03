import os
import shutil

import pytest

from deepmice.cdock import (
    MLPCDock,
    MLPCScreen,
    MDNV2CScreen
)
from deepmice.models.mlp.score import MLPScore
from deepmice.tools import ob_calc_rmsd

from data_info import ligand_path, generate_ligand_path, protein_path, box_path


with open(box_path) as f:
    box = [float(l.strip()) for l in f.readlines() if l.strip() != 0]
mlpcdock_save_dir = '/tmp/deepmice_test_tmpfile/mlpcdock'
mlpcdock_save_path = os.path.join(mlpcdock_save_dir, 'result.pdb')
if os.path.exists(mlpcdock_save_path):
    os.remove(mlpcdock_save_path)
mlpcscreen_save_dir = '/tmp/deepmice_test_tmpfile/mlpcscreen'
mlpcscreen_save_path = os.path.join(mlpcscreen_save_dir, 'result.pdb')
if os.path.exists(mlpcscreen_save_path):
    os.remove(mlpcscreen_save_path)
mdnv2cscreen_save_dir = '/tmp/deepmice_test_tmpfile/mdnv2cscreen'
mdnv2cscreen_save_path = os.path.join(mdnv2cscreen_save_dir, 'result.pdb')
if os.path.exists(mdnv2cscreen_save_path):
    os.remove(mdnv2cscreen_save_path)


def test_mlpcdock():
    if os.path.exists(mlpcdock_save_dir):
        shutil.rmtree(mlpcdock_save_dir)
    os.makedirs(mlpcdock_save_dir)
    dock = MLPCDock(generate_ligand_path, protein_path, box)
    docked_score = dock.run(mlpcdock_save_path)
    docked_score = docked_score[0]
    score_model = MLPScore()
    rescore = score_model.run(
        mlpcdock_save_path, protein_path,
        include_auxiliary_score=True,
        include_ligand_covalent_bond_score=False,
        include_ligand_noncovalent_bond_score=True
    )

    assert(abs(docked_score - (rescore + score_model.bias)) < 0.3)
    assert os.path.exists(mlpcdock_save_path)

    result_rmsd = ob_calc_rmsd(ligand_path, mlpcdock_save_path)
    assert result_rmsd <= 4
    assert result_rmsd <= 2


@pytest.mark.xfail(raises=AssertionError, reason='MLPCDock对接结果RMSD值小于2埃')
def test_mlpcdock_result():
    if not os.path.exists(mlpcdock_save_path):
        return
    result_rmsd = ob_calc_rmsd(ligand_path, mlpcdock_save_path)
    assert result_rmsd <= 2


def test_mlpcscreen():
    if os.path.exists(mlpcscreen_save_dir):
        shutil.rmtree(mlpcscreen_save_dir)
    os.makedirs(mlpcscreen_save_dir)
    screen = MLPCScreen(protein_path, box)
    docked_score = screen.run(generate_ligand_path, mlpcscreen_save_path)
    docked_score = docked_score[0]

    score_model = MLPScore()
    rescore = score_model.run(
        mlpcscreen_save_path, protein_path,
        include_auxiliary_score=True,
        include_ligand_covalent_bond_score=False,
        include_ligand_noncovalent_bond_score=True
    )
    assert(abs(docked_score - (rescore + score_model.bias)) < 0.3)
    assert os.path.exists(mlpcscreen_save_path)

    result_rmsd = ob_calc_rmsd(ligand_path, mlpcscreen_save_path)
    assert result_rmsd <= 4
    assert result_rmsd <= 2


@pytest.mark.xfail(raises=AssertionError, reason='MLPCScreen对接结果RMSD值小于2埃')
def test_mlpcscreen_result():
    if not os.path.exists(mlpcscreen_save_path):
        return
    result_rmsd = ob_calc_rmsd(ligand_path, mlpcscreen_save_path)
    assert result_rmsd <= 2


def test_mdnv2cscreen():
    if os.path.exists(mdnv2cscreen_save_dir):
        shutil.rmtree(mdnv2cscreen_save_dir)
    os.makedirs(mdnv2cscreen_save_dir)
    screen = MDNV2CScreen(protein_path, box)
    docked_score = screen.run(generate_ligand_path, mdnv2cscreen_save_path)
    assert os.path.exists(mdnv2cscreen_save_path)

    result_rmsd = ob_calc_rmsd(ligand_path, mdnv2cscreen_save_path)
    assert result_rmsd <= 4
    assert result_rmsd <= 2


@pytest.mark.xfail(raises=AssertionError, reason='MDNV2CScreen对接结果RMSD值小于2埃')
def test_mdnv2cscreen_result():
    if not os.path.exists(mdnv2cscreen_save_path):
        return
    result_rmsd = ob_calc_rmsd(ligand_path, mdnv2cscreen_save_path)
    assert result_rmsd <= 2

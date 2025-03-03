import numpy as np


class TreeShareInfo:

    def __init__(self) -> None:
        self.positions = None
        self.local_positions = None
        self.local_positions_offset = None
        self.rotated_positions = None

        self.is_increment = False
        self.rotate_move = None
        self.rotate_move_derivative = None

        self.num_freedegree = 0
        self.num_segment = 0


class LigandSegment:
    def __init__(self,
                 root_atom,
                 axis_begin_atom,
                 axis_end_atom,
                 free_degree_start_index,
                 is_root_segment=False):

        self.atoms = []
        self.rotate_atoms = []
        self.root_atom = root_atom
        self.axis_begin_atom = axis_begin_atom
        self.axis_end_atom = axis_end_atom
        self.axis = None
        self.parent = None
        self.children = []
        self.is_root_segment = is_root_segment
        self.segment_index = free_degree_start_index
        self.tree_info = None

    def set_tree_info(self, multiple_positions, num_segment):
        multiple_positions = np.array(multiple_positions)
        assert len(multiple_positions.shape) == 3
        assert self.is_root_segment is True
        self.tree_info = TreeShareInfo()
        self.tree_info.num_segment = num_segment
        self.tree_info.num_freedegree = num_segment - 1 + 6 + 1
        self.tree_info.positions = multiple_positions.copy()
        self.tree_info.local_positions = multiple_positions.copy()
        self.tree_info.rotated_positions = np.zeros_like(multiple_positions[0])
        self.tree_info.local_positions_offset = np.zeros([len(multiple_positions), num_segment, 3])
        self.tree_info.local_positions_offset[:, self.segment_index] = self.tree_info.positions[:, self.root_atom].copy()

    def add_atom(self, atom):
        self.atoms.append(atom)

        if self.is_root_segment:
            self.rotate_atoms.append(atom)
            self.tree_info.local_positions[:, atom] -= self.tree_info.local_positions_offset[:, self.segment_index]
        else:
            if atom != self.root_atom:
                self.rotate_atoms.append(atom)
                self.tree_info.local_positions[:, atom] -= self.tree_info.local_positions_offset[:, self.segment_index]
            else:
                self.tree_info.local_positions[:, atom] -= self.tree_info.local_positions_offset[:, self.parent.segment_index]

    def add_child(self, child_segment):
        child_segment.parent = self
        child_segment.tree_info = self.tree_info

        self.tree_info.local_positions_offset[:, child_segment.segment_index] =\
            self.tree_info.positions[:, child_segment.root_atom].copy()

        tmp_axis = (self.tree_info.positions[:, child_segment.axis_end_atom]
            - self.tree_info.positions[:, child_segment.axis_begin_atom])
        child_segment.axis = tmp_axis / np.linalg.norm(tmp_axis, axis=1)[:, np.newaxis]
        if child_segment.root_atom not in self.rotate_atoms:
            self.rotate_atoms.append(child_segment.root_atom)
        self.children.append(child_segment)

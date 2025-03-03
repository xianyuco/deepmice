import numpy as np
import torch


def generate_grid(box_size, grid_spacing=0.375):
    num_grid = np.ceil(box_size / grid_spacing) + 1
    x_num_grid, y_num_grid, z_num_grid = num_grid.astype(np.int64)

    grid_coord = np.mgrid[:x_num_grid,
                          :y_num_grid,
                          :z_num_grid]

    grid_coord = grid_coord.transpose(1, 2, 3, 0).astype(np.float32)

    return grid_coord


def generate_grid_0_distance(grid_coord, grid_spacing=0.375):
    x_num_grid, y_num_grid, z_num_grid, _ = grid_coord.shape
    zero = torch.tensor([[0, 0, 0]], dtype=torch.float32)
    g0_distance = torch.cdist(
        torch.from_numpy(grid_coord.reshape(-1, 3)), zero)
    g0_distance = g0_distance.view([x_num_grid, y_num_grid, z_num_grid])
    g0_distance = g0_distance * grid_spacing
    g0_distance[g0_distance == 0] = 1e-5
    return g0_distance.to(torch.float32).numpy()


def generate_grid_protein_distance(grid_coord, protein_positions, grid_spacing=0.375):
    x_num_grid, y_num_grid, z_num_grid, _ = grid_coord.shape
    protein_positions = protein_positions.astype(np.float32)

    grid = grid_coord * grid_spacing
    grid_flatten = grid.reshape([-1, 3])
    gp_distance = torch.cdist(
        torch.from_numpy(grid_flatten),
        torch.from_numpy(protein_positions))

    gp_distance = gp_distance.view(
        [x_num_grid, y_num_grid, z_num_grid, len(protein_positions)])
    return gp_distance.to(torch.float32).numpy()

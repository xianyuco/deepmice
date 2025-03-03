import numpy as np

from .quaternion import r3_angle_to_quaternion, quaternion_multiply, normalize_angle


def increment(x_increment, change, alpha):
    change = change.copy()
    x_increment = x_increment.copy()

    change *= alpha
    x_increment[: 3] += change[: 3]
    x_increment[3: 7] = quaternion_multiply(
        x_increment[3: 7], r3_angle_to_quaternion(change[3: 6]))
    x_increment[7:] += change[6:]
    x_increment[7:] = normalize_angle(x_increment[7:])

    return x_increment


class BFGS:

    def __init__(self, estimator, max_steps=20, max_line_search_steps=10) -> None:
        self.estimator = estimator
        self.max_steps = max_steps
        self.max_line_search_steps = max_line_search_steps

    def run(self, x_increment):

        hessian = np.eye(self.estimator.ligand_tree.tree_info.num_freedegree, dtype=np.float32)
        energy = self.estimator.calc_energy(x_increment)
        grad = self.estimator.derivative()

        best_x_increment = x_increment.copy()
        best_energy = energy.copy()

        last_x_increment = x_increment.copy()
        last_energy = energy.copy()
        last_grad = grad.copy()
        for step in range(self.max_steps):

            if np.dot(last_grad, last_grad) < 1e-10:
                break

            grad_direction = -hessian.dot(last_grad)

            tmp_x_increment, tmp_energy, tmp_grad, alpha = self.line_search(
                last_x_increment, last_grad, grad_direction, last_energy)

            if tmp_energy < best_energy:
                best_energy = tmp_energy
                best_x_increment = tmp_x_increment.copy()

            y = tmp_grad - last_grad
            if step == 0:
                yy = y.dot(y)
                if yy > 1e-6:
                    fill_value = alpha * np.dot(y, grad_direction) / yy
                    np.fill_diagonal(hessian, fill_value)
            hessian = self.bfgs_update(hessian, grad_direction, y, alpha)

            last_x_increment = tmp_x_increment
            last_grad = tmp_grad
            last_energy = tmp_energy

        return best_x_increment, best_energy

    def bfgs_update(self, hessian, changep, changey, alpha):
        yp = (changey * changep).sum()
        if alpha * yp < 1e-16:
            return hessian
        minus_hy = -hessian.dot(changey)
        yhy = (-changey * minus_hy).sum()
        r = 1 / (alpha * yp)

        mata = np.matmul(minus_hy.reshape(-1, 1), changep.reshape(1, -1))
        matb = mata.T
        matc = np.matmul(changep.reshape(-1, 1), changep.reshape(1, -1))
        hessian = hessian + alpha * r * (mata + matb) + \
            alpha * alpha * (r*r*yhy+r) * matc
        return hessian

    def line_search(self, x_increment, x_g, x_p, energy):
        c0 = 0.0001
        alpha = 1
        multiplier = 0.5
        pg = x_g.dot(x_p)
        for step in range(self.max_line_search_steps):
            tmp_x_increment = increment(x_increment, x_p, alpha)
            tmp_energy = self.estimator.calc_energy(tmp_x_increment)

            if tmp_energy - energy < c0 * alpha * pg:
                break
            alpha *= multiplier
        res_grad = self.estimator.derivative()
        return tmp_x_increment, tmp_energy, res_grad, alpha


class GD:

    def __init__(self, estimator, max_steps=20, alpha=1e-3, max_line_search_steps=0) -> None:
        self.estimator = estimator
        self.max_steps = max_steps
        self.max_line_search_steps = max_line_search_steps
        self.alpha = alpha

    def run(self, x_increment):

        energy = self.estimator.calc_energy(x_increment)

        best_x_increment = x_increment.copy()
        best_energy = energy.copy()
        if self.max_line_search_steps > 0:
            tmp_energy = self.estimator.calc_energy(x_increment)
            last_grad = self.estimator.derivative()
        for step in range(self.max_steps):
            if self.max_line_search_steps > 0:
                x_increment, tmp_energy, last_grad, alpha = self.line_search(
                    x_increment, last_grad, -last_grad, tmp_energy)
            else:
                tmp_energy = self.estimator.calc_energy(x_increment)
                last_grad = -self.estimator.derivative()
                x_increment = increment(x_increment, last_grad, self.alpha)

            if tmp_energy < best_energy:
                best_energy = tmp_energy
                best_x_increment = x_increment.copy()

        return best_x_increment, best_energy

    def line_search(self, x_increment, x_g, x_p, energy):
        c0 = 0.0001
        alpha = 1
        multiplier = 0.5
        pg = x_g.dot(x_p)
        for step in range(self.max_line_search_steps):
            tmp_x_increment = increment(x_increment, x_p, alpha)
            tmp_energy = self.estimator.calc_energy(tmp_x_increment)

            if tmp_energy - energy < c0 * alpha * pg:
                break
            alpha *= multiplier
        res_grad = self.estimator.derivative()
        return tmp_x_increment, tmp_energy, res_grad, alpha

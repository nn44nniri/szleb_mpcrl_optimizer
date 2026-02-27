from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class RLSConfig:
    lam: float = 0.99      # forgetting factor
    delta: float = 10.0    # initial covariance scale


class RLS:
    """
    Recursive Least Squares for a linear model:
        x_{k+1} = A x_k + B u_k + E z_k + b
    We stack features phi = [x, u, z, 1].
    """

    def __init__(self, nx: int, nu: int, nz: int, cfg: RLSConfig = RLSConfig()):
        self.nx, self.nu, self.nz = nx, nu, nz
        self.cfg = cfg

        self.nphi = nx + nu + nz + 1
        # Theta maps phi -> x_next, shape (nx, nphi)
        self.Theta = np.zeros((nx, self.nphi), dtype=float)
        self.P = np.eye(self.nphi, dtype=float) * cfg.delta

    def predict(self, x: np.ndarray, u: np.ndarray, z: np.ndarray) -> np.ndarray:
        phi = np.concatenate([x, u, z, np.ones(1)])
        return (self.Theta @ phi).reshape(self.nx)

    def update(self, x: np.ndarray, u: np.ndarray, z: np.ndarray, x_next: np.ndarray) -> None:
        lam = self.cfg.lam
        phi = np.concatenate([x, u, z, np.ones(1)]).reshape(-1, 1)  # (nphi,1)

        # Gain
        Pphi = self.P @ phi
        #denom = lam + float(phi.T @ Pphi)
        denom = lam + (phi.T @ Pphi).item()
        K = Pphi / denom  # (nphi,1)

        # Prediction error
        yhat = (self.Theta @ phi).reshape(self.nx)
        err = (x_next.reshape(self.nx) - yhat).reshape(self.nx, 1)

        # Update Theta row-wise
        self.Theta = self.Theta + (err @ K.T)

        # Update covariance
        self.P = (self.P - K @ (phi.T @ self.P)) / lam
# Functions for environment behaviour description
import json
import torch as th
import numpy as np

# Type options
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard # type: ignore

# Hyper_parameters loading
with open(r'data/parameters.json') as json_file:
    env_parameters = json.load(json_file)

# Necessary for typing verification
patch_typeguard()


@typechecked
def receiver(w: TensorType[-1, env_parameters["env"]["N_rx"], th.cfloat],
             h: TensorType[env_parameters["env"]["N_rx"], -1, th.cfloat],
             x: TensorType[1, th.float32],
             N: int, n: th.distributions.multivariate_normal.MultivariateNormal) -> TensorType[-1, th.cfloat]:
    # Noise sampling
    z = n.sample(th.Size((N,))) + 1j * n.sample(th.Size((N,)))

    # Received symbol
    y = (th.conj(th.t(w)) * h).sum(dim = 0) * x + (th.conj(w) * z).sum(dim = 1)

    return y


@typechecked
def a(phi: TensorType[-1, th.float32], N_rx: int = env_parameters["env"]["N_rx"]) -> (
        TensorType)[env_parameters["env"]["N_rx"], -1, th.cfloat]:
    phi_numpy = phi.numpy()
    return th.tensor([np.exp(1j * np.pi * i * np.sin(phi_numpy)) for i in range(N_rx)],
                     dtype = th.cfloat, requires_grad = False, device = 'cpu')


@typechecked
def Bayesian_update(w: TensorType[-1, env_parameters["env"]["N_rx"], th.cfloat],
                 pi: TensorType[-1, env_parameters["env"]["N"], th.float32],
                 alpha: TensorType[-1, th.cfloat],
                 phi: TensorType[-1, th.float32],
                 phi_set: TensorType[env_parameters["env"]["N"], th.float32],
                 P_tx: TensorType[1, th.float32],
                 n: th.distributions.multivariate_normal.MultivariateNormal) -> TensorType[-1, env_parameters["env"]["N"], th.float32]:
    # Channel computing
    h = alpha * a(phi)

    n_estimates = h.size()[-1]

    # Received symbol
    y = receiver(w, h, th.sqrt(P_tx), n_estimates, n)

    # Likelihood computing
    exp_pow = y.view(-1, 1) - th.sqrt(P_tx) * alpha.view(-1, 1) * th.matmul(th.conj(w), a(phi_set))
    LH = 1 / np.pi * th.exp(-(th.pow(exp_pow.real, 2) + th.pow(exp_pow.imag, 2)))

    # Bayesian update
    pi_new = pi * LH / th.sum(pi * LH, dim = 1, keepdim = True)

    return pi_new


if __name__ == '__main__':
    print("Use it for environment.py debugging.")
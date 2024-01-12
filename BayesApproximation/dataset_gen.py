# Full dataset generation
import pickle
import numpy as np
import torch as th
import environment as env
import matplotlib.pyplot as plt

# Type options
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard # type: ignore

# Necessary for typing verification
patch_typeguard()


@typechecked
class CustomDataSet(th.utils.data.Dataset):
    """
    Full dataset generation.
    """
    def __init__(self, N_samples: int = env.env_parameters["training"]["N_dataset"],
                       P_tx: TensorType[1, th.float32] = th.tensor(env.env_parameters["env"]["pilots_power"], dtype = th.float32, requires_grad = False),
                       n: th.distributions.multivariate_normal.MultivariateNormal =
                            th.distributions.multivariate_normal.MultivariateNormal(loc = th.zeros(env.env_parameters["env"]["N_rx"]),
                                                                         covariance_matrix = th.eye(env.env_parameters["env"]["N_rx"])),
                       dataset_path: str = r'/data/dataset.pickle',
                       g_or_l: bool = True,
                       seed: int = env.env_parameters["env"]["seed"]):
        super(CustomDataSet, self).__init__()

        self.angle_set = th.linspace(np.deg2rad(env.env_parameters["env"]["angles"][0]),
                                     np.deg2rad(env.env_parameters["env"]["angles"][1]),
                                     env.env_parameters["env"]["N"], dtype = th.float32, requires_grad = False)

        # New dataset generation or loading existing dataset
        if g_or_l:
            # Initialize random seed
            th.manual_seed(seed)
            np.random.seed(seed)

            # Weights and channel generation
            w = n.sample(th.Size((N_samples,))) + 1j * n.sample(th.Size((N_samples,)))
            w = w / th.norm(w, dim = 1, keepdim = True)

            alpha = th.randn((N_samples,)) + 1j * th.randn((N_samples,))
            phi = th.tensor(np.random.randint(0, env.env_parameters["env"]["N"], size = (N_samples,)),
                            dtype = th.long, requires_grad = False)

            # Random a priori probabilities
            pi = th.ones((1, env.env_parameters["env"]["N"]), dtype = th.float32, requires_grad = False) / env.env_parameters["env"]["N"]
            pi_a = env.Bayesian_update(w, pi, alpha, self.angle_set[phi], self.angle_set, P_tx, n)

            self.dataset = {'Probabilities': pi_a, 'Perfect_phi': phi, 'Alpha': alpha}

            # Saving dataset
            with open(dataset_path, 'wb') as f:
                pickle.dump(self.dataset, f)

        else:
            # Loading dataset
            with open(dataset_path, 'rb') as f:
                self.dataset = pickle.load(f)


    def __len__(self):
        return self.dataset['Perfect_phi'].size(0)


    def __getitem__(self, item):
        return {'Probabilities': th.squeeze(self.dataset['Probabilities'][item, :]),
                'Perfect_phi': self.dataset['Perfect_phi'][item],
                'Alpha': self.dataset['Alpha'][item]}


    def show_random_sample(self) -> None:
        # Choose random sample from complete dataset
        idx = th.randint(0, len(self), (1,))
        sample = self[idx]

        x = np.rad2deg(self.angle_set.numpy())
        y = sample['Probabilities'].numpy()

        # Figure size initialization
        plt.figure(figsize = (16, 10), dpi = 80)

        # Plot perfectly known phi
        phi_rad = np.rad2deg(self.angle_set[sample['Perfect_phi']].item())
        stem = plt.stem(phi_rad, 1, linefmt = 'k--', markerfmt = 'k.', basefmt = 'k.')
        stem[1].set_linewidth(2.5)

        # Add annotation
        if phi_rad < 0:
            x_a = phi_rad
            y_a = np.max(y)
            x_txt = phi_rad + 1
            y_txt = y_a
        else:
            x_a = phi_rad
            y_a = np.max(y)
            x_txt = phi_rad - 11
            y_txt = y_a
        plt.annotate(f'$\phi = {phi_rad:.2f}^\circ$', xy = (x_a, y_a), xytext = (x_txt, y_txt), fontsize = 14)

        stem = plt.stem(x, y, markerfmt = '.', basefmt = '.')
        stem[1].set_linewidth(2.5)
        plt.xlim((x[0], x[-1]))
        plt.xlabel(r'$\phi, deg.$', fontsize = 16)
        plt.xticks(fontsize = 13)
        plt.ylim((0, np.max(y) + 0.01))
        plt.ylabel(r'$P_{a}(\phi)$', fontsize = 16)
        plt.yticks(fontsize = 13)
        plt.title('A priori Probability', fontsize = 16)
        plt.grid()
        plt.show()


if __name__ == '__main__':
    noise = th.distributions.multivariate_normal.MultivariateNormal(loc = th.zeros(env.env_parameters["env"]["N_rx"]),
                                                                    covariance_matrix = th.eye(env.env_parameters["env"]["N_rx"]))

    dataset = CustomDataSet(N_samples = env.env_parameters['training']['N_dataset'],
                            P_tx = th.pow(10, th.tensor(env.env_parameters['env']['pilots_power']).view(-1) / 10),
                            n = noise,
                            dataset_path = r'data/dataset.pickle',
                            g_or_l = False,
                            seed = env.env_parameters['env']['seed'])

    # Show random a priori distribution and perfect known angle phi
    dataset.show_random_sample()
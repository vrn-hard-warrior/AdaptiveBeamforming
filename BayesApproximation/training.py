# Training model
import json
import datetime
import torch as th
import numpy as np
import environment as env
import dataset_gen as dataset
from torch.utils.tensorboard import SummaryWriter

# Type options
from typing import Tuple
from typeguard import typechecked
from torchtyping import TensorType, patch_typeguard # type: ignore

# Hyper_parameters loading
with open(r'data/parameters.json') as json_file:
    training_parameters = json.load(json_file)

# Necessary for typing verification
patch_typeguard()


@typechecked
class CustomNet(th.nn.Module):
    """
    Custom network for G:P(phi) -> w function approximation.
    """
    def __init__(self, hidden_dims: Tuple[int, ...] = (256, 256),
                       activation_fc = th.nn.Tanh(),
                       n: th.distributions.multivariate_normal.MultivariateNormal =
                       th.distributions.multivariate_normal.MultivariateNormal(loc = th.zeros(training_parameters["env"]["N_rx"]),
                                                                               covariance_matrix = th.eye(training_parameters["env"]["N_rx"]))):
        super(CustomNet, self).__init__()

        self.P_tx = th.pow(10, th.tensor(training_parameters['env']['pilots_power']).view(-1) / 10)
        self.noise = n
        self.phi_set = th.linspace(np.deg2rad(training_parameters["env"]["angles"][0]),
                                     np.deg2rad(training_parameters["env"]["angles"][1]),
                                     training_parameters["env"]["N"], dtype = th.float32, requires_grad = False)

        # Model architecture
        self.activation_fc = activation_fc
        self.input_layer = th.nn.Linear(training_parameters["env"]["N"], hidden_dims[0])
        self.hidden_layers = th.nn.ModuleList()

        for i in range(len(hidden_dims) - 1):
            # Add batch normalization layer for faster convergence
            batch_norm_layer = th.nn.BatchNorm1d(hidden_dims[i], affine = True,
                                                                 eps = training_parameters["training"]["eps"],
                                                                 momentum = 0.04)
            self.hidden_layers.append(batch_norm_layer)

            hidden_layer = th.nn.Linear(hidden_dims[i], hidden_dims[i + 1])
            self.hidden_layers.append(hidden_layer)

        self.output_layer = th.nn.Linear(hidden_dims[-1], training_parameters["env"]["N_rx"] * 2)


    def forward(self, prob: TensorType[-1, training_parameters["env"]["N"], th.float32],
                      phi: TensorType[-1, th.long],
                      alpha: TensorType[-1, th.cfloat]) -> TensorType[-1, training_parameters["env"]["N"], th.float32]:
        pi_new = prob

        for k in range(training_parameters["env"]["tau"]):
            # Forward propagation through model
            x = self.activation_fc(self.input_layer(pi_new))

            for hidden_layer in self.hidden_layers:
                x = self.activation_fc(hidden_layer(x))
            x = self.output_layer(x)

            # Bayesian update with returned weights
            w_output = self.transform_output(x)
            pi_new = env.Bayesian_update(w = w_output,
                                         pi = prob,
                                         alpha = alpha,
                                         phi = self.phi_set[phi],
                                         phi_set = self.phi_set,
                                         P_tx = self.P_tx,
                                         n = self.noise)

        return pi_new


    @staticmethod
    def transform_output(w: TensorType[-1, training_parameters["env"]["N_rx"] * 2, th.float32]) -> (
            TensorType)[-1, training_parameters["env"]["N_rx"], th.cfloat]:
        w_complex = w[:, : training_parameters["env"]["N_rx"]] + 1j * w[:, training_parameters["env"]["N_rx"]:]
        return w_complex / th.norm(w_complex, p = 2, dim = 1, keepdim = True)


@typechecked
class Trainer:
    """
    Model training with loaded hyperparameters.
    """
    def __init__(self, dataset: th.utils.data.Dataset):
        super(Trainer, self).__init__()

        self.N_dataset = training_parameters["training"]["N_dataset"]
        self.batch_size = training_parameters["training"]["batch_size"]
        self.N_epochs = training_parameters["training"]["N_epochs"]

        self.model = CustomNet(hidden_dims = tuple(training_parameters["training"]["N_hidden"]),
                               activation_fc = eval(training_parameters["training"]["activation_fc"] + '()'))

        self.loss_fc = eval(training_parameters["training"]["loss_fc"] + '()')
        self.optimizer = th.optim.Adam(params = self.model.parameters(),
                                       lr = training_parameters["training"]["learning_rate"],
                                       betas = training_parameters["training"]["betta"],
                                       eps = training_parameters["training"]["eps"])

        # Create scheduler
        self.scheduler = th.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma = 0.8, last_epoch = -1)

        # Splitting data to train and validation sets
        indices = np.arange(self.N_dataset, dtype = int)
        split = int(np.floor(training_parameters["training"]["validation_split"] * self.N_dataset))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[: split]

        # Creating data samplers
        train_sampler = th.utils.data.SubsetRandomSampler(list(train_indices))
        val_sampler = th.utils.data.SubsetRandomSampler(list(val_indices))

        self.train_data_loader = th.utils.data.DataLoader(dataset,
                                                          batch_size = self.batch_size,
                                                          sampler = train_sampler,
                                                          drop_last = True)

        self.val_data_loader = th.utils.data.DataLoader(dataset,
                                                        batch_size = self.batch_size,
                                                        sampler = val_sampler,
                                                        drop_last = True)


    def train_one_epoch(self, epoch_idx: np.int32, tb_writer: th.utils.tensorboard.writer.SummaryWriter) -> np.float64:
        running_loss = 0.
        last_loss = 0.

        for i, data in enumerate(self.train_data_loader):
            # Take input/label pairs
            inputs, labels = data['Probabilities'], data['Perfect_phi']

            # Zero all gradients
            self.optimizer.zero_grad()

            # Make predictions for this batch
            outputs = self.model(inputs, labels, data['Alpha'])

            # Compute the loss and it's gradients
            loss = self.loss_fc(th.log(outputs + training_parameters["training"]["eps"]), labels)
            loss.backward()

            # Adjust learning weights
            self.optimizer.step()

            # Gather data and report
            running_loss += loss.item()
            if i % 100 == 99:
                last_loss = running_loss / 10  # loss per batch
                print('    batch {} loss: {}'.format(i + 1, last_loss))
                tb_x = epoch_idx * len(self.train_data_loader) + i + 1
                tb_writer.add_scalar('Loss/train', last_loss, tb_x)
                running_loss = 0.

        return np.float64(last_loss)


    def train(self) -> None:
        # Initializing in a separate cell, so we can easily add more epochs to the same run
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        writer = SummaryWriter('runs/bayes_trainer_{}'.format(timestamp))

        best_vloss = 1_000_000.

        for epoch in np.arange(self.N_epochs, dtype = np.int32):
            print('EPOCH {}:'.format(epoch + 1))

            # Make sure gradient tracking is on, and do a pass over the data
            self.model.train(True)
            avg_loss = self.train_one_epoch(epoch, writer)

            # Update scheduler
            self.scheduler.step()

            running_vloss = 0.0

            # Set the model to evaluation mode, disabling dropout and using population statistics for batch normalization
            self.model.eval()

            # Disable gradient computation and reduce memory consumption
            with th.no_grad():
                for i, vdata in enumerate(self.val_data_loader):
                    vinputs, vlabels = vdata['Probabilities'], vdata['Perfect_phi']
                    voutputs = self.model(vinputs, vlabels, vdata['Alpha'])
                    vloss = self.loss_fc(th.log(voutputs + 1e-8), vlabels)
                    running_vloss += vloss.item()

            avg_vloss = running_vloss / (i + 1)
            print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))

            # Log the running loss averaged per batch for both training and validation
            writer.add_scalars(main_tag = 'Training vs. Validation Loss',
                               tag_scalar_dict = {'Training': avg_loss, 'Validation': avg_vloss},
                               global_step = epoch + 1)
            writer.flush()

            # Track the best performance, save the model's state
            if avg_vloss < best_vloss:
                best_vloss = avg_vloss
                model_path = 'models/model_{}_{}'.format(timestamp, epoch)
                th.save(self.model.state_dict(), model_path)

        # Close tensorboard logger
        writer.close()

        # Save model weights and biases
        th.save(self.model.state_dict(), r'models/full_model')


if __name__ == '__main__':
    # Create noise generator
    noise = th.distributions.multivariate_normal.MultivariateNormal(loc = th.zeros(env.env_parameters["env"]["N_rx"]),
                                                                    covariance_matrix = th.eye(
                                                                        env.env_parameters["env"]["N_rx"]))

    # Instantiate DataSet object
    dataset_ = dataset.CustomDataSet(N_samples = env.env_parameters['training']['N_dataset'],
                            P_tx = th.pow(10, th.tensor(env.env_parameters['env']['pilots_power']).view(-1) / 10),
                            n = noise,
                            dataset_path = r'data/dataset.pickle',
                            g_or_l = False,
                            seed = env.env_parameters['env']['seed'])

    # Instantiate Trainer object
    trainer_object = Trainer(dataset_)

    # Model training
    trainer_object.train()
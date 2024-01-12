# Hyper_parameters generation

parameters = {'env': {'N_rx': 32,                               # Number of Rx antennas
                        'N_tx': 1,                              # Number of Tx antennas
                        'tau': 14,                              # Number of time frames
                        'N': 128,                               # Size of angle grid set
                        'angles': [-60, 60],                    # Min and max angles, [deg.]
                        'pilots_power': 0,                      # Power of pilot symbols, [dB]
                        'seed': 73
                      },
              'training': {'N_dataset': 100000,                 # Dataset length
                           'batch_size': 1024,                  # Number of batches per epoch
                           'N_epochs': 50,                      # Total number of epochs
                           'N_hidden': (512, 512, 512),         # Hidden layers description
                           'activation_fc': 'th.nn.Tanh',       # Neuron's activation function
                           'loss_fc': 'th.nn.NLLLoss',          # Loss function
                           'learning_rate': 1e-1,               # Learning rate
                           'betta': (0.9, 0.999),               # Betta hyperparameters initialization
                           'eps': 1e-8,                         # Epsilon for zero division control
                           'validation_split': 0.2              # Split dataset on train and validation sets
                           }
              }


if __name__ == "__main__":
    import json

    with open(r'data/parameters.json', 'w') as outfile:
        json.dump(parameters, outfile, ensure_ascii = False, indent = 4)
# Adaptive Beamforming Projects
Here is all my research activity and developments in adaptive beamforming area, initial beam alignment, beam tracking tasks, etc. Also this repository will contain some well known optimization algorithms for hybrid beamforming systems in multi-user MIMO wireless communications. Channel tensors are computed by [QuaDRiGa][1], [Saleh-Valenzuela model][2] or more simple [sparse channel models][3] for millimeter waves. Separate folders are allocated for all projects and a detailed descriptions for each of them are provided below: 
- #### Deep Learning approximation for Bayesian initial beam alignment
An attempt to repeat an F. Sohrabi's [article][4], in which Bayesian method is used for initial beam alignment task. With almost identical input parameters FCNN (Fully-Connected Neural Network) can't be trained even in $10^{3}$ epochs. Can we implement this algorithm in principle?

[comment]: # (All necessary links and references)
[1]: https://github.com/fraunhoferhhi/QuaDRiGa
[2]: https://ieeexplore.ieee.org/document/6848765
[3]: https://ieeexplore.ieee.org/document/6834753
[4]: https://ieeexplore.ieee.org/document/9448070
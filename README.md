To run experiment 4.1 for L_proj:

run ./experiment4_1/Convex_SNNs.ipynb for multi-layer SNN fitting convex function surface.

change the func in ./experiment4_1/Convex_SNNs.ipynb update_weights_2 to update_weights for single-layer fitting convex function surface.

To test this local loss function for MNIST:

use ./mnist_***_train/test.py and calaccuracy.py for testaccuracy

Set time_steps=numbins-1=3,dt=3e-5,leak=2,sigmav=0,mu=0,alpha1=1e7,alpha2=1e-1

To run experiment 4.1 for L_ce:

python torch_snn.py --model mlp --dataset MNIST --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin snn --loss-sup pred

To run experiment 4.1 for L_corr:

python torch_snn.py --model mlp --dataset MNIST --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin snn --loss-sup sim

To run experiment 4.1 for L_ce_corr:

python torch_snn.py --model mlp --dataset MNIST --lr 5e-4 --num-layers 3 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin snn

To run experiment 4.2 on MNIST:

python torch_snn.py --model snn1m --dataset MNIST --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin snn --loss-sup pred

python torch_snn.py --model vgg8a --dataset MNIST --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin snn --loss-sup pred

To run experiment 4.3 on FashionMNIST:

python torch_snn.py --model snn2m --dataset FashionMNIST --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin snn --loss-sup pred

python torch_snn.py --model vgg8a --dataset FashionMNIST --lr 5e-4 --epochs 100 --lr-decay-milestones 50 75 89 94 --nonlin snn --loss-sup pred


We run our experiment on one gpu-titan.
And the FashionMNIST use the MIT license.
We refer to the relevant engineering codes in [1][2], and conduct SNN non-BP algorithm experiments on this basis.
[1] Mancoo, A., Keemink, S. and Machens, C. K. [2020], Understanding spiking networks through convex optimization, in ‘Advances in Neural Information Processing Systems’, Vol. 33.
[2] Nøkland, A. and Eidnes, L. H. [2019], Training neural networks with local error signals, in ‘International Conference on Machine Learning’, PMLR, pp. 4839–4850.

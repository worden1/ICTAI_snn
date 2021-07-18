import numpy as np
from numpy.random import RandomState
import numba as nb
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import holoviews as hv
import snn_cvx



from mnist import MNIST
#loading MNIST dataset
mndata = MNIST('~/data/MNIST/')
# mndata.gz = False
GrayLevels = 255  # Image GrayLevels
tmax = 256  # Simulatin time
cats = [4, 1, 0, 7, 9, 2, 3, 5, 8, 6]  # Reordering the categories
images = []  # To keep training images
labels = []  # To keep training labels
images_test = []  # To keep test images
labels_test = []  # To keep test labels

Images, Labels = mndata.load_training()
Images=np.array(Images)    
for i in range(len(Labels)):
    if Labels[i] in cats:
        images.append(np.floor(Images[i].reshape(28,28)).astype(int))
        #images.append(np.floor((GrayLevels-Images[i].reshape(28,28))*tmax/GrayLevels).astype(int))
        labels.append(cats.index(Labels[i]))
  
Images, Labels = mndata.load_testing()
Images=np.array(Images)
for i in range(len(Labels)):
    if Labels[i] in cats:
        images_test.append(np.floor(Images[i].reshape(28,28)).astype(int))            
        #images_test.append(np.floor((GrayLevels-Images[i].reshape(28,28))*tmax/GrayLevels).astype(int)) 
        labels_test.append(cats.index(Labels[i]))
                        
del Images,Labels

images = np.transpose(np.asarray(images), (1, 2, 0))
labels = np.asarray(labels)
images_test = np.transpose(np.asarray(images_test), (1, 2, 0))
labels_test = np.asarray(labels_test)



def update_weights_images(images,
                   labels,
                   F_weights,
                   G_weights,
                   omega,
                   omegak,
                   thresholds,
                   buffer_bins,
                   dt,
                   leak,
                   leak_thresh,
                   alpha_thresh,
                   alpha_F,
                   mu=0.,
                   sigma_v=0.,
                   ):
    """
    Train the network in one trial with one presented input-target pair (x_sample, y_target_sample)
    The function returns the updated thresholds and feed-forward weights after that trial
    Parameters
    ----------
    images: array
        Input array (shape=[??])
    labels: array
        target sample (shape=[??])
    F_weights: array
        Feed-forward weights (shape=[N, K])
    G_weights: array
        Encoder weights (shape=[N, M])
    omega: array
        Recurrent weights (shape=[N, N])
    thresholds: array
        Neurons thresholds (shape=[N,])
    buffer_bins: int
        Number of bins before learning starts
    dt: float
        time step
    leak: float
        membrane leak time-constant
    leak_thresh: float
        controls the speed of drift in thresholds
    alpha_thresh: float
        learning rate of thresholds (>> leak thresh)
    alpha_F: float
        learning rate for forward weights
    mu: float
        controls spike cost
    sigma_v: float
        controls variance of voltage noise
    Returns
    -------
    array
        updated thresholds array
    array
        updated feed-forward weights array
    """

    # initialize system
    #beta = 1 / 500
    N = F_weights.shape[0]
    num_bins = images.shape[2]
    firing_rates = np.zeros((N, num_bins))
    V_membrane = np.zeros(N)
    #V_membrane_2 = np.zeros(N)
    #firing_rates_2 = np.zeros((N, num_bins))
    #print('init')

    # implement the Euler method to solve the differential equations
    for t in range(num_bins - 1):
        #print('t=',t)
        # compute command signal
        command_x = (images[:, :, t + 1] -
                     images[:, :, t]) / dt + leak * images[:, :, t]
        #print('layer1in')

        # update membrane potential
        V_membrane += dt * (-leak * V_membrane +
                            np.tensordot(F_weights, command_x, ([1,2],[0,1]))
                            ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)
        # update rates
        firing_rates[:, t + 1] = (1 - leak * dt) * firing_rates[:, t]
        #print('layer1--')

        # Check if any neurons are past their threshold during the last time-step
        diff_voltage_thresh = V_membrane - thresholds
        spiking_neurons_indices = np.arange(N)[diff_voltage_thresh >= 0]

        if spiking_neurons_indices.size > 0:
            # Pick the neuron which likely would have spiked first, by max distance from threshold
            to_pick = np.argmax(V_membrane[spiking_neurons_indices] - thresholds[spiking_neurons_indices])
            s = spiking_neurons_indices[to_pick]

            # Update membrane potential
            V_membrane[s] -= mu
            omegak = np.sqrt(np.nanmax(abs(V_membrane)) / 1000)
            V_membrane += omegak**2 * omega[:, s]

            # Update rates with spikes
            firing_rates[s, t + 1] += 1
            #print('layer1spike')

            # !! Update weights
            if t >= buffer_bins:
                #omegak = np.sqrt(np.nanmax(abs(V_membrane)) / 1000)
                proj_error_neuron = np.tensordot(F_weights[s, :], images[:, :, t]) - thresholds[
                    s] - omegak * G_weights[s, :] @ labels
                #proj_error_neuron = np.tensordot(F_weights[s, :], images[:, :, t]) - thresholds[
                    #s] - G_weights[s, :] @ labels - beta * (np.tensordot(F_weights[s, :], images[:, :, t]) - thresholds[s] )

                dLdthresh = -proj_error_neuron
                dLdf_weights = proj_error_neuron * images[:, :, t]

                thresholds[s] -= alpha_thresh * dLdthresh
                F_weights[s, :, :] -= alpha_F * dLdf_weights
                #print('t=',t)
                #print('layer1grad')

        else:
            pass
        
        # drift thresholds
        if t >= buffer_bins:
            thresholds -= dt * leak_thresh
            #thresholds_2 -= dt * leak_thresh

    return thresholds, F_weights, omegak



# setting up dimensions and initial parameters 
#M = 1
M = 10
K = 2
K0 = images.shape[0]
K1 = images.shape[1]
N = 500
leak = 2

# initialize my gamma matrix
#random_state = np.random.RandomState(seed=3)
random_state = RandomState(seed=3)
#D_weights_init = random_state.rand(M, N)
D_weights_init = np.ones((M, N))
#D_weights_init = D_weights_init / np.linalg.norm(D_weights_init, axis=0)
D_weights_init = D_weights_init / np.linalg.norm(D_weights_init, axis=0)
print(D_weights_init)
G_weights_init = D_weights_init.copy().T
print(G_weights_init.shape)
F_weights_init = random_state.randn(K0, K1, N).T
print(F_weights_init.shape)
omega_init = -G_weights_init @ D_weights_init - 999 * np.eye(N)
#e_vals,e_vecs = np.linalg.eig(-omega_init)
print(omega_init)
thresholds_init = 2*random_state.rand(N) - 1

T = 4 # simulation time
dt = 3e-03 # time step
t_span = np.arange(0, T, dt)
num_bins = t_span.size
#buffer_bins = int(1/dt)
buffer_bins = int(2)
#buffer_zeros = int(buffer_bins/2)
buffer_zeros = int(1)
x_sample = np.zeros((K, num_bins))
images_sample = np.zeros((K0, K1, num_bins))

# initialize network parameters
D_weights = D_weights_init.copy()
G_weights = G_weights_init.copy()
F_weights = F_weights_init.copy()
omega = omega_init.copy()
thresholds = thresholds_init.copy()

# run supervised learning
#alpha_thresh_init = 1e-03
alpha_thresh_init = 1e-04
#alpha_F_init = 1e-03
alpha_F_init = 1e-07
leak_thresh = 0.
#leak_thresh = 1e-03
omegak = 1

num_epochs = 100
thresholds_array_fit = np.zeros((N, num_epochs))
F_weights_array_fit = np.zeros((N, K0, K1, num_epochs))
decrease_learning_rate = True



for epoch in range(num_epochs):
    print ('-----------------------------------------------------iteration: ',epoch+1)
    data_index_list = np.arange(images.shape[2])
    np.random.shuffle(data_index_list)
    
    if decrease_learning_rate:
        alpha_thresh = alpha_thresh_init * np.exp(-0.0001 * (epoch + 1))
        alpha_F = alpha_F_init * np.exp(-0.0001 * (epoch + 1))

    else:
        alpha_thresh = alpha_thresh_init
        alpha_F = alpha_F_init
    
    #for data_index in range(10):
    turn = 0
    for data_index in data_index_list:
        #print('i=',data_index)
        if turn > 10:
            break
        print('i=',turn)
        turn += 1

        #omegak = 1
        images_sample[:, :, buffer_zeros:] = images[:, :, data_index][:, :, None]
        labels_sample = np.zeros((M, 1))
        #labels_sample[labels[data_index]] = 1000
        labels_sample[labels[data_index]] = 1
        #labels_sample = labels[data_index]

        thresholds, F_weights, omegak = update_weights_images(
            images_sample,
            labels_sample,
            F_weights,
            G_weights,
            omega,
            omegak,
            thresholds,
            buffer_bins,
            dt,
            leak,
            leak_thresh,
            alpha_thresh,
            alpha_F,
            mu=0,
            sigma_v=0.
        )
        print(omegak)

    thresholds_array_fit[:, epoch] = thresholds
    F_weights_array_fit[:, :, :, epoch] = F_weights


print(omegak)
#save trained parameters
np.savez('~/data/mnist_1_500', thresholds_array_fit = thresholds_array_fit, F_weights_array_fit = 
F_weights_array_fit, omega = omegak**2 * omega, D_weights = omegak * D_weights, omegak = omegak)
import numpy as np
import numba as nb


#@nb.jit(nopython=True)
def run_snn_trial_2(images,
                  F_weights,F_weights_2,
                  omega,omega_2,
                  thresholds,thresholds_2,
                  dt,
                  leak,
                  mu=0.,
                  sigma_v=0.
                  ):
    """
    Function to simulate the spiking network for defined connectivity parameters, thresholds and time parameters.
    It returns the instantaneous firing rates of neurons for the whole simulation time
    Parameters
    ----------
    x_sample: array
        Input array (shape=[K, num_bins])
    F_weights: array
        Feed-forward weights (shape=[N, K])
    omega: array
        Recurrent weights (shape=[N, N])
    thresholds: array
        Neurons thresholds (shape=[N,])
    dt: float
        time step
    leak: float
        membrane leak time-constant
    mu: float
        controls spike cost
    sigma_v: float
        controls variance of voltage noise

    Returns
    -------
    array
        network instantaneous firing rates (shape=[N x num_bins])
    """

    # initialize system
    N = F_weights.shape[0]  # number of neurons
    num_bins = images.shape[2]  # number of time bins
    firing_rates = np.zeros((N, num_bins))
    V_membrane = np.zeros(N)
    V_membrane_2 = np.zeros(N)
    firing_rates_2 = np.zeros((N, num_bins))
    #print('init')


    # implement the Euler method to solve the differential equations
    for t in range(num_bins - 1):
        # compute command signal
        command_x = (images[:, :, t + 1] -
                     images[:, :, t]) / dt + leak * images[:, :, t]

        #print(np.tensordot(F_weights, command_x, ([1,2],[0,1])))
        #print(-leak * V_membrane)
        # update membrane potential
        V_membrane += dt * (-leak * V_membrane +
                            np.tensordot(F_weights, command_x, ([1,2],[0,1]))
                            ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)

        # update firing rates
        firing_rates[:, t + 1] = (1 - leak * dt) * firing_rates[:, t]

        # Check if any neurons are past their threshold during the last time-step
        diff_voltage_thresh = V_membrane - thresholds
        spiking_neurons_indices = np.arange(N)[diff_voltage_thresh >= 0]
        if spiking_neurons_indices.size > 0:
            # Pick the neuron which likely would have spiked first, by max distance from threshold
            to_pick = np.argmax(V_membrane[spiking_neurons_indices] - thresholds[spiking_neurons_indices])
            s = spiking_neurons_indices[to_pick]

            # Update membrane potential
            V_membrane[s] -= mu
            V_membrane += omega[:, s]

            # Update firing rates
            firing_rates[s, t + 1] += 1

        else:
            pass


        #layer 2
        #V_membrane_2 += dt * (-leak * V_membrane_2 +
        #                    F_weights_2 @ firing_rates[:, t]
        #                    ) + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)
        V_membrane_2 = F_weights_2 @ firing_rates[:, t] + dt * leak * firing_rates[:, t
                            ] + np.sqrt(2 * dt * leak) * sigma_v * np.random.randn(N)

        firing_rates_2[:, t + 1] = (1 - leak * dt) * firing_rates_2[:, t]
        #print('layer2--')

        diff_voltage_thresh_2 = V_membrane_2 - thresholds_2
        spiking_neurons_indices_2 = np.arange(N)[diff_voltage_thresh_2 >= 0]
        if spiking_neurons_indices_2.size > 0:
            # Pick the neuron which likely would have spiked first, by max distance from threshold
            to_pick_2 = np.argmax(V_membrane_2[spiking_neurons_indices_2] - thresholds_2      [spiking_neurons_indices_2])
            s_2 = spiking_neurons_indices_2[to_pick_2]

            # Update membrane potential
            V_membrane_2[s_2] -= mu
            V_membrane_2 += omega_2[:, s_2]

            # Update rates with spikes
            firing_rates_2[s_2, t + 1] += 1
            #print('layer2spike')
        
        else:
            pass

    return firing_rates,firing_rates_2



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



T = 4 # simulation time
dt = 3e-03 # time step
t_span = np.arange(0, T, dt)
num_bins = t_span.size
buffer_bins = int(1/dt)
buffer_zeros = int(buffer_bins/2)
#buffer_zeros = int(1)
K0 = images.shape[0]
K1 = images.shape[1]
leak = 2
# test accuracy main()
# run snn with learnt parameters
images_sample = np.zeros((K0, K1, num_bins))

# call learnt parameters
mnist_train=np.load('D:/code/SNN-IB/mnist_2_100_100.npz')
F_weights_fit = mnist_train['F_weights_array_fit'][:, :, :, -1]
thresholds_fit = mnist_train['thresholds_array_fit'][:, -1]
thresholds_2_fit = mnist_train['thresholds_2_array_fit'][:, -1] 
F_weights_2_fit = mnist_train['F_weights_2_array_fit'][:, :, -1]
D_weights = mnist_train['D_weights']
omega = mnist_train['omega']
y_readout = []

for data_index in range(images_test.shape[2]):
    print('i=',data_index)
    images_sample[:, :, buffer_zeros:] = images[:, :, data_index][:, :, None]
    #images_sample[:, :, buffer_zeros:] = images_test[:, :, data_index][:, :, None]
    #print(images_sample[:, :, buffer_zeros])

    rates, rates_2 = run_snn_trial_2(
        images_sample,
        F_weights_fit,F_weights_2_fit,
        omega,omega/100,
        thresholds_fit,thresholds_2_fit,
        dt,
        leak,
    )
    temp = D_weights @ rates
    temp2 = D_weights @ rates_2

    y_readout += [np.copy(D_weights @ rates_2)]
    #if data_index > 0:
       # print((y_readout[data_index] == y_readout[data_index - 1]).all())
    
#average_readouts_fit = (np.array(y_readout)[:, buffer_zeros + 500:].mean(axis=2)).argmax(axis=1)
average_readouts_fit = np.array(y_readout)[:, :, buffer_zeros + 500:].mean(axis=2)
test_accuracy = (labels_test == average_readouts_fit).sum() / len(labels_test)

#save test labels
np.savez('~/data/mnist_2_100_100_testlabels', average_readouts_fit = average_readouts_fit, 
test_accuracy = test_accuracy)
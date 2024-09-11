# -*- coding: utf-8 -*-

import numpy as np
from scipy.stats import qmc
import pandas as pd
import math

# Define the parameter ranges
number_of_apps = [1, 50]
number_of_devices = [1, 30]
resource_allocation = [0.1, 3]  # ratio (x100%)
network_bandwidth = [200, 1000]  # in MBps

# Number of samples to generate
num_samples = 50

# Initialize the LHS sampler
sampler = qmc.LatinHypercube(d=4)

# Generate LHS samples
lhs_samples = sampler.random(n=num_samples)
print(lhs_samples.shape)

# Custom scale function
def scale_samples(samples, lower_bound, upper_bound):
    return lower_bound + samples * (upper_bound - lower_bound)

# Scale the samples to the parameter ranges
number_of_devices_samples = scale_samples(lhs_samples[:, 0], number_of_devices[0], number_of_devices[1])
number_of_apps_samples = scale_samples(lhs_samples[:, 1], number_of_apps[0], number_of_apps[1])
resource_allocation_samples = scale_samples(lhs_samples[:, 2], resource_allocation[0], resource_allocation[1])
network_bandwidth_samples = scale_samples(lhs_samples[:, 3], network_bandwidth[0], network_bandwidth[1])

number_of_devices_samples = [math.floor(x) for x in number_of_devices_samples]
number_of_apps_samples = [math.floor(x) for x in number_of_apps_samples]
resource_allocation_samples = [round(x, 2) for x in resource_allocation_samples]
network_bandwidth_samples = [math.floor(x) for x in network_bandwidth_samples]
# Combine the scaled samples into a DataFrame
df_samples = pd.DataFrame({
    'NumberDevices': number_of_devices_samples,
    'NumberApps': number_of_apps_samples,
    'ResourceAllocation': resource_allocation_samples,
    'NetworkBandwidth': network_bandwidth_samples
})

print(df_samples)


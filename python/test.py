import dynamic_python_model
import numpy as np
import array as arr
import math
import threading

def true_prior_expectations():
    default_prior_expectation = np.zeros(shape=(7,3), dtype="double")
    default_prior_expectation[0,:] = [1,1,1]
    default_prior_expectation[1,:] = [0.97000436, -0.97000436, 0]
    default_prior_expectation[2,:] = [-0.24308753, 0.24308753, 0]
    default_prior_expectation[3,:] = [0, 0, 0]
    default_prior_expectation[4,:] = [0.93240737 / 2, 0.93240737 / 2,
      -0.93240737]
    default_prior_expectation[5,:] = [0.86473146 / 2, 0.86473146 / 2,
      -0.86473146]
    default_prior_expectation[6,:] = [0, 0, 0]
    return np.reshape(default_prior_expectation, 21)

def default_prior_expectations():
    default_prior_expectation = np.zeros(shape=(7,3), dtype="double")
    x = 0.04
    default_prior_expectation[0,:] = [1 - x,1 + x,1 + x]
    default_prior_expectation[1,:] = [0.97000436 + x, -0.97000436 - x, 0 +x ]
    default_prior_expectation[2,:] = [-0.24308753 + x, 0.24308753 + x, 0 - x]
    default_prior_expectation[3,:] = [0 + x, 0 + x, 0 - x]
    default_prior_expectation[4,:] = [0.93240737 / 2 + x, 0.93240737 / 2 - x,
      -0.93240737 + x]
    default_prior_expectation[5,:] = [0.86473146 / 2 + x, 0.86473146 / 2 - x,
      -0.86473146 - x]
    default_prior_expectation[6,:] = [0 + x, 0 - x, 0 + x]
    return np.reshape(default_prior_expectation, 21)

def default_prior_covariances():
    flat = 1.0
    informative = 1/16
    precise = 1/256
    fixed = 1/2048
    default_prior_covariance = np.zeros(shape=(7,3), dtype="double")
    default_prior_covariance[0,:] = np.array([1,1,1]) * informative
    default_prior_covariance[1,:] = np.array([1,1,1]) * informative
    default_prior_covariance[2,:] = np.array([1,1,1]) * informative
    default_prior_covariance[3,:] = np.array([1,1,1]) * informative
    default_prior_covariance[4,:] = np.array([1,1,1]) * informative
    default_prior_covariance[5,:] = np.array([1,1,1]) * informative
    default_prior_covariance[6,:] = np.array([1,1,1]) * informative
    return_default_prior_covariance = np.zeros(shape=(21,21), dtype="double")
    np.fill_diagonal(
        return_default_prior_covariance,
        default_prior_covariance.reshape(21)
    )
    return return_default_prior_covariance

def default_hyper_expectations():
    return np.zeros(3)

def default_hyper_covariances():
    flat = 1.0
    informative = 1/16
    precise = 1/256
    fixed = 1/2048
    default_hyper_covariance = np.zeros(shape=(3), dtype="double")
    default_hyper_covariance = np.array([1,1,1]) * precise
    return_default_hyper_covariance = np.zeros(shape=(3,3), dtype="double")
    np.fill_diagonal(
        return_default_hyper_covariance,
        default_hyper_covariance
    )
    return return_default_hyper_covariance

def rungekutta(func, parameters, h):
    k1 = func(parameters)
    k2 = func(parameters + (h * k1 / 2))
    k3 = func(parameters + (h * k2 / 2))
    k4 = func(parameters + (h * k3))
    return h / 6 * (k1 + 2*k2 + 2*k3 + k4)


def differential_eq(state_in):
    G = 1
    state = np.copy(np.reshape(state_in, (7,3)))
    return_matrix = np.zeros(shape=(7,3), dtype="double")
    for i in range(state.shape[1]):
        return_matrix[1, i] = state[4, i]
        return_matrix[2, i] = state[5, i]
        return_matrix[3, i] = state[6, i]
        for j in range(state.shape[1]):
            if (i == j):
                continue
            distancex = state[1, j] - state[1, i]
            distancey = state[2, j] - state[2, i]
            distancez = state[3, j] - state[3, i]
            distance_euclidian = (
                (distancex * distancex) + (distancey * distancey) +
                (distancez * distancez)
            )**0.5
            return_matrix[4, i] += (G * state[0, j] * distancex) / \
                distance_euclidian**3
            return_matrix[5, i] += (G * state[0, j] * distancey) / \
                distance_euclidian**3
            return_matrix[6, i] += (G * state[0, j] * distancez) / \
                distance_euclidian**3
    return np.reshape(return_matrix, 21)


def generative_model(parameters, timeseries_length, selection):
    
    output = np.zeros(
        shape=(timeseries_length, len(parameters)), dtype="double"
    )
    select = np.array(selection, dtype="int")
    h = 0.001
    state = np.array(parameters)
    output[0,:] = parameters
    for i in range(1, timeseries_length):
        for j in range (10):
            state_delta = rungekutta(differential_eq, state, h)
            state += state_delta
        output[i,:] = state
    output = output[:, select]
    np.reshape(output, len(select) * timeseries_length)

    return output

def generative_model_thread(parameters, timeseries_length, selection):
    
    output = np.zeros(
        shape=(timeseries_length, len(parameters)), dtype="double"
    )
    select = np.array(selection, dtype="int")
    h = 0.001
    state = np.array(parameters)
    output[0,:] = parameters
    for i in range(1, timeseries_length):
        for j in range (10):
            state_delta = rungekutta(differential_eq, state, h)
            state += state_delta
        output[i,:] = state
    output = output[:, select]
    np.reshape(output, len(select) * timeseries_length)

    return output

dynamic_test = dynamic_python_model.PyDynamicModel()

dynamic_test.prior_parameter_expectations = \
    default_prior_expectations()
dynamic_test.prior_parameter_covariances = \
    default_prior_covariances()
dynamic_test.prior_hyper_expectations = \
    default_hyper_expectations()
dynamic_test.prior_hyper_covariances = \
    default_hyper_covariances()

dynamic_test.select_response_vars = np.array([3,6,9], dtype='intc')
dynamic_test.num_samples = 1000
dynamic_test.num_response_vars = 3

dynamic_test.external_generative_model = generative_model

true_in = generative_model(
    true_prior_expectations(),
    dynamic_test.num_samples,
    dynamic_test.select_response_vars
)

dynamic_test.response_vars = np.reshape(
    true_in,
    (
        dynamic_test.num_samples,
        len(dynamic_test.select_response_vars)
    )
)

true_out = generative_model(
    true_prior_expectations(),
    dynamic_test.num_samples,
    np.linspace(0,20,21)
)

perm0 = np.linspace(0,20,21)
perm1 = np.reshape(perm0, (7, 3))
perm = np.reshape(perm1.T, 21).astype("int")
np.reshape(true_out, (dynamic_test.num_samples, 21))

np.savetxt("../../dcm-examples/visualisation/py_true.csv", true_out[:, perm], delimiter=",")
true_cpp = np.loadtxt("../../dcm-examples/visualisation/true_generative.csv", delimiter=",")

dynamic_test.invert_model()

out = generative_model(
    dynamic_test.conditional_parameter_expectations,
    dynamic_test.num_samples,
    np.linspace(0,20,21)
)

np.savetxt("../../dcm-examples/visualisation/py_deriv.csv", out[:, perm], delimiter=",")

print("test.py")
print(np.array(dynamic_test.conditional_parameter_expectations))


print("test")



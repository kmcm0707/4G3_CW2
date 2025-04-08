import math
from enum import IntEnum

import numpy as np


class weight_options_enum(IntEnum):
    """
    Enum for weight options.
    """

    zeros = 0
    random_symmetric = 1
    symmetric_ring = 2
    balanced_ring = 3
    random_uniform_symmetric = 4
    non_normal_balanced_ring = 5


class V1_simulation:
    def __init__(
        self,
        number_of_orientations,
        kappa=1.0,
        number_of_neurons=100,
        weight_options=weight_options_enum.zeros,
        alpha=0.1,
        time_step=0.001,
        tau=0.1,
        sigma=1,
        diff_beta=False,
    ):
        """
        number_of_orientations: number of orientations to simulate (int)
        kappa: parameter for the Gaussian function (float)
        number_of_neurons: number of neurons in the simulation (int)
        """
        self.number_of_orientations = number_of_orientations
        self.kappa = kappa
        self.number_of_neurons = number_of_neurons
        self.orientations = (
            2
            * np.pi
            * np.arange(self.number_of_orientations)
            / self.number_of_orientations
        )
        self.alpha = alpha
        self.B_matrix = np.eye(self.number_of_neurons, self.number_of_orientations)
        if diff_beta:
            self.B_matrix = np.zeros(
                (self.number_of_neurons, self.number_of_orientations)
            )
            for i in range(self.number_of_orientations):
                """self.B_matrix[i, :] = np.exp(
                    (np.cos(self.orientations[i] - self.orientations) - 1)
                    / self.kappa**2
                )
                self.B_matrix[i + self.number_of_orientations, :] = -np.exp(
                    (np.cos(self.orientations[i] - self.orientations) - 1)
                    / self.kappa**2
                )"""
                """self.B_matrix[i, :] = 0.01
                self.B_matrix[i, i] = 1"""
                self.B_matrix[i, i] = 1
                self.B_matrix[i + self.number_of_orientations, i] = -1
            self.B_matrix = self.B_matrix / np.sum(self.B_matrix[0:200, :], axis=0)
        self.C_matrix = np.eye(self.number_of_orientations, self.number_of_neurons)

        self.time_step = time_step
        self.tau = tau
        self.sigma = sigma

        # Initialize the recurrent weights
        self.recurrent_weights = None
        self.weight_options = weight_options
        self.set_recurrent_weights(weight_options)

    def stimulus(self, left_orientation, right_orientation):
        # Simulate a stimulus based on the orientation
        stimulus_input = np.exp(
            (np.cos(left_orientation - right_orientation) - 1) / self.kappa**2
        )
        return stimulus_input

    def set_recurrent_weights(self, weight_options):
        """
        Set the recurrent weights based on the specified weight options.
        """
        if weight_options == weight_options_enum.zeros:
            assert self.number_of_neurons == self.number_of_orientations, (
                "The number of neurons must be equal to the number of orientations"
            )
            self.recurrent_weights = np.zeros(
                (self.number_of_neurons, self.number_of_neurons)
            )
        elif weight_options == weight_options_enum.random_symmetric:
            assert self.number_of_neurons == self.number_of_orientations, (
                "The number of neurons must be equal to the number of orientations"
            )
            self.recurrent_weights = np.random.randn(
                self.number_of_neurons, self.number_of_neurons
            )
            self.recurrent_weights = self.recurrent_weights + self.recurrent_weights.T
            self.recurrent_weights = self.set_weight_alpha(self.recurrent_weights)
        elif weight_options == weight_options_enum.random_uniform_symmetric:
            assert self.number_of_neurons == self.number_of_orientations, (
                "The number of neurons must be equal to the number of orientations"
            )
            self.recurrent_weights = (
                np.random.rand(self.number_of_neurons, self.number_of_neurons) - 0.5
            ) * 2
            self.recurrent_weights = self.recurrent_weights + self.recurrent_weights.T
            self.recurrent_weights = self.set_weight_alpha(self.recurrent_weights)
        elif weight_options == weight_options_enum.symmetric_ring:
            assert self.number_of_neurons == self.number_of_orientations, (
                "The number of neurons must be equal to the number of orientations"
            )
            self.recurrent_weights = np.zeros(
                (self.number_of_neurons, self.number_of_neurons)
            )
            for i in range(self.number_of_neurons):
                self.recurrent_weights[i, :] = self.stimulus(
                    self.orientations[i], self.orientations
                )
            self.recurrent_weights = self.set_weight_alpha(self.recurrent_weights)
        elif weight_options == weight_options_enum.balanced_ring:
            assert self.number_of_neurons == 2 * self.number_of_orientations, (
                "The number of neurons must be 2* the number of orientations"
            )
            self.recurrent_weights = np.zeros(
                (self.number_of_neurons, self.number_of_neurons)
            )
            temp_matrix = np.zeros(
                (self.number_of_orientations, self.number_of_orientations)
            )
            for i in range(self.number_of_orientations):
                temp_matrix[i, :] = self.stimulus(
                    self.orientations[i], self.orientations
                )
            temp_matrix = self.set_weight_alpha(temp_matrix)
            self.recurrent_weights = np.block(
                [[temp_matrix, -temp_matrix], [temp_matrix, -temp_matrix]]
            )
        elif weight_options == weight_options_enum.non_normal_balanced_ring:
            assert self.number_of_neurons == 2 * self.number_of_orientations, (
                "The number of neurons must be 2* the number of orientations"
            )
            self.recurrent_weights = np.zeros(
                (self.number_of_neurons, self.number_of_neurons)
            )
            temp_matrix = np.zeros(
                (self.number_of_orientations, self.number_of_orientations)
            )
            for i in range(self.number_of_orientations):
                temp_matrix[i, :] = self.stimulus(
                    self.orientations[i], self.orientations
                )
            temp_matrix = self.set_weight_alpha(temp_matrix)
            self.recurrent_weights = np.block(
                [[temp_matrix, -1.2 * temp_matrix], [temp_matrix, -1.2 * temp_matrix]]
            )
        else:
            raise ValueError("Invalid weight options. Choose from weight_options_enum.")

    def set_alpha(self, alpha):
        """
        Set the alpha parameter.
        """
        self.alpha = alpha
        self.recurrent_weights = self.set_weight_alpha(self.recurrent_weights)

    def recreate_recurrent_weights(self, weight_options):
        """
        Recreate the recurrent weights with the specified weight options.
        """
        self.set_recurrent_weights(weight_options)
        self.weight_options = weight_options

    def set_weight_alpha(self, weight, alpha=None):
        current_alpha = self.alpha if alpha is None else alpha
        largest_real_eigenvalue = np.max(np.real(np.linalg.eigvals(weight)))
        scale = current_alpha / largest_real_eigenvalue
        weight = scale * weight
        assert math.isclose(
            np.max(np.real(np.linalg.eigvals(weight))), current_alpha, rel_tol=1e-5
        ), "The largest real eigenvalue is not equal to alpha"
        return weight

    def get_recurrent_weights(self):
        """
        Get the recurrent weights.
        """
        return self.recurrent_weights

    def get_stimulus(self, orientation):
        """
        Get the stimulus function.
        """
        return self.stimulus(self.orientations, orientation)

    def run_simulation(self, orientation, time):
        """
        Run the simulation for a given time with the specified orientations.
        """
        stimulus_input = self.stimulus(self.orientations, orientation)
        rate = np.zeros((self.number_of_neurons, int(time / self.time_step)))
        noisy_rate = np.zeros((self.number_of_orientations, int(time / self.time_step)))
        noisy_readout = np.zeros((1, int(time / self.time_step)))

        time_array = np.arange(0, time, self.time_step)

        time_index = 1

        rate[:, 1] = (self.B_matrix @ stimulus_input) / self.tau
        noisy_rate[:, 1] = self.C_matrix @ rate[:, 1] + self.sigma * np.random.randn(
            self.number_of_orientations
        )
        noisy_readout[:, 1] = np.atan2(
            np.sum(np.sin(self.orientations) * noisy_rate[:, 1]),
            np.sum(np.cos(self.orientations) * noisy_rate[:, 1]),
        )

        time_index = 2
        for i in range(2, int(time / self.time_step)):
            rate[:, i] = (
                rate[:, i - 1]
                + self.time_step
                * (-rate[:, i - 1] + self.recurrent_weights @ rate[:, i - 1])
                / self.tau
            )
            noisy_rate[:, i] = self.C_matrix @ rate[
                :, i
            ] + self.sigma * np.random.randn(self.number_of_orientations)
            noisy_readout[:, i] = np.atan2(
                np.sum(np.sin(self.orientations) * noisy_rate[:, i]),
                np.sum(np.cos(self.orientations) * noisy_rate[:, i]),
            )
            time_index += 1
        return rate.T, noisy_rate.T, noisy_readout.T, time_array

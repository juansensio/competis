import torch
import math
import numpy as np
import matplotlib.pyplot as plt


Y_REG_FREQ_BINS = [1.7924e+04, 1.0090e+03, 5.9500e+02, 4.5500e+02, 4.2000e+02,
       3.1500e+02, 2.8500e+02, 2.9000e+02, 2.1400e+02, 2.0200e+02,
       2.0000e+02, 1.5700e+02, 1.6000e+02, 1.5800e+02, 1.2900e+02,
       1.1100e+02, 1.2100e+02, 1.0500e+02, 1.1600e+02, 1.2600e+02,
       1.0200e+02, 8.3000e+01, 8.7000e+01, 8.7000e+01, 6.7000e+01,
       5.3000e+01, 5.6000e+01, 6.9000e+01, 4.6000e+01, 3.7000e+01,
       4.4000e+01, 4.0000e+01, 4.2000e+01, 3.4000e+01, 3.8000e+01,
       3.2000e+01, 2.9000e+01, 2.3000e+01, 1.8000e+01, 3.0000e+01,
       2.7000e+01, 2.0000e+01, 2.2000e+01, 2.9000e+01, 1.8000e+01,
       2.5000e+01, 9.0000e+00, 1.9000e+01, 1.8000e+01, 1.3000e+01,
       9.0000e+00, 1.7000e+01, 1.4000e+01, 1.2000e+01, 8.0000e+00,
       7.0000e+00, 1.3000e+01, 7.0000e+00, 1.0000e+01, 5.0000e+00,
       8.0000e+00, 5.0000e+00, 8.0000e+00, 3.0000e+00, 8.0000e+00,
       6.0000e+00, 9.0000e+00, 6.0000e+00, 6.0000e+00, 3.0000e+00,
       5.0000e+00, 3.0000e+00, 7.0000e+00, 4.0000e+00, 5.0000e+00,
       3.0000e+00, 1.0000e+00, 6.0000e+00, 4.0000e+00, 3.0000e+00,
       4.0000e+00, 4.0000e+00, 2.0000e+00, 2.0000e+00, 1.0000e+00,
       2.0000e+00, 2.0000e+00, 4.0000e+00, 0.0000e+00, 2.0000e+00,
       4.0000e+00, 1.0000e+00, 8.0000e+00, 0.0000e+00, 0.0000e+00,
       1.0000e+00, 3.0000e+00, 1.0000e+00, 3.0000e+00, 3.0000e+00,
       1.0000e+00, 0.0000e+00, 1.0000e+00, 3.0000e+00, 0.0000e+00,
       2.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
       0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00, 0.0000e+00,
       0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
       1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00,
       0.0000e+00, 0.0000e+00, 0.0000e+00, 1.0000e+00, 0.0000e+00]

Y_REG_MAG_BINS  = [0. , 0.00772506, 0.01545012, 0.02317518, 0.03090023,
       0.03862529, 0.04635035, 0.05407541, 0.06180047, 0.06952552,
       0.07725058, 0.08497564, 0.0927007 , 0.10042576, 0.10815082,
       0.11587588, 0.12360094, 0.13132599, 0.13905105, 0.14677611,
       0.15450117, 0.16222623, 0.16995129, 0.17767635, 0.18540141,
       0.19312647, 0.20085153, 0.20857657, 0.21630163, 0.22402669,
       0.23175175, 0.23947681, 0.24720187, 0.25492692, 0.26265198,
       0.27037704, 0.2781021 , 0.28582716, 0.29355222, 0.30127728,
       0.30900234, 0.3167274 , 0.32445246, 0.33217752, 0.33990258,
       0.34762764, 0.3553527 , 0.36307776, 0.37080282, 0.37852788,
       0.38625294, 0.393978  , 0.40170306, 0.40942812, 0.41715315,
       0.42487821, 0.43260327, 0.44032833, 0.44805339, 0.45577845,
       0.46350351, 0.47122857, 0.47895363, 0.48667869, 0.49440375,
       0.50212878, 0.50985384, 0.5175789 , 0.52530396, 0.53302902,
       0.54075408, 0.54847914, 0.5562042 , 0.56392926, 0.57165432,
       0.57937938, 0.58710444, 0.5948295 , 0.60255456, 0.61027962,
       0.61800468, 0.62572974, 0.6334548 , 0.64117986, 0.64890492,
       0.65662998, 0.66435504, 0.6720801 , 0.67980516, 0.68753022,
       0.69525528, 0.70298034, 0.7107054 , 0.71843046, 0.72615552,
       0.73388058, 0.74160564, 0.7493307 , 0.75705576, 0.76478082,
       0.77250588, 0.78023094, 0.787956  , 0.79568106, 0.80340612,
       0.81113118, 0.81885624, 0.82658124, 0.8343063 , 0.84203136,
       0.84975642, 0.85748148, 0.86520654, 0.8729316 , 0.88065666,
       0.88838172, 0.89610678, 0.90383184, 0.9115569 , 0.91928196,
       0.92700702, 0.93473208, 0.94245714, 0.9501822 , 0.95790726,
       0.96563232, 0.97335738, 0.98108244, 0.9888075, 0.99653256]


def weighted_gaussian_kde(x_eval, data_points, weights, bandwidth):
    """
    Computes a differentiable, weighted Gaussian KDE.

    Args:
        x_eval (torch.Tensor): The points at which to evaluate the KDE (shape [m]).
        data_points (torch.Tensor): The locations of the empirical data (shape [n]).
        weights (torch.Tensor): The weights for each data point (shape [n]), 
                                (e.g., your 'y' tensor). Should sum to 1.
        bandwidth (float or torch.Tensor): The bandwidth (standard deviation) 
                                           of the Gaussian kernels.

    Returns:
        torch.Tensor: The estimated probability density at each x_eval point (shape [m]).
    """
    # Use broadcasting to compute the PDF of N(x_eval; data_points, bandwidth)
    # for all combinations.
    # x_eval shape: [m, 1]
    # data_points shape: [1, n]
    # bandwidth is a scalar
    
    # Create the normal distributions N(mu=data_points, sigma=bandwidth)
    # The 'loc' (mean) will be broadcast to shape [1, n]
    dist = torch.distributions.Normal(data_points.unsqueeze(0), bandwidth)
    
    # Evaluate the log_prob of all x_eval points under all distributions
    # x_eval.unsqueeze(1) has shape [m, 1]
    # This calculation broadcasts to shape [m, n]
    # log_probs[i, j] = log( N(x_eval[i]; data_points[j], bandwidth) )
    log_probs = dist.log_prob(x_eval.unsqueeze(1))
    
    # Exponentiate to get probabilities
    # probs shape: [m, n]
    probs = torch.exp(log_probs)
    
    # Apply the weights
    # probs shape [m, n], weights.unsqueeze(0) shape [1, n]
    # Broadcasting results in shape [m, n]
    weighted_probs = probs * weights.unsqueeze(0)
    
    # Sum along the data_points dimension (dim=1)
    # This is the final step: sum(w_i * N(x; x_i, h^2))
    pdf_values = torch.sum(weighted_probs, dim=1)
    
    return pdf_values

# # --- Example Usage ---

# # 1. Define the empirical distribution
# # Let's say our data is at points [1.0, 2.0, 5.0]
# data_points = torch.tensor(bins)

# # And 'y' (the weights) are [0.5, 0.3, 0.2]
# y_weights = torch.tensor(pmf)

# # Verify weights sum to 1
# # print(f"Weights sum: {y_weights.sum()}")

# # 2. Define bandwidth and evaluation points
# bandwidth = 0.001

# x_eval = torch.linspace(0.0, 1.0, 100, requires_grad=True)

# # 3. Compute the KDE
# pdf_values = weighted_gaussian_kde(x_eval, data_points, y_weights, bandwidth)

# # # Calculate a scalar loss (e.g., the mean of the PDF)
# # scalar_output = pdf_values.mean()

# # # Run backward pass
# # scalar_output.backward()

# # plt.plot(x_eval.detach().numpy(), pdf_values.detach().numpy());
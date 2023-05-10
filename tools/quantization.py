import numpy as np

# Quantize the array values using a uniform quantization. The number of quantization levels equals 2^num_bits
def bit_quantization(array, num_bits):
    max_value = np.max(array)
    min_value = np.min(array)
    step_size = (max_value - min_value) / (2 ** num_bits - 1)
    quantized_array = np.round((array - min_value) / step_size) * step_size + min_value
    return quantized_array
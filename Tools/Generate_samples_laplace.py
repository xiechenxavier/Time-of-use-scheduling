import json
import numpy as np

# Function to generate samples from Laplace distribution
def generate_samples(mu, sigma, num_samples=1000):
    return np.random.laplace(mu, sigma, num_samples)

def main():
    # Load input data from JSON file
    input_filename = "E:/ToU重做3/dict_experiment_data/D_0.01/alpha=0.1/9_81.json"
    with open(input_filename, 'r') as json_file:
        data = json.load(json_file)
    
    # Extract mu and sigma values
    mu_values = data["mu_{i,j}"]
    sigma_values = data["sigma_{i,j}"]
    
    # Get dimensions m and n
    m = len(mu_values)
    n = len(mu_values[0]) if m > 0 else 0

    # Create an empty list to store the generated samples
    samples = []

    # Generate 1000 samples for each (i, j) pair
    num_samples = 1000
    for i in range(m):
        row_samples = []
        for j in range(n):
            mu_ij = mu_values[i][j]
            sigma_ij = sigma_values[i][j]
            samples_ij = generate_samples(mu_ij, sigma_ij, num_samples)
            row_samples.append(samples_ij)
        samples.append(row_samples)

    # Convert the samples list to a numpy array of shape (1000, m, n)
    samples_array = np.array(samples).transpose(2, 0, 1)

    # Save the generated samples to a new JSON file
    output_filename = "E:/ToU重做3/dict_experiment_data/D_0.01/Generate_samples/9_81_samples.json"
    output_data = {"samples": samples_array.tolist()}
    with open(output_filename, 'w') as json_file:
        json.dump(output_data, json_file, indent=4)

    print(f"Generated samples saved to {output_filename}")

if __name__ == "__main__":
    main()
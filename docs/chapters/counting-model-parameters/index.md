# Counting Model Parameters

Many discussion on deep learning models describe a model based on how
many "parameters" it has.  The bigger the model, the more parameters
it has.  But what exactly is a model parameter and how do we
calculate the number of parameters for a deep neural network?

Let's take a look at the formula for calculating the total parameters in a fully connected neural network:

1. For weights:
   - Each connection between two layers needs a weight parameter
   - For each pair of adjacent layers: (nodes in previous layer × nodes in current layer)
   - Since input layer doesn't have incoming connections, we multiply by (numLayers - 1)
   - Total weight parameters = neuronsPerLayer × neuronsPerLayer × (numLayers - 1)

2. For biases:
   - Each neuron (except in input layer) has one bias parameter
   - Total bias parameters = neuronsPerLayer × (numLayers - 1)

3. Total parameters = Total weight parameters + Total bias parameters

For example, in a network with 3 layers (including input) and 4 neurons per layer:

- Weight parameters: 4 × 4 × (3-1) = 32
- Bias parameters: 4 × (3-1) = 8
- Total parameters: 32 + 8 = 40



We have created an interactive visualization where you can:
1. Adjust the number of layers (2-7)
2. Adjust the neurons per layer (2-10)
3. See the network structure
4. Get a breakdown of weight parameters, bias parameters, and total parameters

Would you like me to add any additional features to the visualization?
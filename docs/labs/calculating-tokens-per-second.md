# Calculating Tokens Per Second

How quickly a model returns text is a key metric.  Here is a sample
program that calculates the number of tokens per second for Deepseek-r1:7b
running in an [Ollama](../glossary.md#ollama) framework.  This test was run on my
local GPU which is a NVIDIA RTX 2080 Ti with 12GB RAM running CUDA 12.6.  The
size of the model was 4.7GB which fits well within the 12GB ram of the GPU.

To time the performance of a model we do the following:

### 1. Record the time before the model runs with 

```python
start_time = time.time()
```

### 2. Record the end time and calculate the elapsed time

```python
end_time = time.time()
elapsed_time = end_time - start_time
```

### 3. Count the total number of tokens in the result and calculate the tokens per second

```python
token_count = len(result.split())
tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0
```

## Complete Program

```python
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Define your prompt template.
template = """Question: {question}

Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

# Initialize the Ollama model.
model = OllamaLLM(model="llama3.1")

# Chain the prompt and the model.
chain = prompt | model

# Record the start time.
start_time = time.time()

# Invoke the chain with your question.
result = chain.invoke({"question": "What are the 50 most imporant concepts in a college level course on deep learning?"})

# Record the end time.
end_time = time.time()
elapsed_time = end_time - start_time

# Count tokens in the output.
# Here we assume tokens are separated by whitespace.
# (For more accurate token counts, consider using the model's specific tokenizer.)
token_count = len(result.split())

# Calculate tokens per second.
tokens_per_second = token_count / elapsed_time if elapsed_time > 0 else 0

# Print the result and statistics.
print("Result:")
print(result)
print("\nStatistics:")
print(f"Token Count: {token_count}")
print(f"Time Elapsed: {elapsed_time:.2f} seconds")
print(f"Tokens per Second: {tokens_per_second:.2f}")
```

## Result

What a fascinating question!

To tackle this, let's break down the process into manageable steps. Here's a suggested approach:

### **Step 1: Identify Key Topics in Deep Learning**

At a high level, deep learning is a subset of machine learning that focuses on complex models inspired by the structure and function of the brain. The field encompasses various techniques for tasks such as image classification, object detection, speech recognition, natural language processing, and more.

Some key topics to consider when identifying important concepts include:

1. Neural network architectures (e.g., convolutional neural networks, recurrent neural networks)
2. Deep learning frameworks (e.g., TensorFlow, PyTorch)
3. Optimization algorithms (e.g., stochastic gradient descent, Adam)
4. Regularization techniques (e.g., dropout, L1 and L2 regularization)

### **Step 2: Consider Mathematical Foundations**

Deep learning relies heavily on mathematical concepts from linear algebra, calculus, probability theory, and statistics.

Some essential math concepts to cover include:

1. Vector spaces and transformations
2. Matrix operations (e.g., multiplication, inversion)
3. Eigenvalue decomposition and singular value decomposition
4. Probability distributions (e.g., Gaussian, Bernoulli)
5. Expectation and variance

### **Step 3: Explore Practical Aspects of Deep Learning**

This includes the implementation, evaluation, and deployment of deep learning models.

Some important practical concepts to consider:

1. Data preprocessing and augmentation techniques
2. Model selection and hyperparameter tuning
3. Overfitting and regularization strategies
4. Model interpretability and explainability
5. Deployment and scalability in production environments

### **Step 4: Review Advanced Topics and Emerging Trends**

As the field of deep learning continues to evolve, it's essential to stay up-to-date with recent developments.

Some advanced topics and emerging trends include:

1. Transfer learning and pre-trained models
2. Adversarial attacks and defenses
3. Graph neural networks and graph-based techniques
4. Attention mechanisms and transformer architectures
5. Quantum computing and its potential applications in deep learning

### **Step 5: Refine the List of Important Concepts**

Considering the topics identified above, let's prioritize them based on their relevance to a college-level course on deep learning.

Here are the top 50 most important concepts:

### **Mathematical Foundations (1-10)**

1. Vector spaces and transformations
2. Matrix operations (e.g., multiplication, inversion)
3. Eigenvalue decomposition and singular value decomposition
4. Probability distributions (e.g., Gaussian, Bernoulli)
5. Expectation and variance
6. Calculus basics (e.g., gradients, Hessians)
7. Linear algebra review (e.g., determinants, eigenvectors)
8. Probability and statistics review
9. Optimization algorithms (e.g., gradient descent)
10. Regularization techniques (e.g., L1, L2)

### **Neural Network Architectures (11-20)**

11. Convolutional neural networks (CNNs)
12. Recurrent neural networks (RNNs)
13. Long short-term memory (LSTM) networks
14. Gated recurrent units (GRUs)
15. Residual connections and skip connections
16. Autoencoders and variational autoencoders
17. U-Net architectures for image segmentation
18. Transformers and self-attention mechanisms
19. Graph neural networks (GNNs)
20. Other specialized architectures (e.g., capsule networks)

### **Deep Learning Frameworks and Tools (21-25)**

21. TensorFlow and Keras APIs
22. PyTorch and Lightning-PyTorch
23. Deep learning frameworks for GPU acceleration (e.g., cuDNN)
24. Model serving and deployment tools (e.g., Docker, Kubernetes)
25. Deep learning software development kits (SDKs) and libraries

### **Practical Aspects of Deep Learning (26-35)**

26. Data preprocessing and augmentation techniques
27. Model selection and hyperparameter tuning
28. Overfitting and regularization strategies
29. Model interpretability and explainability
30. Deployment and scalability in production environments
31. Model evaluation metrics (e.g., accuracy, precision)
32. Common pitfalls and debugging techniques
33. Data efficiency and transfer learning strategies
34. Regularization techniques for large models
35. Distributed training and parallelization

### **Advanced Topics and Emerging Trends (36-50)**

36. Transfer learning and pre-trained models
37. Adversarial attacks and defenses
38. Graph neural networks and graph-based techniques
39. Attention mechanisms and transformer architectures
40. Quantum computing and its potential applications in deep learning
41. Exponential family distributions and link functions
42. Causal inference and counterfactual reasoning
43. Generative models (e.g., GANs, VAEs)
44. Time series analysis with LSTM networks
45. Text classification and sentiment analysis
46. Image recognition and object detection
47. Speech recognition and natural language processing
48. Reinforcement learning and deep Q-networks (DQN)
49. Multi-agent systems and distributed decision-making
50. Explainability techniques for complex models

Of course, this list is not exhaustive, but it should give you a solid starting point for creating a comprehensive college-level course on deep learning.

How's that?

## Inference Statistics

**Token Count:** 687<br/>
**Time Elapsed:** 13.33 seconds<br/>
**Tokens per Second:** 51.55<br/>

!!! note
    Depending on your model's tokenization, you might need a more precise token counter   (e.g., using the `tiktoken` library for models like GPT).

## Model Metadata

Knowing about the structure of a model is key to understanding its performance.

Here is the information that ollama provided about deepseek-r1:

```sh
$ ollama show deepseek-r1
  Model                          
  	arch            	qwen2 	         
  	parameters      	7.6B  	         
  	quantization    	Q4_K_M	         
  	context length  	131072	         
  	embedding length	3584  	         
```

Let's do a deep dive into each of these model metadata fields.

### 1.  **arch (Architecture):**

-   **What it means:** This parameter indicates the underlying neural network architecture on which the model is based.
-   **In this case:** The model uses the **qwen2** architecture. This tells you which design or blueprint the model follows (e.g., similar to transformer-based architectures like GPT or BERT), which influences how it processes input data and generates responses.

### 2.  **parameters (Number of Parameters):**

- **What it means:** This shows the total number of learnable weights (and biases) in the model. The size of this number is often used as a rough proxy for the model's capacity to learn and represent complex patterns.
-  **In this case:** The model has **7.6B** (7.6 billion) parameters. More parameters generally mean a higher capacity model, though they also require more memory and computational resources during inference.

### 3.  **quantization:**

- **What it means:** Quantization refers to reducing the numerical precision of the model's parameters. This process converts high-precision weights (e.g., 32-bit floats) into lower-precision representations (e.g., 4-bit integers) to reduce model size and speed up computations with a minimal loss in accuracy.
-  **In this case:** The value **Q4\_K\_M** indicates that a 4-bit quantization scheme is used. The "Q4" part tells you that the weights are represented with 4-bit precision, and "K\_M" likely refers to the specific quantization method or variant implemented. This balance helps the model run more efficiently while retaining as much performance as possible.

### 4.  **context length:**

- **What it means:** This parameter defines the maximum number of tokens the model can process in a single input prompt (or conversation). In transformer-based models, the context length determines how much text the model can consider at one time.
- **In this case:** The model can handle a context of up to **131072** tokens. This is an exceptionally long context compared to most language models, which typically support only a few thousand tokens. It enables the model to process very large documents or maintain extended conversations.

5.  **embedding length:**

- **What it means:** This is the size (or dimensionality) of the vector used to represent each token in the model's internal computations. In other words, every token in the input is mapped to a vector of this length, which the model uses to capture semantic and syntactic information.
- **In this case:** An embedding length of **3584** means that each token is converted into a 3584-dimensional vector. A higher embedding dimension can allow for richer representations but also increases the model's computational complexity.

### Summary

- **Architecture (arch):** Defines the model's design (here, **qwen2**).
- **Parameters:** Indicates the model's size in terms of learnable weights (**7.6B** parameters).
- **Quantization:** Shows how the model's weights are stored (using **4-bit precision** with the specific scheme **Q4\_K\_M**).
- **Context Length:** The maximum number of tokens the model can process at once (**131072** tokens).
- **Embedding Length:** The dimensionality of token representations within the model (**3584**).

Each of these parameters provides insight into the model's design, capacity, efficiency, and the scale at which it can process input data.
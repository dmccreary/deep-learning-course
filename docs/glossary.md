# Glossary of Terms

Below is an alphabetical glossary of 200 key concepts, each defined according to ISO 11179 guidelines. When relevant, an example of how the term is used in a Deep Learning context is provided.  A term definition is considered to be consistent with ISO metadata registry guideline 11179 if it meets the following criteria:

1. Precise
2. Concise
3. Distinct
4. Non-circular
5. Unencumbered with business rules

#### Accuracy
A performance metric that measures the proportion of correct predictions out of all predictions made by a model.  
**Example:** In a binary classification problem, accuracy is the ratio of true positives and true negatives to the total sample size.

#### Activation Functions
Mathematical functions applied to neuron outputs that introduce non-linearity into a neural network.  
**Example:** ReLU and Sigmoid are common activation functions used to help networks learn complex patterns.

#### Actor-Critic
A class of reinforcement learning algorithms that maintain both a policy (actor) and a value function (critic) to guide action selection.  
**Example:** The actor-critic approach is used to stabilize training by combining policy gradient methods with value-based techniques.

#### AdaGrad
An adaptive gradient algorithm that individually scales learning rates based on the historical gradient magnitudes for each parameter.  
**Example:** AdaGrad can help train models more efficiently in scenarios where some parameters require more frequent updates than others.

#### Adam Optimizer
An optimization algorithm that combines ideas from momentum (exponential moving averages of gradients) and AdaGrad (adaptive learning rates).  
**Example:** Adam is often the default choice for training deep neural networks due to its speed and stable convergence properties.

#### Adversarial Examples
Inputs to a model that have been intentionally crafted to cause incorrect predictions, despite appearing normal to humans.  
**Example:** Slightly perturbed images that force a well-trained image classifier to misclassify a stop sign as a yield sign.

#### Agile Methods
Project management practices emphasizing iterative development, collaboration, and adaptability.  
**Example:** In a machine learning project, sprints are used to rapidly prototype models and incorporate stakeholder feedback.

#### AI Branches
Various subfields of Artificial Intelligence, including symbolic AI, data-driven AI, evolutionary computation, and others.  
**Example:** A course might compare symbolic reasoning systems with modern deep learning architectures.

#### AI Definition
The study and development of machines capable of performing tasks that typically require human intelligence.  
**Example:** AI encompasses problem-solving, perception, language understanding, and learning processes.

#### Attention

The area of a subject that has a high weight when classifying the subject.

The paper "Attention is All You Need" was the name of the landmark paper on Transformers.

#### AutoML
Automated Machine Learning techniques that handle tasks such as model selection, hyperparameter tuning, and feature engineering with minimal human intervention.  
**Example:** Tools like Google AutoML or AutoKeras can automatically try multiple pipelines and present the best-performing model.

#### Autoencoder DR
Dimensionality reduction technique that uses an autoencoder neural network to learn a compressed representation of data.  
**Example:** An autoencoder might reduce a 1000-dimensional dataset down to 50 dimensions for visualization.

#### Autoencoders (AE)
Neural network architectures designed to learn a compressed representation (encoding) of input data, then reconstruct it (decoding).  
**Example:** AEs are used to denoise images or perform feature extraction for anomaly detection.

#### Autonomous Driving
The application of AI/ML methods to enable vehicles to sense their environment and operate without human intervention.  
**Example:** Deep neural networks can process camera and lidar data in real-time for self-driving car navigation.

#### AUC-ROC
A performance metric summarizing the trade-off between true positive rate and false positive rate across different thresholds.  
**Example:** The area under the ROC curve (AUC) is often used to compare classifiers independent of a specific decision threshold.

#### Backprop Through Time (BPTT)
A method for training recurrent neural networks by unrolling the network for a certain number of timesteps and applying backpropagation.  
**Example:** LSTM networks rely on BPTT to learn dependencies in sequential data such as text or time series.

#### Backpropagation
The algorithm used to calculate gradients of error with respect to each weight in a neural network, enabling efficient training.  
**Example:** During backprop, partial derivatives of the loss function are propagated backward to update model parameters.

#### Baseline Algorithms
Simple or well-understood methods used as reference points for performance comparison.  
**Example:** A linear regression might serve as a baseline for a more complex deep model on a regression task.

#### Batch Normalization
A technique to normalize intermediate layer outputs to accelerate training and stabilize learning in neural networks.  
**Example:** By normalizing activations, batch norm helps reduce internal covariate shift.

#### Batch Training
A training paradigm where the model parameters are updated after processing an entire batch of training examples.  
**Example:** Large datasets are often split into batches to efficiently utilize GPU memory.

#### BatchNorm
Another name for batch normalization, focusing on normalizing activations per mini-batch.  
**Example:** “BatchNorm” layers are common in architectures like ResNet and Inception.

#### Bayesian Optimization
A global optimization strategy using Bayesian inference to intelligently sample hyperparameters and find optimal configurations.  
**Example:** Given a function that’s expensive to evaluate (e.g., a deep network training loop), Bayesian optimization tries to minimize the total evaluations required.

#### BERT Overview
Description of Bidirectional Encoder Representations from Transformers, a popular transformer-based NLP model that learns context in both directions.  
**Example:** BERT is used to improve tasks like question-answering and sentence classification.

#### Bias Mitigation
Techniques to reduce or remove unwanted bias in AI models, ensuring fair and equitable outcomes.  
**Example:** Reweighting training data to address underrepresentation of certain groups.

#### Bias-Variance Tradeoff
The balance between the accuracy of a model (bias) and its sensitivity to small fluctuations in the training set (variance).  
**Example:** A highly complex model might overfit (low bias, high variance), while a simple model might underfit (high bias, low variance).

#### Biological Inspiration
The design of AI techniques based on biological systems or processes, such as neural networks modeled after the human brain.  
**Example:** Convolutional neural networks are partially inspired by the visual cortex in mammals.

#### BLEU Score
Bilingual Evaluation Understudy; a metric used to evaluate the quality of machine-translated text against reference translations.  
**Example:** BLEU is commonly reported in NLP tasks such as neural machine translation experiments.

#### Capsule Networks
Neural architectures that group neurons into “capsules” to better capture positional and hierarchical relationships in data.  
**Example:** Capsule networks aim to preserve spatial relationships more effectively than traditional CNNs.

#### ChatGPT Concept
A high-level idea of OpenAI’s ChatGPT, which uses large language models to generate context-aware text responses.  
**Example:** Students explore ChatGPT’s approach to answer generation and how it leverages conversational context.

#### Cloud Services
Remote computing resources (servers, storage, APIs) for hosting, training, and deploying AI models.  
**Example:** AWS, Azure, and Google Cloud offer GPU/TPU instances to accelerate model training.

#### CNN Architectures
Convolutional Neural Network designs specialized for handling spatial data such as images.  
**Example:** Classic architectures include LeNet, AlexNet, VGG, ResNet, and DenseNet.

#### Code Review
The process of examining and discussing code changes to improve code quality and share knowledge within a team.  
**Example:** Before merging new data preprocessing scripts, peers review for correctness and maintainability.

#### Computer Vision
A field of AI focusing on enabling machines to interpret and understand visual information from the world.  
**Example:** Tasks include image classification, object detection, and image segmentation.

#### Conditional GAN
A GAN variant where both the generator and discriminator receive additional context information, such as class labels.  
**Example:** Conditional GANs can generate images conditioned on textual descriptions like “cats” or “dogs.”

#### Confusion Matrix
A tabular layout displaying the number of correct and incorrect predictions grouped by class.  
**Example:** Rows typically represent actual classes, while columns represent predicted classes, making misclassifications visible.

#### Conda Environment
A virtual environment manager provided by Anaconda, allowing users to isolate Python packages and dependencies.  
**Example:** Students create a separate conda environment to manage a specific ML project’s libraries without interfering with system Python.

#### Convolution Operation
A mathematical operation where a filter (kernel) is applied to input data (e.g., an image) to detect features.  
**Example:** In CNNs, convolution layers learn filters that activate in response to relevant patterns.

#### Cross-Validation
A technique for evaluating model performance by partitioning data into multiple folds, cycling through training and validation sets.  
**Example:** k-fold cross-validation helps assess how well the model generalizes to unseen data.

#### CycleGAN
A GAN-based architecture for unpaired image-to-image translation.  
**Example:** CycleGAN can transform horse images into zebra images without needing paired training examples.

#### Data Acquisition
The process of gathering relevant datasets from various sources, such as APIs, sensors, or public repositories.  
**Example:** In a project, data acquisition might involve scraping websites or querying a medical imaging database.

#### Data Annotation
Labeling or tagging raw data (images, text, audio) with meaningful information to facilitate supervised learning.  
**Example:** Crowdworkers often annotate bounding boxes on images to prepare object detection datasets.

#### Data Augmentation
Techniques used to increase the diversity of a training set by applying transformations (e.g., flips, rotations) to existing samples.  
**Example:** Randomly flipping or cropping images to help CNNs generalize better.

#### Data Cleaning
The process of identifying and correcting incomplete, inaccurate, or irrelevant parts of a dataset.  
**Example:** Removing duplicates or fixing inconsistent labels in a CSV file before model training.

#### Data-Driven AI
AI methodologies that rely heavily on large datasets and statistical learning rather than explicit symbolic rules.  
**Example:** Deep learning is a classic form of data-driven AI, requiring abundant labeled data for training.

#### Data Ethics
The standards and principles guiding the responsible collection, storage, and use of data.  
**Example:** Ensuring personal data is handled with user consent and compliance with legal frameworks such as GDPR.

#### Data Imbalance
A condition where classes or categories in a dataset are not equally represented.  
**Example:** In fraud detection, “fraud” cases are often far fewer than “non-fraud” cases, impacting model training.

#### Data Parallelism
Distributing subsets of the input data across multiple processors or machines to train a model faster.  
**Example:** Large mini-batches are split across GPUs, and gradients are aggregated at each step.

#### Data Preprocessing
Preparatory steps taken on raw data, such as cleaning, normalization, or encoding.  
**Example:** Converting categorical variables into one-hot vectors or normalizing images to zero mean and unit variance.

#### Data Profile

A summary of key statistical characteristics and quality metrics of a dataset, including distributions, missing values, cardinality, and data types for each feature.

**Example:** An MNIST data profile would reveal that each image is 28x28 pixels (784 features), pixel values range from 0-255, there are 10 distinct label classes with roughly equal distribution (~6000 samples each), and the dataset contains no missing values.  See 

#### Data Splits
Partitioning data into subsets (e.g., training, validation, test) for model development and evaluation.  
**Example:** 80% of the data for training, 10% for validation, 10% for testing.

#### Data Visualization
Graphical representation of data to reveal patterns, trends, or outliers.  
**Example:** Creating scatter plots in Matplotlib to explore relationships between features.

#### Data Visualization Tools
Software libraries and platforms that facilitate the creation of charts, graphs, and interactive plots.  
**Example:** Matplotlib, Seaborn, and Plotly are used in notebooks to illustrate key data insights.

#### Deep Learning
Neural network designs featuring multiple hidden layers, enabling hierarchical feature extraction.

Deep Learning is a subset of machine learning that focus
on complex models.  Deep learning is used for tasks such as image classification, object detection, speech recognition, natural language processing and other problems that can't be solved by simpler models.

**Example:** A typical deep learning system might have a dozen or more layers in a CNN.

#### Deep Q-Network (DQN)
A reinforcement learning approach where a deep neural network approximates the Q-function for action selection.  
**Example:** DQN was famously applied by DeepMind to play Atari games at a superhuman level.

#### Deepseek

An open-source large-langauge model with reasoning announced in 2025 that has superior performance.

The Deepseek-R1 7B model is used on a local GPU for testing the [tokens-per-second](./labs/calculating-tokens-per-second.md) lab in this course.


#### DenseNet
A CNN architecture where each layer is connected to every other layer in a feed-forward manner.  
**Example:** DenseNet alleviates the vanishing gradient problem by encouraging feature reuse.

#### Depth Vs Width
The tradeoff between making a network deeper (more layers) or wider (more neurons per layer).  
**Example:** Deeper networks can learn more abstract features, whereas wider networks can capture more detail at each level.

#### Denoising Diffusion
A class of generative models where noise is incrementally added to data, then learned in reverse to generate samples.  
**Example:** Used in image generation tasks to progressively refine noise into realistic images.

#### Diffusion Models
Probabilistic models that learn to reverse a gradual noising process to produce new samples.  
**Example:** DALL·E 2 incorporates diffusion methods for high-quality image synthesis.

#### Dimensionality Concepts
Principles and considerations related to the number of features or variables in a dataset (dimensionality).  
**Example:** High-dimensional data can be more difficult to visualize and may require techniques like PCA.

#### Documentation
Written guides, explanations, or references describing software or processes.  
**Example:** Thorough documentation ensures that future team members can understand the codebase and its usage.

#### Drug Discovery
The application of AI to identify potential new medications by analyzing large chemical and biological datasets.  
**Example:** ML models predict molecular binding affinities to reduce trial-and-error in drug design.

#### Dropout
A regularization method that randomly “drops” or sets some neurons to zero during training to reduce overfitting.  
**Example:** Setting a 50% dropout rate in a dense layer to encourage robust feature learning.

#### Early Stopping
A regularization technique that halts training when validation performance stops improving.  
**Example:** Prevents overfitting by not allowing the model to train excessively on one dataset.

#### Edge Deployment
Running AI models directly on edge devices (smartphones, IoT devices) rather than on centralized servers.  
**Example:** A small CNN for object recognition deployed on a mobile phone for real-time inference.

#### Energy-Based Models
A class of probabilistic models that define an energy function over configurations, and learning involves shaping this energy landscape.  
**Example:** Boltzmann machines are one type of energy-based model used for representation learning.

#### Evaluation Metrics
Quantitative measures to assess model performance, guiding model selection and tuning.  
**Example:** Accuracy, precision, recall, and F1 score are common metrics for classification tasks.

#### Explainable AI (XAI)
Methods designed to make AI system decisions interpretable by humans.  
**Example:** Feature attribution maps that highlight which parts of an image influenced a CNN’s classification.

#### Exploration Vs Exploitation
The balance in reinforcement learning between trying new actions (exploration) and using known rewarding actions (exploitation).  
**Example:** An RL agent might explore different states early on, then exploit the best actions discovered.

#### Exploding Gradient Control
Techniques to prevent gradients from becoming excessively large during backpropagation.  
**Example:** Gradient clipping is a common way to avoid unstable updates in RNN training.

#### Exploding Gradients
A situation where gradients grow uncontrollably during training, causing large parameter updates and potential instability.  
**Example:** Deep RNNs may experience exploding gradients unless special measures (like clipping) are applied.

#### F1 Score
The harmonic mean of precision and recall, providing a single measure of a test’s accuracy.  
**Example:** Used when you care about both false positives and false negatives in an imbalanced classification.

#### Fairness
The principle of designing AI systems to avoid discriminatory outcomes or biased treatment.  
**Example:** Ensuring loan approval models do not systematically disadvantage applicants from certain demographics.

#### Feature Engineering
The process of creating or transforming input features to improve model performance.  
**Example:** Combining multiple text columns into a single normalized “bag of words” vector.

#### Feature Space
The multidimensional space where each dimension corresponds to a feature of the data.  
**Example:** Visualizing points in feature space helps understand how the model separates classes.

#### Filters And Kernels
Learnable convolution operators that detect features in images or other spatial data.  
**Example:** Early layers might learn edge detectors, while deeper layers capture more complex patterns.

#### Financial Forecasting
Using AI/ML to predict market movements, asset prices, or economic indicators.  
**Example:** LSTM networks analyzing historical stock data for next-day price predictions.

#### Flow-Based Models
Generative models that transform noise into data samples via a series of invertible transformations.  
**Example:** RealNVP or Glow architectures produce exact likelihood estimates and allow sampling.

#### Forward Propagation
The process of passing input data through a neural network to get an output prediction.  
**Example:** In a feedforward network, data flows from the input layer through hidden layers to the output.

#### Frozen in Time
The term used to describe that the knowledge base that a LLM is trained on
has a cutoff date which can often omit current events in the past year.

#### Fully Connected Layers
Layers where every neuron is connected to every neuron in the next layer, typically appearing after convolution blocks.  
**Example:** After convolution and pooling, the extracted features might go into a fully connected classifier.

#### GAN Basics
The foundational idea of Generative Adversarial Networks, involving a generator and a discriminator in a minimax game.  
**Example:** A generator tries to create realistic images, while the discriminator attempts to distinguish them from real images.

#### Generative Models
Models that learn the joint probability distribution of data, enabling them to generate new, synthetic samples.  
**Example:** Models like GANs, VAEs, and diffusion models can create realistic images or text.

#### Generator Vs Discriminator
Two components of a GAN: the generator synthesizes data, and the discriminator classifies whether data is real or generated.  
**Example:** Training alternates between improving the generator’s realism and tightening the discriminator’s detection.

#### GPT Overview
Highlights of the Generative Pre-trained Transformer series, focusing on autoregressive language modeling at scale.  
**Example:** GPT-3 can generate coherent paragraphs of text based on a given prompt.

#### Graph Neural Networks (GNNs)
Networks designed to process graph-structured data, learning node or edge representations through message passing.  
**Example:** GNNs can predict molecular properties by treating atoms as nodes and bonds as edges.

#### Gradient Descent
An iterative optimization method that updates parameters in the opposite direction of the gradient of the loss function.  
**Example:** Simple gradient descent uses the entire dataset to compute gradients each iteration.

#### GRU Units
Gated Recurrent Unit cells that manage hidden state transitions without a separate cell state.  
**Example:** GRUs are often computationally simpler yet comparable to LSTMs for sequence tasks.

#### Grid Search
A hyperparameter tuning method that exhaustively tries every combination of a specified parameter grid.  
**Example:** Searching over different learning rates and regularization strengths for logistic regression.

#### GPU Acceleration
Utilizing graphics processing units to speed up parallelizable operations in neural network training.  
**Example:** Matrix multiplications in backprop are greatly accelerated on GPUs.

#### He Initialization
Weight initialization method adapted for ReLU-like activation functions, aiming to maintain variance across layers.  
**Example:** Also known as Kaiming initialization, used to stabilize training in deep ReLU networks.

#### High-Dimensional Data
Data with many features, which can lead to the “curse of dimensionality” and sparse observations in feature space.  
**Example:** Text data with thousands of unique terms is inherently high-dimensional.

#### History Of AI
The chronological development and milestones of AI, from symbolic systems to modern deep learning breakthroughs.  
**Example:** Tracing from early logic-based AI in the 1950s to recent achievements in large-scale neural models.

#### Hyperparameter Tuning
The process of finding optimal values for parameters not directly learned during training, like learning rates or layer sizes.  
**Example:** Using cross-validation to compare multiple hyperparameter configurations.

#### ImageNet
A large-scale image dataset widely used as a benchmark for deep learning, especially in computer vision.  
**Example:** Models that excel on ImageNet often generalize well to a range of vision tasks.

#### Inception Modules
Architectural blocks that perform convolutions of different sizes in parallel, then concatenate outputs.  
**Example:** GoogLeNet (Inception v1) introduced inception modules for more efficient resource usage.

#### Initialization Methods
Techniques for setting initial weight values in neural networks to aid stable convergence.  
**Example:** He or Xavier initialization is often chosen based on the activation function used.

#### Jupyter Notebooks
Interactive web-based computational tools that combine code, visualizations, and text.  
**Example:** Commonly used in teaching ML, allowing students to experiment step by step and visualize results.

#### Key Figures
Influential researchers or pioneers in AI who contributed foundational theories or breakthroughs.  
**Example:** Alan Turing, Marvin Minsky, Geoffrey Hinton, Yoshua Bengio, and Yann LeCun.

#### Language Modeling
Predicting the likelihood of a sequence of words, forming the basis for many NLP tasks.  
**Example:** A language model might predict the next word in a sentence or evaluate the fluency of generated text.

#### Large Language Models (LLMs)
Very large neural network-based models trained on massive text corpora to perform complex language tasks.  
**Example:** GPT-3, BERT, and PaLM can perform text generation, question answering, and zero-shot tasks.

#### Latent Space
The lower-dimensional internal representation of data learned by a model, especially in generative methods.  
**Example:** In a VAE, points in the latent space can be sampled and decoded into new images.

#### Layer Stacking
Arranging multiple layers in a network to build deeper models capable of complex feature extraction.  
**Example:** Adding layers in a CNN to capture higher-level abstractions of the input.

#### Learning Rate
A hyperparameter controlling the step size in gradient-based optimization.  
**Example:** A learning rate that’s too high might overshoot minima, while one that’s too low could prolong training.

#### Learning Rate Scheduling
Strategies to adjust the learning rate over time, typically reducing it as training progresses.  
**Example:** Step decay, exponential decay, and warm restarts are all scheduling techniques.

#### LIME
Local Interpretable Model-agnostic Explanations; explains predictions of any classifier by approximating it locally with an interpretable model.  
**Example:** LIME can highlight text snippets that most influenced a sentiment classifier’s decision.

#### LSTM Units
Long Short-Term Memory cells that maintain and control access to an internal cell state for long-range dependencies in sequences.  
**Example:** LSTMs are popular in NLP for capturing context across lengthy texts.

#### Likelihood Estimation
The process of determining parameters of a probabilistic model by maximizing the likelihood of observed data.  
**Example:** In generative models, training often involves maximizing log-likelihood.

#### Local Minima
Points in the loss landscape where no small move decreases the loss, but they may not be the global optimum.  
**Example:** Neural networks often rely on large parameter spaces that can have many local minima.

#### Loss Functions
Mathematical functions quantifying the difference between predictions and targets, guiding parameter updates.  
**Example:** Mean Squared Error is a loss function for regression; Cross-Entropy is common for classification.

#### Loss Surface
The multidimensional space defined by network parameters on which the loss function value is plotted.  
**Example:** Gradient descent methods navigate the loss surface to find minima.

#### Manifold Hypothesis
The assumption that high-dimensional data points lie on lower-dimensional manifolds embedded in the input space.  
**Example:** Techniques like t-SNE or UMAP attempt to uncover these manifolds for visualization.

#### Markov Decision Process (MDP)
A mathematical framework for sequential decision-making, characterized by states, actions, rewards, and transition probabilities.  
**Example:** Reinforcement learning often models environments as MDPs for algorithmic exploration.

#### Matplotlib Basics
Fundamental features of Matplotlib, a Python library for creating static, animated, and interactive visualizations.  
**Example:** Plotting histograms, line graphs, and scatter plots to understand dataset distributions.

#### Mean Squared Error (MSE)
A regression loss function that averages the squared differences between predictions and targets.  
**Example:** MSE is minimized during training of linear regression or certain autoencoders.

#### Medical Imaging
The application of AI to interpret and analyze medical scans such as MRIs, CTs, and X-rays.  
**Example:** CNNs detecting tumors in MRI scans to assist radiologists.

#### Memory Management
Techniques to efficiently allocate and handle data in GPU/CPU memory during model training.  
**Example:** Gradient checkpointing helps reduce memory usage in very deep networks.

#### Mini-Batch Training
A compromise between batch and stochastic training, updating parameters after a small subset of samples.  
**Example:** Common in practice for balancing computational efficiency with stable gradient estimates.

#### ML Definition
A subfield of AI that focuses on algorithms learning patterns from data rather than being explicitly programmed.  
**Example:** Classification and regression tasks where models improve their performance as they process more data.

#### MLOps
A set of practices for deploying and maintaining machine learning models in production, analogous to DevOps.  
**Example:** Automated CI/CD pipelines that retrain and redeploy models as new data arrives.

#### Model Compression
Techniques to reduce model size and inference costs, such as pruning, quantization, or knowledge distillation.  
**Example:** Compressing a large CNN so it can run on edge devices with limited memory.

#### Model Interpretability
The clarity with which a human can understand a model’s internal processes and decisions.  
**Example:** Using saliency maps to see which pixels in an image were key for a CNN’s classification.

#### Model Parallelism
Splitting a model’s layers or parameters across different computational units.  
**Example:** Large language models can be distributed across multiple GPUs, each storing only part of the network.

#### Momentum Optimizer
An extension of gradient descent that accumulates velocity from past updates to damp oscillations and accelerate learning.  
**Example:** Typically combined with a learning rate schedule to converge faster.

#### Multi-Layer Perceptron
A fully connected feedforward network with one or more hidden layers.  
**Example:** Often used as a baseline for tabular data classification or regression tasks.

#### Natural Language Processing (NLP)
AI techniques enabling machines to understand, interpret, and generate human language.  
**Example:** Chatbots, machine translation, and sentiment analysis are common NLP applications.

#### Neural Network
A network of calculations designed to mimic the neurons in the human brain.

#### Neural Machine Translation
An approach to automated translation that uses deep neural networks, often encoder-decoder architectures.  
**Example:** Systems like Google Translate or DeepL rely on neural machine translation to handle multiple languages.

#### Neural Network Origins
The historical foundation of neural networks, tracing back to early perceptron models and Hebbian learning.  
**Example:** The perceptron’s creation in the 1950s laid groundwork for modern deep learning.

#### Nobel Prizes
Prestigious awards that, while not commonly granted specifically for AI, have occasionally been given for foundational contributions relevant to AI fields.  
**Example:** John Nash’s work in game theory influenced multi-agent AI, recognized with a Nobel in Economics.

#### Nonlinear Embeddings
Mapping high-dimensional data into a lower-dimensional space using nonlinear transformations.  
**Example:** t-SNE and UMAP produce nonlinear embeddings to visualize clusters.

#### NumPy Basics
Core functionalities of the NumPy library for handling n-dimensional arrays and performing vectorized operations.  
**Example:** Creating and reshaping arrays, broadcasting, and using universal functions like `np.exp()`.

#### Object Detection
Identifying and localizing objects within images or video frames, typically returning bounding boxes and class labels.  
**Example:** A YOLO-based model scanning real-time video to detect pedestrians and vehicles.

#### Ollama
A community-driven project that democratizes access to powerful LLMs developed by various organizations.

Ollama was founded by Michael Chiang and Jeffrey Morgan in Palo Alto, CA, and is an independent startup that participated in the W21 batch of Y Combinator.

Ollama is not directly associated with Meta or the development of the Llama models.

Ollama makes it easy to run LLMs on local hardware such as a consumer-grade GPU.  Installation can be done with two simple UNIX shell commands.

* [Ollama website](https://ollama.com/)

#### One-hot Encoded Format

A data preprocessing technique that represents categorical variables as binary vectors where exactly one element is 1 (hot) and all others are 0 (cold).

**Example:** In MNIST digit classification, each target label (0-9) is encoded as a 10-dimensional binary vector: the digit '3' becomes [0,0,0,1,0,0,0,0,0,0], where only the fourth position (index 3) contains a 1.

#### Overfitting Vs Underfitting
Overfitting occurs when a model learns spurious details in the training data, while underfitting fails to capture underlying trends.  
**Example:** A deep network might overfit a small dataset; a simple linear model might underfit a complex dataset.

#### PCA
Principal Component Analysis; a linear dimensionality reduction method finding directions of maximum variance.  
**Example:** PCA can compress a 100-dimensional dataset into a handful of principal components for visualization.

#### Pandas Basics
Common operations in the Pandas library, such as DataFrame creation, cleaning, merging, and analysis.  
**Example:** Using `pandas.read_csv()` to load a dataset and `DataFrame.describe()` for summary statistics.

#### Perceptron Model
A single-layer linear classifier that outputs a binary decision based on a weighted sum of inputs.  
**Example:** Historically important as the earliest form of a neural network unit.

#### Policy Gradient
A family of RL algorithms that optimize a parameterized policy by ascending an estimate of the gradient of expected reward.  
**Example:** REINFORCE is a basic policy gradient method using sampled trajectories.

#### Pooling Layers
Layers that reduce spatial dimensions by combining nearby feature responses, helping control overfitting.  
**Example:** Max pooling or average pooling in a CNN to halve the width and height of feature maps.

#### Positional Encoding
A technique used in transformer architectures to inject information about sequence order without recurrence.  
**Example:** Sine and cosine functions of different frequencies are added to word embeddings in Transformers.

#### Precision
The fraction of predicted positives that are truly positive, measuring correctness among positive predictions.  
**Example:** High precision in a medical test means few healthy people are incorrectly diagnosed as sick.

#### Pruning
Removing weights or neurons from a trained model to reduce complexity and size, often with minimal accuracy loss.  
**Example:** Pruning can eliminate near-zero weights, speeding up inference in production.

#### Privacy
The right or requirement that individuals control their personal data and how it is used.  
**Example:** Differential privacy in ML ensures training data cannot be reverse-engineered from model parameters.

#### Prompt Engineering
The practice of carefully designing prompts or instructions to guide large language models for desired outputs.  
**Example:** Providing explicit context and constraints in a GPT prompt to extract factual answers.

#### Protein Folding
The computational prediction of a protein’s 3D structure from its amino acid sequence, aided by AI breakthroughs.  
**Example:** DeepMind’s AlphaFold made significant progress in accurately modeling protein configurations.

#### Project Scoping
Defining the objectives, resources, and deliverables of a machine learning or AI project.  
**Example:** Determining the data needed, performance targets, and timeline for a sentiment analysis project.

#### PyTorch Intro
An overview of the PyTorch deep learning framework, emphasizing its dynamic computation graph.  
**Example:** PyTorch’s imperative style makes debugging simpler and is favored by many researchers.

#### Python Setup
Basic installation and configuration of Python environments, packages, and tools.  
**Example:** Installing Python 3.x, pip, and setting up a virtual environment on a workstation.

#### Q-Learning
A model-free RL algorithm that learns an action-value function predicting future rewards.  
**Example:** The algorithm updates a Q-table or Q-network using Bellman equations and experiences from the environment.

#### Quantization
Reducing the precision of model parameters (e.g., from 32-bit floating-point to 8-bit integers) to improve efficiency.  
**Example:** Quantized networks often run faster on edge devices with limited compute power.

#### Random Search
A hyperparameter tuning method selecting parameter combinations randomly within predefined ranges.  
**Example:** Often more efficient than grid search when the parameter space is large.

#### Recall
The fraction of actual positives correctly identified by the model, measuring completeness.  
**Example:** In a medical test, recall indicates the percentage of sick patients who are correctly diagnosed.

#### Recommender Systems
Algorithms designed to suggest items, such as products or media, to users based on preferences or similarities.  
**Example:** Matrix factorization or deep collaborative filtering for movie recommendations.

#### Regression Vs Classification
Distinguishes between predicting a continuous value (regression) and predicting a discrete label (classification).  
**Example:** House price prediction is regression, while determining spam vs non-spam email is classification.

#### Regularization Overview
Techniques (like weight decay, dropout) that constrain model complexity to prevent overfitting.  
**Example:** Adding L2 regularization to a neural network’s cost function penalizes large weights.

#### Requirement Analysis
Identifying the scope, constraints, and success criteria for an AI project before implementation.  
**Example:** Determining user needs, regulatory constraints, and performance thresholds for a medical imaging system.

#### Residual Networks (ResNet)
Deep neural networks with skip connections that help mitigate vanishing gradients by enabling identity mappings.  
**Example:** ResNet-50 is a popular architecture for ImageNet classification tasks.

#### Responsible AI
AI development and deployment aligned with legal, ethical, and societal values.  
**Example:** Designing systems that respect user privacy, avoid bias, and are transparent.

#### Reward Shaping
Modifying or adding auxiliary rewards in RL to guide the agent toward desired behaviors.  
**Example:** Giving an agent a small reward for each time step it stays on track in a self-driving simulation.

#### RL Definition
Reinforcement Learning is a subfield of AI where agents learn optimal behaviors through trial-and-error feedback from an environment.  
**Example:** An RL agent in a game environment receives rewards (scores) for achieving objectives.

#### Robotics
The intersection of engineering and AI focusing on designing and controlling physical machines that perform tasks autonomously or semi-autonomously.  
**Example:** Automated robotic arms in manufacturing lines guided by computer vision.

#### Scaling And Normalization
Rescaling features to standardized ranges or distributions for more stable training.  
**Example:** Applying min-max scaling to each feature so that all values lie between 0 and 1.

#### Score Matching
A technique for training generative models by matching the score function (gradient of log-density) of the model to that of the data.  
**Example:** Used in diffusion-based methods to iteratively denoise samples.

#### SciPy Basics
Fundamental capabilities of SciPy for scientific computations, including optimization, integration, and statistics.  
**Example:** Using `scipy.optimize` to implement advanced fitting procedures in an ML pipeline.

#### Scikit-Learn Overview
Core features of the scikit-learn library, offering high-level APIs for classical ML algorithms, preprocessing, and validation.  
**Example:** Using `GridSearchCV` for hyperparameter tuning on a random forest model.

#### Security
Protective measures ensuring AI systems resist unauthorized access, tampering, or malicious attacks.  
**Example:** Hardening model APIs against adversarial inputs that might reveal sensitive data.

#### Self-Attention
A mechanism in transformer models enabling each position in a sequence to attend to every other position for context.  
**Example:** BERT’s self-attention layers capture dependencies in a sentence without relying on recurrence.

#### Sequence Modeling
Approaches to handle sequential data (text, time series, etc.), capturing dependencies across timesteps.  
**Example:** LSTMs, GRUs, and Transformers are common sequence modeling architectures.

#### Sequence-To-Sequence
Neural architectures mapping an input sequence (e.g., a sentence in English) to an output sequence (e.g., a sentence in French).  
**Example:** Used extensively in neural machine translation and speech recognition tasks.

#### SHAP
SHapley Additive exPlanations; a method based on Shapley values to interpret predictions by attributing contributions of each feature.  
**Example:** Generating a SHAP plot to see which features most influence a credit-scoring model’s decisions.

#### Skip Connections
Links that bypass one or more layers, helping gradients flow more easily in deep networks.  
**Example:** ResNet’s skip connections add layer outputs directly to subsequent layers to mitigate vanishing gradients.

#### Speech Recognition
The task of converting spoken language into text using acoustic and language models.  
**Example:** Voice assistants transcribe user commands in real-time for further processing.

#### Stochastic Training
Updating model parameters after each individual sample or a small random subset (mini-batch).  
**Example:** Known as SGD, helps the model converge faster with more frequent updates.

#### Stride And Padding
Parameters in convolution operations that determine how filters slide over input data and whether border regions are preserved.  
**Example:** Padding=“same” ensures output dimensions remain the same as input for certain CNN layers.

#### Style Transfer
Neural technique that reworks the style of one image onto the content of another.  
**Example:** Merging a photo with the painting style of Van Gogh’s “Starry Night.”

#### Supervised Vs Unsupervised
A distinction between learning with labeled data (supervised) vs. discovering patterns in unlabeled data (unsupervised).  
**Example:** Classifying labeled images is supervised; clustering unlabeled images is unsupervised.

#### Symbolic AI
AI methods using explicit, human-readable representations of problems, logic, and rules.  
**Example:** Expert systems that encode domain knowledge in symbolic form.

#### Team Collaboration
Collective effort where multiple individuals share responsibilities and knowledge to complete AI projects.  
**Example:** Data engineers, data scientists, and software developers working together on a production pipeline.

#### t-SNE
t-Distributed Stochastic Neighbor Embedding; a non-linear dimensionality reduction method for visualization.  
**Example:** Often used to plot high-dimensional data (like embeddings) in 2D or 3D.

#### TensorFlow Intro
An overview of the TensorFlow framework emphasizing its computational graph, eager mode, and ecosystem tools.  
**Example:** Students build and train neural networks with Keras, a high-level TensorFlow API.

#### Text Generation
The task of producing coherent text sequences, often using language models.  
**Example:** GPT-based systems can generate paragraphs of natural-sounding text from a prompt.

#### Text Generation Models
Architectures specialized in producing novel text, typically via learned probability distributions over tokens.  
**Example:** LSTM-based decoders or transformer-based language models for writing summaries or creative content.

#### Text-To-Image
Models that generate images based on textual input descriptions.  
**Example:** DALL·E variants produce custom images from user prompts like “an armchair shaped like an avocado.”

#### Time-Series Forecasting
Techniques to predict future values of a sequence based on past observations.  
**Example:** LSTMs or Prophet library used for forecasting stock prices or energy demands.

#### Training-Validation-Test
Standard data splitting strategy: a training set for model fitting, a validation set for hyperparameter tuning, and a test set for final evaluation.  
**Example:** 60% train, 20% validation, 20% test split for a typical classification dataset.

#### Transfer Learning Basics
Techniques for leveraging pre-trained models on new tasks with limited additional data.  
**Example:** Using a pre-trained ResNet on ImageNet as a feature extractor for a custom dataset.

#### Transformers

An architecture for training deep neural networks that can be parallelized by GPUs.

#### TPU Acceleration
Using Tensor Processing Units (custom ASICs by Google) to speed up large-scale model training.  
**Example:** TPUs can be accessed on Google Cloud to train large transformer models more efficiently than GPUs.

#### UMAP
Uniform Manifold Approximation and Projection; a non-linear technique for dimensionality reduction.  
**Example:** Faster and often better at preserving global structure compared to t-SNE on large datasets.

#### Value Functions
In reinforcement learning, functions estimating expected future rewards from a given state (or state-action pair).  
**Example:** The critic in actor-critic algorithms learns a value function to guide the actor’s updates.

#### Vanilla RNN
A basic recurrent neural network that uses hidden states to process sequences one step at a time.  
**Example:** Applied to simple sequence tasks but prone to vanishing/exploding gradient issues for long sequences.

#### Vanishing Gradients
A phenomenon where gradients diminish in magnitude through backprop, hindering learning in deep networks.  
**Example:** Sigmoid activations can exacerbate vanishing gradients as the network depth grows.

#### Version Control
Systems that track changes to code over time, allowing collaboration and reversion if needed.  
**Example:** Git and GitHub store historical commits and manage parallel development branches.

#### Version Control Workflow
Best practices and procedures for using version control systems, including branching, merging, and reviewing.  
**Example:** A feature branch workflow ensures code is tested and reviewed before merging into `main`.

#### Variational Autoencoders (VAE)
Generative models that learn a latent distribution of data using an encoder-decoder framework with a KL-divergence term.  
**Example:** VAEs create smooth latent spaces, enabling interpolation between different generated samples.

#### Vision Transformers (ViT)
Transformer-based architectures adapted for computer vision tasks by splitting images into patches.  
**Example:** ViT processes each patch as a token, leveraging self-attention for image classification.

#### Weights And Biases
Trainable parameters in a neural network that transform inputs into outputs.  
**Example:** A linear layer with 10 inputs and 5 outputs would have 10×5 weights plus 5 bias terms.

#### Weight Decay
A regularization method applying an L2 penalty on weights to encourage smaller parameter values.  
**Example:** Often implemented as a parameter in optimizers like SGD or Adam (e.g., `weight_decay=1e-4`).

#### Xavier Initialization
A weight initialization method maintaining variance in both forward and backward passes for certain activation functions.  
**Example:** Often used with tanh or sigmoid activations to keep gradients stable early in training.

#### Text-to-Image
*(Duplicate check — We already covered "Text-To-Image" at concept 110. Ensuring no duplication. It's already listed. We’ll ignore any duplicates.)*

*(No more new Z or Y terms, so we’re done.)*

This completes the alphabetical glossary of all 200 concepts.

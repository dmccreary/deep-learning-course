import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml

def plot_mnist_distribution():
    # Load MNIST dataset from scikit-learn
    print("Loading MNIST dataset... (this might take a minute)")
    X, y = fetch_openml('mnist_784', version=1, as_frame=False, return_X_y=True, parser='auto')
    y = y.astype(int)
    
    # Count the frequency of each digit
    unique, counts = np.unique(y, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    # Set up the plot style
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    
    # Create bar plot
    bars = plt.bar(
        distribution.keys(),
        distribution.values(),
        color=sns.color_palette("husl", 10)
    )
    
    # Customize the plot
    plt.title('Distribution of Digits in MNIST Dataset', fontsize=14, pad=20)
    plt.xlabel('Digit', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2.,
            height,
            f'{int(height):,}',
            ha='center',
            va='bottom'
        )
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Add a grid for better readability
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    # Ensure y-axis starts at 0
    plt.ylim(bottom=0)
    
    # Add a brief description as text
    plt.figtext(
        0.99, 0.01,
        'Total samples: {:,}'.format(len(y)),
        ha='right',
        va='bottom',
        fontsize=10,
        style='italic'
    )
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate and display the plot
    plot_mnist_distribution()
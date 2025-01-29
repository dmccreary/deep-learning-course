import React, { useEffect, useRef } from 'react';
import WordCloud from 'wordcloud';

const ConceptWordcloud = () => {
  const canvasRef = useRef(null);

  const concepts = [
    ['Neural Networks', 70],
    ['Backpropagation', 68],
    ['Deep Architecture', 67],
    ['Gradient Descent', 65],
    ['Loss Functions', 64],
    ['Activation Functions', 63],
    ['Supervised Learning', 62],
    ['CNN Architectures', 61],
    ['Learning Rate', 60],
    ['Transformers', 59],
    ['LSTM Units', 58],
    ['Optimization', 57],
    ['Regularization', 56],
    ['Feature Engineering', 55],
    ['Batch Normalization', 54],
    ['Transfer Learning', 53],
    ['Data Preprocessing', 52],
    ['Model Evaluation', 51],
    ['Overfitting', 50],
    ['Attention Mechanism', 49],
    ['GPU Acceleration', 48],
    ['Computer Vision', 47],
    ['NLP', 46],
    ['Reinforcement Learning', 45],
    ['GAN Basics', 44],
    ['Cross-Validation', 43],
    ['Hyperparameter Tuning', 42],
    ['Data Augmentation', 41],
    ['Model Architecture', 40],
    ['Deep Learning', 70],
    ['Machine Learning', 65],
    ['PyTorch', 50],
    ['TensorFlow', 50],
    ['Weights and Biases', 48],
    ['Convolutional Neural Networks', 60],
    ['Recurrent Neural Networks', 55],
    ['Transformer Architecture', 58],
    ['Training Process', 52],
    ['Optimization Algorithms', 54],
    ['Loss Landscape', 46],
    ['Gradient Flow', 45],
    ['Model Parallelism', 44],
    ['Neural Architecture', 53],
    ['Deep Networks', 51],
    ['Residual Connections', 49],
    ['Attention Layers', 47],
    ['Embedding Layers', 46],
    ['Softmax Function', 43],
    ['Dropout Layers', 42],
    ['Mini-Batch Training', 41]
  ];

  useEffect(() => {
    if (canvasRef.current) {
      WordCloud(canvasRef.current, {
        list: concepts,
        weightFactor: 1,
        fontFamily: 'Inter, system-ui, Avenir, Helvetica, Arial, sans-serif',
        color: 'random-dark',
        rotateRatio: 0.5,
        click: function(item) {
          const slug = item[0].toLowerCase().replace(/\s+/g, '-');
          window.location.href = `../../glossary.md#${slug}`;
        },
        hover: function(item) {
          canvasRef.current.style.cursor = 'pointer';
        },
        backgroundColor: 'white'
      });
    }
  }, []);

  return (
    <div className="w-full max-w-4xl mx-auto p-4">
      <canvas 
        ref={canvasRef} 
        className="w-full h-96 border rounded-lg shadow-sm"
        width="800"
        height="400"
      />
      <p className="text-sm text-gray-600 mt-2 text-center">
        Click on any concept to view its definition in the glossary
      </p>
    </div>
  );
};

export default ConceptWordcloud;
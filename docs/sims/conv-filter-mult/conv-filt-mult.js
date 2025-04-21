// CNN Image Convolution MicroSim with Matrix Multiplication Display
// Demonstrates how convolution works with realistic image data and edge detection filters
// Includes matrix multiplication visualization

// Canvas dimensions
let canvasWidth = 400;                      // Initial width that will be updated responsively
let drawHeight = 370;                       // Height of simulation/drawing area
let controlHeight = 90;                     // Height of controls area (increased for filter selection)
let canvasHeight = drawHeight + controlHeight; // Total canvas height
let margin = 25;                            // Margin for visual elements
let sliderLeftMargin = 170;                 // Left margin for slider positioning
let defaultTextSize = 16;                   // Base text size for readability

// Global variables for responsive design
let containerWidth;                         // Calculated from container upon resize
let containerHeight = canvasHeight;         // Usually fixed height on page

// Simulation variables
let inputMatrix = [];                       // 8x8 input matrix (now represents image data)
let outputMatrix = [];                      // Output matrix after convolution
let windowPositionSlider;                   // Controls the window position
let windowPosition = 0;                     // Current position of sliding window
let windowSize = 3;                         // Size of sliding window (3x3)
let matrixSize = 8;                         // Size of the input matrix (8x8)
let outputMatrixSize;                       // Size of output matrix (matrixSize - windowSize + 1)
let isRunning = false;                      // Animation state
let startButton;                            // Button to start/pause animation
let resetButton;                            // Button to reset simulation
let animationSpeed = 500;                   // ms between window movements
let lastMoveTime = 0;                       // Tracks last window movement
let filterSelect;                           // Dropdown for filter selection
let currentFilter = 'horizontal';           // Current filter type

// Predefined filters (kernels)
const filters = {
  horizontal: [
    [-1, -1, -1],
    [0, 0, 0],
    [1, 1, 1]
  ],
  vertical: [
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
  ],
  laplacian: [
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
  ],
  identity: [
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
  ],
  blur: [
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
  ],
  sharpen: [
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
  ]
};

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  
  // Calculate output matrix size
  outputMatrixSize = matrixSize - windowSize + 1;
  
  // Generate sample image data
  generateSampleImage();
  
  // Initialize output matrix with empty values (-1 represents empty)
  initializeOutputMatrix();
  
  // Create slider for window position
  let maxSliderPosition = (matrixSize - windowSize + 1) * (matrixSize - windowSize + 1) - 1;
  windowPositionSlider = createSlider(0, maxSliderPosition, 0);
  windowPositionSlider.position(sliderLeftMargin, drawHeight + 10);
  windowPositionSlider.size(containerWidth - sliderLeftMargin - 25);
  windowPositionSlider.input(updateWindowPosition);
  
  // Create filter selection dropdown
  filterSelect = createSelect();
  filterSelect.position(sliderLeftMargin, drawHeight + 40);
  filterSelect.option('Horizontal Edge', 'horizontal');
  filterSelect.option('Vertical Edge', 'vertical');
  filterSelect.option('Laplacian Edge', 'laplacian');
  filterSelect.option('Identity', 'identity');
  filterSelect.option('Blur', 'blur');
  filterSelect.option('Sharpen', 'sharpen');
  filterSelect.changed(filterChanged);
  
  // Create start/pause button
  startButton = createButton('Start');
  startButton.position(10, drawHeight + 65);
  startButton.mousePressed(toggleSimulation);
  
  // Create reset button
  resetButton = createButton('Reset');
  resetButton.position(70, drawHeight + 65);
  resetButton.mousePressed(resetSimulation);
  
  // Initialize first output value for starting position
  updateOutputMatrix();
  
  describe('A MicroSim demonstrating CNN convolution with realistic image data using various edge detection and image processing filters, including matrix multiplication visualization.', LABEL);
}

function generateSampleImage() {
  // Create a simple pattern resembling an image with edges
  // Values are between 0-255 to simulate grayscale image
  inputMatrix = [
    [50,  50,  50,  50,  200, 200, 200, 200],
    [50,  50,  50,  50,  200, 200, 200, 200],
    [50,  50,  100, 100, 100, 100, 200, 200],
    [50,  50,  100, 150, 150, 100, 200, 200],
    [50,  50,  100, 150, 150, 100, 200, 200],
    [50,  50,  100, 100, 100, 100, 200, 200],
    [50,  50,  50,  50,  200, 200, 200, 200],
    [50,  50,  50,  50,  200, 200, 200, 200]
  ];
}

function initializeOutputMatrix() {
  for (let i = 0; i < outputMatrixSize; i++) {
    outputMatrix[i] = [];
    for (let j = 0; j < outputMatrixSize; j++) {
      outputMatrix[i][j] = -1; // Use -1 to represent empty cells
    }
  }
}

function filterChanged() {
  currentFilter = filterSelect.value();
  resetSimulation();
}

function updateCanvasSize() {
  const container = document.querySelector('main').getBoundingClientRect();
  containerWidth = Math.floor(container.width);
  canvasWidth = containerWidth;
}

function windowResized() {
  updateCanvasSize();
  resizeCanvas(containerWidth, containerHeight);
  
  // Update slider size
  windowPositionSlider.size(containerWidth - sliderLeftMargin - 25);
}

function draw() {
  // Draw area (background)
  fill('aliceblue');
  stroke('silver');
  strokeWeight(1);
  rect(0, 0, canvasWidth, drawHeight);
  
  // Controls area
  fill('white');
  stroke('silver');
  strokeWeight(1);
  rect(0, drawHeight, canvasWidth, controlHeight);
  
  // Draw title
  fill('black');
  noStroke();
  textSize(20); // Reduced for 5 panels
  textAlign(CENTER, TOP);
  text("CNN Image Convolution with Matrix Multiplication", canvasWidth/2, margin/2);
  
  // Layout calculations for 5 panels
  let colWidth = containerWidth / 5;
  let fixedCellHeight = (drawHeight - 5*margin) / matrixSize * 0.8; // Slightly reduced
  let inputCellSize = fixedCellHeight;
  let kernelCellSize = (colWidth - 2*margin) / windowSize;
  let windowCellSize = kernelCellSize;
  let matMulCellSize = kernelCellSize;
  let outputCellSize = fixedCellHeight * 1.2;
  
  let startY = margin*3;
  
  // Get window row and column from position
  let windowRow = floor(windowPosition / (matrixSize - windowSize + 1));
  let windowCol = windowPosition % (matrixSize - windowSize + 1);
  
  // Calculate center positions for each matrix
  let inputStartX = (colWidth - (inputCellSize * matrixSize)) / 2 + 10;
  let kernelStartX = colWidth + (colWidth - (kernelCellSize * windowSize)) / 2 + 20;
  let windowStartX = 2 * colWidth + (colWidth - (windowCellSize * windowSize)) / 2 + 10;
  let matMulStartX = 3 * colWidth + (colWidth - (matMulCellSize * windowSize)) / 2;
  let outputStartX = 4 * colWidth + (colWidth - (outputCellSize * outputMatrixSize)) / 2;
  
  // Draw section titles
  fill('blue');
  strokeWeight(0);
  textSize(defaultTextSize - 2); // Slightly smaller for more panels
  textAlign(CENTER, BOTTOM);
  text("Input Image", colWidth / 2 + 10, startY - 10);
  text("Filter/Kernel", colWidth + colWidth / 2 + 20, startY - 10);
  text("Window Values", 2 * colWidth + colWidth / 2 + 10, startY - 10);
  text("Multiplication", 3 * colWidth + colWidth / 2, startY - 10);
  text("Feature Map", 4 * colWidth + colWidth / 2, startY - 10);
  
  // Draw the matrices
  drawImageMatrix(inputMatrix, matrixSize, inputStartX, startY, inputCellSize);
  drawSlidingWindow(inputStartX, startY, inputCellSize, windowRow, windowCol);
  
  // Draw the current filter/kernel
  drawKernel(filters[currentFilter], windowSize, kernelStartX, startY, kernelCellSize);
  
  // Extract and draw the window contents
  let windowMatrix = extractWindow(windowRow, windowCol);
  drawImageMatrix(windowMatrix, windowSize, windowStartX, startY, windowCellSize);
  
  // Calculate and draw the matrix multiplication result
  let multiplicationMatrix = calculateMatrixMultiplication(windowMatrix, filters[currentFilter]);
  drawMatrixMultiplication(multiplicationMatrix, windowSize, matMulStartX, startY, matMulCellSize);
  
  // Draw the output matrix (feature map)
  drawFeatureMap(outputMatrix, outputMatrixSize, outputStartX, startY, outputCellSize);
  highlightOutputPosition(outputStartX, startY, outputCellSize, windowRow, windowCol);
  
  // Draw control labels
  fill('black');
  noStroke();
  textSize(defaultTextSize);
  textAlign(LEFT, CENTER);
  text('Window Position: ' + windowPosition, 10, drawHeight + 18);
  text('Filter Type:', 80, drawHeight + 48);
  
  // Animate the sliding window if running
  if (isRunning && millis() - lastMoveTime > animationSpeed) {
    let maxPosition = (matrixSize - windowSize + 1) * (matrixSize - windowSize + 1) - 1;
    windowPosition = (windowPosition + 1) % (maxPosition + 1);
    windowPositionSlider.value(windowPosition);
    updateWindowPosition();
    lastMoveTime = millis();
  }
}

function drawImageMatrix(matrix, size, startX, startY, cellSize) {
  // Draw matrix cells as grayscale image
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let x = startX + j * cellSize;
      let y = startY + i * cellSize;
      
      // Draw cell background as grayscale
      let value = matrix[i][j];
      if (value === -1) { // Empty cell
        fill(235);
      } else {
        // Use grayscale value for image data
        fill(value);
      }
      stroke('gray');
      strokeWeight(1);
      rect(x, y, cellSize, cellSize);
      
      // Draw cell value for debugging/learning
      if (value !== -1) {
        fill(value > 127 ? 'black' : 'white'); // Contrast text color
        noStroke();
        textSize(cellSize * 0.4);
        textAlign(CENTER, CENTER);
        text(value, x + cellSize/2, y + cellSize/2);
      }
    }
  }
}

function drawKernel(kernel, size, startX, startY, cellSize) {
  // Draw kernel/filter with values
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let x = startX + j * cellSize;
      let y = startY + i * cellSize;
      
      // Background color based on kernel value
      let value = kernel[i][j];
      let bgColor = map(value, -4, 5, 50, 255);
      fill(bgColor);
      
      stroke('gray');
      strokeWeight(1);
      rect(x, y, cellSize, cellSize);
      
      // Draw kernel value
      fill(bgColor > 127 ? 'black' : 'white'); // Contrast text color
      noStroke();
      textSize(cellSize * 0.6);
      textAlign(CENTER, CENTER);
      
      // Format value (round if necessary)
      let displayValue = value.toFixed(value % 1 === 0 ? 0 : 2);
      text(displayValue, x + cellSize/2, y + cellSize/2);
    }
  }
}

function drawMatrixMultiplication(matrix, size, startX, startY, cellSize) {
  // Draw the element-wise multiplication results
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let x = startX + j * cellSize;
      let y = startY + i * cellSize;
      
      // Background color for multiplication result
      let value = matrix[i][j];
      let bgColor;
      
      // Color-code positive and negative values
      if (value > 0) {
        bgColor = color(200, 255, 200); // Light green for positive
      } else if (value < 0) {
        bgColor = color(255, 200, 200); // Light red for negative
      } else {
        bgColor = color(240); // Light gray for zero
      }
      
      fill(bgColor);
      stroke('gray');
      strokeWeight(1);
      rect(x, y, cellSize, cellSize);
      
      // Draw multiplication value - properly round for blur filter
      fill('black');
      noStroke();
      textSize(cellSize * 0.5);
      textAlign(CENTER, CENTER);
      
      // Round the display value appropriately
      let displayValue;
      if (currentFilter === 'blur') {
        // For blur filter, round to 1 decimal to show nice values like 5.6
        displayValue = value.toFixed(1);
      } else if (value % 1 === 0) {
        // For whole numbers, display without decimals
        displayValue = value.toFixed(0);
      } else {
        // For other fractional values, show 1 decimal
        displayValue = value.toFixed(1);
      }
      
      text(displayValue, x + cellSize/2, y + cellSize/2);
    }
  }
  
  // Draw sum indicator
  let sum = 0;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      sum += matrix[i][j];
    }
  }
  
  // Draw the sum equation below the matrix
  let textY = startY + size * cellSize + 15;
  fill('black');
  textSize(12);
  textAlign(CENTER, TOP);
  text("Sum = " + sum.toFixed(1), startX + (size * cellSize) / 2, textY);
}

function drawFeatureMap(matrix, size, startX, startY, cellSize) {
  // Draw feature map with different visualization
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let x = startX + j * cellSize;
      let y = startY + i * cellSize;
      
      // Draw cell background
      let value = matrix[i][j];
      let mappedValue;
      
      if (value === -1) {
        fill(235); // Empty cell
      } else {
        // Map convolution output to color
        mappedValue = constrain(map(value, -255, 255, 0, 255), 0, 255);
        fill(mappedValue);
      }
      
      stroke('gray');
      strokeWeight(1);
      rect(x, y, cellSize, cellSize);
      
      // Draw value if not empty
      if (value !== -1) {
        fill(mappedValue > 127 ? 'black' : 'white'); // Contrast text color
        noStroke();
        textSize(cellSize * 0.4);
        textAlign(CENTER, CENTER);
        text(value.toFixed(0), x + cellSize/2, y + cellSize/2);
      }
    }
  }
}

function calculateMatrixMultiplication(windowMatrix, kernel) {
  // Calculate element-wise multiplication
  let resultMatrix = [];
  
  for (let i = 0; i < windowSize; i++) {
    resultMatrix[i] = [];
    for (let j = 0; j < windowSize; j++) {
      resultMatrix[i][j] = windowMatrix[i][j] * kernel[i][j];
    }
  }
  
  return resultMatrix;
}

function calculateConvolution(windowMatrix, kernel) {
  let result = 0;
  
  // Element-wise multiplication and sum
  for (let i = 0; i < windowSize; i++) {
    for (let j = 0; j < windowSize; j++) {
      result += windowMatrix[i][j] * kernel[i][j];
    }
  }
  
  // Apply ReLU activation (simple non-linearity)
  return max(0, result);
}

function updateOutputMatrix() {
  // Get window row and column
  let windowRow = floor(windowPosition / (matrixSize - windowSize + 1));
  let windowCol = windowPosition % (matrixSize - windowSize + 1);
  
  // Extract current window
  let windowMatrix = extractWindow(windowRow, windowCol);
  
  // Calculate convolution with current filter
  let result = calculateConvolution(windowMatrix, filters[currentFilter]);
  
  // Update output matrix at the corresponding position
  outputMatrix[windowRow][windowCol] = result;
}

function drawSlidingWindow(startX, startY, cellSize, windowRow, windowCol) {
  // Highlight the current sliding window position
  noFill();
  stroke('blue');
  strokeWeight(3);
  rect(
    startX + windowCol * cellSize,
    startY + windowRow * cellSize,
    windowSize * cellSize,
    windowSize * cellSize
  );
}

function highlightOutputPosition(startX, startY, cellSize, row, col) {
  // Highlight the current position in the output matrix
  stroke('red');
  strokeWeight(3);
  noFill();
  rect(
    startX + col * cellSize,
    startY + row * cellSize,
    cellSize,
    cellSize
  );
}

function extractWindow(windowRow, windowCol) {
  // Extract the current window from the input matrix
  let windowMatrix = [];
  for (let i = 0; i < windowSize; i++) {
    windowMatrix[i] = [];
    for (let j = 0; j < windowSize; j++) {
      windowMatrix[i][j] = inputMatrix[windowRow + i][windowCol + j];
    }
  }
  return windowMatrix;
}

function updateWindowPosition() {
  windowPosition = windowPositionSlider.value();
  updateOutputMatrix();
}

function toggleSimulation() {
  isRunning = !isRunning;
  startButton.html(isRunning ? 'Pause' : 'Start');
  lastMoveTime = millis(); // Reset timer on toggle
}

function resetSimulation() {
  // Reset window position
  windowPosition = 0;
  windowPositionSlider.value(0);
  
  // Reset animation state
  isRunning = false;
  startButton.html('Start');
  
  // Reset output matrix to empty values (-1)
  initializeOutputMatrix();
  
  // Calculate initial position
  updateOutputMatrix();
}
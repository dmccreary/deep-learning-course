// CNN Convolution MicroSim
// Demonstrates how convolution works with a sliding window and produces an output matrix
// Shows both the operation and resulting feature map

// Canvas dimensions
let canvasWidth = 400;                      // Initial width that will be updated responsively
let drawHeight = 370;                       // Height of simulation/drawing area
let controlHeight = 70;                     // Height of controls area (increased for radio buttons)
let canvasHeight = drawHeight + controlHeight; // Total canvas height
let margin = 25;                            // Margin for visual elements
let sliderLeftMargin = 170;                 // Left margin for slider positioning
let defaultTextSize = 16;                   // Base text size for readability

// Global variables for responsive design
let containerWidth;                         // Calculated from container upon resize
let containerHeight = canvasHeight;         // Usually fixed height on page

// Simulation variables
let inputMatrix = [];                       // 8x8 input matrix
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
let radioSum;                               // Radio button for sum operation
let radioAvg;                               // Radio button for average operation
let currentOperation = 'sum';               // Current convolution operation (default: sum)

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  
  // Calculate output matrix size
  outputMatrixSize = matrixSize - windowSize + 1;
  
  // Generate random integer matrix (values between 0-9)
  for (let i = 0; i < matrixSize; i++) {
    inputMatrix[i] = [];
    for (let j = 0; j < matrixSize; j++) {
      inputMatrix[i][j] = floor(random(10)); // Random integer between 0-9
    }
  }
  
  // Initialize output matrix with empty values (-1 represents empty)
  for (let i = 0; i < outputMatrixSize; i++) {
    outputMatrix[i] = [];
    for (let j = 0; j < outputMatrixSize; j++) {
      outputMatrix[i][j] = -1; // Use -1 to represent empty cells
    }
  }
  
  // Create slider for window position
  let maxSliderPosition = (matrixSize - windowSize + 1) * (matrixSize - windowSize + 1) - 1;
  windowPositionSlider = createSlider(0, maxSliderPosition, 0);
  windowPositionSlider.position(sliderLeftMargin, drawHeight + 10);
  windowPositionSlider.size(containerWidth - sliderLeftMargin - 25);
  windowPositionSlider.input(updateWindowPosition);
  
  // Create start/pause button
  startButton = createButton('Start');
  startButton.position(10, drawHeight + 35);
  startButton.mousePressed(toggleSimulation);
  
  // Create reset button
  resetButton = createButton('Reset');
  resetButton.position(70, drawHeight + 35);
  resetButton.mousePressed(resetSimulation);
  
  // Create radio buttons for operation type
  createRadioButtons();
  
  // Initialize first output value for starting position
  updateOutputMatrix();
  
  describe('A MicroSim showing a 3x3 sliding window moving across an 8x8 matrix, demonstrating CNN convolution concept and displaying the resulting output matrix.', LABEL);
}

function createRadioButtons() {
  // Create radio button container div
  let radioContainer = createDiv('');
  radioContainer.position(10, drawHeight + 60);
  radioContainer.style('display', 'flex');
  radioContainer.style('align-items', 'center');
  let yOffset = 38;
  
  // Add label to left of radio buttons
  let label = createP('Operation:');
  label.position(sliderLeftMargin, drawHeight + yOffset);
  label.style('margin', '0');
  label.style('padding', '0');
  label.style('font-size', '16px');
  
  // Create radio buttons
  radioSum = createRadio('operation');
  radioSum.option('sum', 'Sum');
  radioSum.position(sliderLeftMargin + 80, drawHeight + yOffset);
  radioSum.selected('sum');
  radioSum.changed(operationChanged);
  
  radioAvg = createRadio('operation');
  radioAvg.option('avg', 'Average');
  radioAvg.position(sliderLeftMargin + 140, drawHeight + yOffset);
  radioAvg.changed(operationChanged);
}

function operationChanged() {
  if (radioSum.selected()) {
    currentOperation = 'sum';
  } else if (radioAvg.selected()) {
    currentOperation = 'avg';
  }
  
  // Reset simulation when operation changes
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
  textSize(24);
  textAlign(CENTER, TOP);
  text("Convolution Sliding Window & Output Matrix", canvasWidth/2, margin/2);
  
  // Divide canvas into three equal columns
  let colWidth = containerWidth / 3;
  
  // Fixed height for matrices regardless of container width
  let fixedCellHeight = (drawHeight - 4*margin) / matrixSize;
  
  // Calculate cell sizes for each matrix
  let inputCellSize = fixedCellHeight;
  let windowCellSize = (colWidth - 2*margin) / windowSize;
  let outputCellSize = fixedCellHeight * 1.35;
  
  // Center vertically with space for title
  let titleSpace = margin*2;  // Space for title and matrix label
  let startY = titleSpace + margin;
  
  // Get window row and column from position
  let windowRow = floor(windowPosition / (matrixSize - windowSize + 1));
  let windowCol = windowPosition % (matrixSize - windowSize + 1);
  
  // Calculate center positions for each matrix
  // Input matrix (8x8) in first column
  let inputMatrixWidth = inputCellSize * matrixSize;
  let inputStartX = (colWidth - inputMatrixWidth) / 2;
  
  // Window matrix (3x3) in second column
  let windowMatrixWidth = windowCellSize * windowSize;
  let windowStartX = colWidth + (colWidth - windowMatrixWidth) / 2;
  
  // Output matrix (6x6) in third column
  let outputMatrixWidth = outputCellSize * outputMatrixSize;
  let outputStartX = 2 * colWidth + (colWidth - outputMatrixWidth) / 2;
  
  // Draw section titles
  fill('blue');
  strokeWeight(0);
  textSize(defaultTextSize + 2);
  textAlign(CENTER, BOTTOM);
  text("Input Matrix (8×8)", colWidth / 2, startY - 10);
  text("Sliding Window (3×3)", colWidth + colWidth / 2, startY - 10);
  text("Output Matrix (6×6)", 2 * colWidth + colWidth / 2, startY - 10);
  
  // Draw the input matrix
  drawMatrix(inputMatrix, matrixSize, inputStartX, startY, inputCellSize);
  
  // Draw the sliding window highlight in blue over the input
  drawSlidingWindow(inputStartX, startY, inputCellSize, windowRow, windowCol);
  
  // Extract the window contents
  let windowMatrix = extractWindow(windowRow, windowCol);
  
  // Draw central panel with the sliding window
  drawMatrix(windowMatrix, windowSize, windowStartX, startY, windowCellSize);
  
  // Draw the output matrix
  drawMatrix(outputMatrix, outputMatrixSize, outputStartX, startY, outputCellSize);
  
  // Highlight current position in output matrix
  highlightOutputPosition(outputStartX, startY, outputCellSize, windowRow, windowCol);
  
  // Draw control labels
  fill('black');
  noStroke();
  textSize(defaultTextSize);
  textAlign(LEFT, CENTER);
  text('Window Position: ' + windowPosition, 10, drawHeight + 18);
  
  // Animate the sliding window if running
  if (isRunning && millis() - lastMoveTime > animationSpeed) {
    let maxPosition = (matrixSize - windowSize + 1) * (matrixSize - windowSize + 1) - 1;

    windowPosition = (windowPosition + 1) % (maxPosition + 1);
    windowPositionSlider.value(windowPosition);
    updateWindowPosition();
    lastMoveTime = millis();
  }
}

function drawMatrix(matrix, size, startX, startY, cellSize) {
  // Draw matrix cells
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      let x = startX + j * cellSize;
      let y = startY + i * cellSize;
      
      // Draw cell background
      fill('white');
      stroke('gray');
      strokeWeight(1);
      rect(x, y, cellSize, cellSize);
      
      // Draw cell value (skip if it's an empty output cell)
      if (matrix === outputMatrix && matrix[i][j] === -1) {
        // Skip drawing for empty output cells
        continue;
      }
      
      fill('black');
      noStroke();
      textSize(cellSize * 0.6);
      textAlign(CENTER, CENTER);
      
      // Format value if it's in the output matrix (could be a decimal)
      let value = matrix[i][j];
      if (matrix === outputMatrix && currentOperation === 'avg') {
        // Round to 1 decimal place for average values
        value = value.toFixed(1);
      }
      text(value, x + cellSize/2, y + cellSize/2);
    }
  }
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

function calculateConvolution(windowMatrix) {
  let result = 0;
  
  if (currentOperation === 'sum') {
    // Sum all values in window
    for (let i = 0; i < windowSize; i++) {
      for (let j = 0; j < windowSize; j++) {
        result += windowMatrix[i][j];
      }
    }
  } else if (currentOperation === 'avg') {
    // Calculate average of values
    let sum = 0;
    for (let i = 0; i < windowSize; i++) {
      for (let j = 0; j < windowSize; j++) {
        sum += windowMatrix[i][j];
      }
    }
    result = sum / (windowSize * windowSize);
  }
  
  return result;
}

function updateOutputMatrix() {
  // Get window row and column
  let windowRow = floor(windowPosition / (matrixSize - windowSize + 1));
  let windowCol = windowPosition % (matrixSize - windowSize + 1);
  
  // Extract current window
  let windowMatrix = extractWindow(windowRow, windowCol);
  
  // Calculate result based on selected operation
  let result = calculateConvolution(windowMatrix);
  
  // Update output matrix at the corresponding position
  outputMatrix[windowRow][windowCol] = result;
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
  
  // Generate new input matrix
  for (let i = 0; i < matrixSize; i++) {
    for (let j = 0; j < matrixSize; j++) {
      inputMatrix[i][j] = floor(random(10));
    }
  }
  
  // Reset output matrix to empty values (-1)
  for (let i = 0; i < outputMatrixSize; i++) {
    for (let j = 0; j < outputMatrixSize; j++) {
      outputMatrix[i][j] = -1;
    }
  }
  
  // Calculate initial position to avoid empty first cell
  updateOutputMatrix();
}
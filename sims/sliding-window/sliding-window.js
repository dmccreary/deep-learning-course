// CNN Sliding Window Concept MicroSim
// Demonstrates how convolution works with a sliding window
// This version centers both matrices in their respective halves of the canvas

// Canvas dimensions
let canvasWidth = 400;                      // Initial width that will be updated responsively
let drawHeight = 370;                       // Height of simulation/drawing area
let controlHeight = 50;                     // Height of controls area
let canvasHeight = drawHeight + controlHeight; // Total canvas height
let margin = 25;                            // Margin for visual elements
let sliderLeftMargin = 170;                 // Left margin for slider positioning
let defaultTextSize = 16;                   // Base text size for readability

// Global variables for responsive design
let containerWidth;                         // Calculated from container upon resize
let containerHeight = canvasHeight;         // Usually fixed height on page

// Simulation variables
let inputMatrix = [];                       // 8x8 input matrix
let windowPositionSlider;                   // Controls the window position
let windowPosition = 0;                     // Current position of sliding window
let windowSize = 3;                         // Size of sliding window (3x3)
let matrixSize = 8;                         // Size of the input matrix (8x8)
let isRunning = false;                      // Animation state
let startButton;                            // Button to start/pause animation
let resetButton;                            // Button to reset simulation
let animationSpeed = 500;                   // ms between window movements
let lastMoveTime = 0;                       // Tracks last window movement

function setup() {
  updateCanvasSize();
  const canvas = createCanvas(containerWidth, containerHeight);
  canvas.parent(document.querySelector('main'));
  
  // Generate random integer matrix (values between 0-9)
  for (let i = 0; i < matrixSize; i++) {
    inputMatrix[i] = [];
    for (let j = 0; j < matrixSize; j++) {
      inputMatrix[i][j] = floor(random(10)); // Random integer between 0-9
    }
  }
  
  // Create slider for window position
  let maxSliderPosition = (matrixSize - windowSize + 1) * (matrixSize - windowSize + 1) - 1;
  windowPositionSlider = createSlider(0, 35, 0);
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
  
  describe('A MicroSim showing a 3x3 sliding window moving across an 8x8 matrix to demonstrate CNN convolution concept.', LABEL);
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
  // Draw area (left side for input matrix, right side for output)
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
  text("CNN Sliding Window", canvasWidth/2, margin/2);
  
  // Calculate cell size to fit matrices within their respective halves
  let cellSize = min((containerWidth/2 - 4*margin) / matrixSize, (drawHeight - 4*margin) / matrixSize);
  
  // Center the input matrix in the left half
  let leftHalfWidth = containerWidth / 2;
  let totalInputMatrixWidth = cellSize * matrixSize;
  let startX = (leftHalfWidth - totalInputMatrixWidth) / 2;
  
  // Center vertically with space for title
  let totalMatrixHeight = cellSize * matrixSize;
  let titleSpace = margin*2;  // Space for title and matrix label
  let startY = titleSpace + margin;
  
  // Get window row and column from position
  let windowRow = floor(windowPosition / (matrixSize - windowSize + 1));
  let windowCol = windowPosition % (matrixSize - windowSize + 1);
  
  // Draw the input matrix with labels
  textSize(defaultTextSize + 4);
  textAlign(CENTER, BOTTOM);
  text("Input Matrix (8×8)", leftHalfWidth / 2, startY - 10);
  
  drawMatrix(inputMatrix, matrixSize, startX, startY, cellSize);
  
  // Draw the sliding window highlight
  drawSlidingWindow(startX, startY, cellSize, windowRow, windowCol);
  
  // Draw the window contents
  let windowMatrix = extractWindow(windowRow, windowCol);
  
  // Center the window matrix in the right half
  let rightHalfStart = containerWidth / 2;
  let windowCellSize = cellSize * 1.5;  // Make window cells 1.5 times larger
  let totalWindowWidth = windowCellSize * windowSize;
  let windowStartX = rightHalfStart + (leftHalfWidth - totalWindowWidth) / 2;
  
  fill('blue');
  strokeWeight(0);
  textSize(defaultTextSize + 4);
  textAlign(CENTER, BOTTOM);
  text("Window Content (3×3)", rightHalfStart + leftHalfWidth / 2, startY - 10);
  
  drawMatrix(windowMatrix, windowSize, windowStartX, startY, windowCellSize);
  
  // Draw control labels
  fill('black');
  noStroke();
  textSize(defaultTextSize);
  textAlign(LEFT, CENTER);
  text('Window Position: ' + windowPosition, 10, drawHeight + 18);
  
  // Animate the sliding window if running
  if (isRunning && millis() - lastMoveTime > animationSpeed) {
    let maxPosition = (matrixSize - windowSize) * (matrixSize - windowSize + 1);
    windowPosition = (windowPosition + 1) % (maxPosition + 1);
    windowPositionSlider.value(windowPosition);
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
      
      // Draw cell value
      fill('black');
      noStroke();
      textSize(cellSize * 0.6);
      textAlign(CENTER, CENTER);
      text(matrix[i][j], x + cellSize/2, y + cellSize/2);
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
}

function toggleSimulation() {
  isRunning = !isRunning;
  startButton.html(isRunning ? 'Pause' : 'Start');
  lastMoveTime = millis(); // Reset timer on toggle
}

function resetSimulation() {
  windowPosition = 0;
  windowPositionSlider.value(0);
  isRunning = false;
  startButton.html('Start');
}
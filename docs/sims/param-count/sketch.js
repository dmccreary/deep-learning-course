// Demo of parameters in neural networks with a responsive design
let containerWidth;  // Will be set based on container size
let containerHeight = 450;  // Fixed height (400 for drawing + 50 for controls)
let drawHeight = 400;
let drawYCenter = drawHeight / 2 - 40; // leave room for labels
let controlHeight = 50;
let margin = 20;
let sliderLeftMargin = 220;
let numLayersSlider;
let numNeuronsSlider;
let parameterCount = 0;
let labelValueWidth = 240;
let numLayers = 3;  // Default number of layers (including Input and Output)
let numNeurons = 4; // Default number of neurons in each layer

function setup() {
    updateCanvasSize();
    const canvas = createCanvas(containerWidth, containerHeight);
    var mainElement = document.querySelector('main');
    canvas.parent(mainElement);
    
    // number of layers
    numLayersSlider = createSlider(2, 7, 3, 1);
    numLayersSlider.position(sliderLeftMargin, drawHeight + 6);
    numLayersSlider.size(containerWidth - sliderLeftMargin - 25);
    numLayersSlider.input(updateNetwork);
    
    // number of neurons per layer
    numNeuronsSlider = createSlider(2, 8, 4, 1);
    numNeuronsSlider.position(sliderLeftMargin, drawHeight + 26);
    numNeuronsSlider.size(containerWidth - sliderLeftMargin - 25);
    numNeuronsSlider.input(updateNetwork);
    
    updateNetwork();
}

function updateCanvasSize() {
    const container = document.querySelector('main').getBoundingClientRect();
    containerWidth = Math.floor(container.width);
}

function windowResized() {
    updateCanvasSize();
    resizeCanvas(containerWidth, containerHeight);
    
    // Update slider positions and sizes
    numLayersSlider.size(containerWidth - sliderLeftMargin - 25);
    numNeuronsSlider.size(containerWidth - sliderLeftMargin - 25);
    
    updateNetwork();
}

function updateNetwork() {
    // make the background of the drawing region a very light blue
    fill('aliceblue');
    stroke('silver');
    rect(0, 0, containerWidth, drawHeight);
    
    fill('white');
    rect(0, drawHeight, containerWidth, controlHeight);
    stroke('blue');
    
    numLayers = numLayersSlider.value();
    numNeurons = numNeuronsSlider.value();
    
    let layerPositions = Array.from(
        {length: numLayers}, 
        (_, i) => map(i, 0, numLayers - 1, 0.2, 0.8) * containerWidth
    );
    
    parameterCount = 0;
    
    for (let i = 0; i < layerPositions.length; i++) {
        let label = "Hidden";
        if (i === 0) label = "Input";
        else if (i === layerPositions.length - 1) label = "Output";
        
        drawNodes(layerPositions[i], drawYCenter, numNeurons, label);
    }
    
    for (let i = 0; i < layerPositions.length - 1; i++) {
        connectLayers(
            layerPositions[i], drawYCenter, numNeurons,
            layerPositions[i + 1], drawYCenter, numNeurons
        );
        parameterCount += numNeurons * numNeurons; // Adding weights
    }
    
    // Add biases for all but the input layer
    biasParameters = (numLayers - 1) * numNeurons;
    // Total weight parameters = neuronsPerLayer × neuronsPerLayer × (numLayers - 1)
    weightParameters = numNeurons * numNeurons * (numLayers - 1);
    // Sum the bias
    totalParameterCount = biasParameters + weightParameters; 
    
    // Draw the title
    noStroke();
    fill('black');
    textSize(16);
    textAlign(CENTER);
    text('Calculating Parameters for a Fully Connected Neural Network' , containerWidth / 2, margin);

    // Show labels and parameter count
    noStroke();
    fill('black');
    textSize(16);
    textAlign(LEFT);
    text('Number of Layers: ' + numLayers, 5, numLayersSlider.y + 5);
    text('Number of Neurons/Layer: ' + numNeurons, 5, numNeuronsSlider.y + 5);
    
    textAlign(LEFT);
    text('Bias Parameters: ' + biasParameters, 10, drawHeight - 20);
    text('Weight Parameters: ' + weightParameters, 180, drawHeight - 20);
    text('Total Parameter Count: ' + totalParameterCount, 370, drawHeight - 20);
}

function drawNodes(x, y, numNodes, label) {
    stroke('blue');
    strokeWeight(2);
    let gap = 40;
    let startY = y - ((numNodes - 1) * gap) / 2;
    for (let i = 0; i < numNodes; i++) {
        ellipse(x, startY + i * gap, 20, 20);
    }
  
    noStroke();
    fill('black');
    textSize(16);
    textAlign(CENTER, CENTER);
    text(label, x, y + ((numNodes + 1) * gap) / 2);
}

function connectLayers(x1, y1, numNodes1, x2, y2, numNodes2) {
    stroke('blue');
    strokeWeight(1);
    let gap1 = 40;
    let gap2 = 40;
    let startY1 = y1 - ((numNodes1 - 1) * gap1) / 2;
    let startY2 = y2 - ((numNodes2 - 1) * gap2) / 2;
    for (let i = 0; i < numNodes1; i++) {
        for (let j = 0; j < numNodes2; j++) {
            line(x1, startY1 + i * gap1, x2, startY2 + j * gap2);
        }
    }
}
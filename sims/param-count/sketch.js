// Demo of parameters in neural networks with a responsive design
let containerWidth;  // Will be set based on container size
let containerHeight = 450;  // Fixed height (400 for drawing + 50 for controls)
let drawHeight = 400;
let controlHeight = 50;
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
    
    numLayersSlider = createSlider(2, 7, 3);
    numLayersSlider.position(sliderLeftMargin, drawHeight + 6);
    numLayersSlider.size(containerWidth - sliderLeftMargin - 25);
    numLayersSlider.input(updateNetwork);
    
    numNeuronsSlider = createSlider(2, 10, 4);
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
        
        drawNodes(layerPositions[i], drawHeight / 2 - 50, numNeurons, label);
    }
    
    for (let i = 0; i < layerPositions.length - 1; i++) {
        connectLayers(
            layerPositions[i], drawHeight / 2- 50, numNeurons,
            layerPositions[i + 1], drawHeight / 2- 50, numNeurons
        );
        parameterCount += numNeurons * numNeurons; // Adding weights
    }
    
    parameterCount += (numLayers - 1) * numNeurons; // Adding biases
    
    // Show labels and parameter count
    noStroke();
    fill('black');
    textSize(16);
    textAlign(LEFT);
    text('Number of Layers: ' + numLayers, 5, numLayersSlider.y + 5);
    text('Number of Neurons/Layer: ' + numNeurons, 5, numNeuronsSlider.y + 5);
    
    textAlign(LEFT);
    text('Bias Weights: ' + numNeurons, 10, drawHeight - 20);
    textAlign(CENTER, CENTER);
    text('Total Parameter Count: ' + parameterCount, containerWidth / 2, drawHeight - 20);
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
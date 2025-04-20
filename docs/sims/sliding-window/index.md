# CNN Sliding Window MicroSim

<iframe src="./main.html" height="440px" scrolling="no"
  style="overflow: hidden;"></iframe>

[Run the Sliding Window MicroSim](./main.html){ .md-button .md-button--primary }

[Edit the Sliding Window MicroSim](https://editor.p5js.org/dmccreary/sketches/HYYryLcEo)

This CNN Sliding Window MicroSim demonstrates one of the fundamental operations in Convolutional Neural Networks - how a kernel or filter moves across an input matrix to perform convolution.

## Key Features

-   **Input Matrix**: An 8×8 grid of random integers (0-9) representing pixel values or feature map data
-   **Sliding Window**: A 3×3 red-outlined window that moves across the input matrix
-   **Window Content**: A separate display showing the current values inside the sliding window
-   **Interactive Controls**:
    -   Slider to manually position the sliding window
    -   Start/Pause button to automate the window movement
    -   Reset button to return the window to the starting position

## How It Works

The simulation follows the MicroSim rules with a responsive width layout. It visually demonstrates how convolution works in CNNs by:

1.  Showing the complete input matrix on the left side
2.  Highlighting the current 3×3 window position with a red outline
3.  Extracting and displaying the values within that window on the right side

This helps learners understand how a CNN processes information by examining small regions at a time, which is the key to how CNNs detect local patterns like edges, textures, and shapes in images.

## Learning Value

This simple simulation establishes a foundation for understanding more complex CNN operations. In an actual CNN, mathematical operations (like multiplication with filter weights and summation) would be performed on these window values to produce a single output value in the resulting feature map.

You can adjust the sliding window position manually using the slider or watch it move automatically by pressing the Start button. The Reset button returns the window to the initial position (top-left corner).

## Changes for Responsive Design

- Replaced static width/height with containerWidth/containerHeight
- Added updateCanvasSize() function to get container dimensions
- Added windowResized() handler to update canvas and UI elements
- Modified slider positioning and sizing to be relative to container width
- Updated all drawing calculations to use containerWidth instead of fixed width
- Set a fixed containerHeight while allowing width to be responsive
- Adjusted margins and spacing to work better with different screen sizes

The visualization will now resize horizontally while maintaining its functionality and proportions.
# MicroSim Templates

Give these templates to your generative AI programs as a starting
point to get a consistent responsive user interface that
scales well in the mkdocs-material iframe reference.

[Circle Radius Responsive Template HTML](./circle-radius.html){ .md-button .md-button--primary }

[Circle Radius Responsive Template JavaScript File](./sketch.js)

<iframe src="./circle-radius.html" width="600px" height="450px" scrolling="no"
  style="overflow: hidden;"></iframe>

## Changes for Responsive Design

- Replaced static width/height with containerWidth/containerHeight
- Added updateCanvasSize() function to get container dimensions
- Added windowResized() handler to update canvas and UI elements
- Modified slider positioning and sizing to be relative to container width
- Updated all drawing calculations to use containerWidth instead of fixed width
- Set a fixed containerHeight while allowing width to be responsive
- Adjusted margins and spacing to work better with different screen sizes

The visualization will now resize horizontally while maintaining its functionality and proportions.
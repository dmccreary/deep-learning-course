// -------------------------------------------------------------
// p5.js  -  Transformer Attention Demo  (v4, fully responsive)
// -------------------------------------------------------------
let canvasHeight = 180;
let margin       = 40;
const R          = 30;                 // circle radius
const ARC_WT     = 6;                  // arrow thickness
const ARC_COL    = [0, 0, 139];        // dark-blue

let blankWord = "tired";
let tokens, positions = [];
let sel;

// ----------------------------------------------------------------
//  ⬇  Figure out how wide the container is right now
function getContainerWidth() {
  // Use <main> if you have one; otherwise fall back to <body>
  const rect = (document.querySelector('main') || document.body)
               .getBoundingClientRect();
  return Math.floor(rect.width);
}

// ----------------------------------------------------------------
function setup() {
  createCanvas(getContainerWidth(), canvasHeight);
  textAlign(CENTER, CENTER);
  textSize(16);
  noLoop();                            // we'll call redraw() manually

  sel = createSelect();
  ["tired", "wide"].forEach(v => sel.option(v));
  sel.changed(() => { blankWord = sel.value(); build(); redraw(); });

  build();
  redraw();                            // first paint
}

// ----------------------------------------------------------------
function build() {
  tokens = ["The","chicken","did","not","cross","the",
            "road","because","it","was","too", blankWord];
}

// ----------------------------------------------------------------
function computePositions() {
  const left  = margin + R;                  // first circle centre
  const right = width  - margin - R;         // last  circle centre
  const step  = (right - left) / (tokens.length - 1);

  positions = tokens.map((_, i) =>
              createVector(left + i * step, height * 0.5));
}

// ----------------------------------------------------------------
function draw() {
  computePositions();                        // always up-to-date
  fill('aliceblue');
  stroke('silver');
  rect(0, 0, width, canvasHeight);          // clear canvas

  // --- curved attention arc -------------------------------------
  const idxIt     = tokens.indexOf("it");
  const idxTarget = (blankWord === "tired")
                    ? tokens.indexOf("chicken")
                    : tokens.indexOf("road");
  drawAttentionArc(positions[idxIt], positions[idxTarget]);

  // --- word circles ---------------------------------------------
  stroke(0);   strokeWeight(1);
  fill(173, 216, 230);                       // light-blue
  positions.forEach(p => circle(p.x, p.y, R * 2));

  // --- labels ----------------------------------------------------
  noStroke();   fill(0);
  tokens.forEach((tok, i) => text(tok, positions[i].x, positions[i].y));

  // keep dropdown in lower-right of canvas
  sel.position(width - 70, height - 25);
}

// ----------------------------------------------------------------
// Draws a quadratic-curve arrow from edge of A to edge of B
function drawAttentionArc(a, b) {
  stroke(...ARC_COL);   strokeWeight(ARC_WT);   noFill();

  const ctrlY   = a.y - 140;  // arc height
  const ctrl    = createVector((a.x + b.x) / 2, ctrlY);

  // trim ends so the arc meets the rims, not the centres
  const tailDir = p5.Vector.sub(ctrl, a).normalize();
  const tail    = p5.Vector.add(a, p5.Vector.mult(tailDir, R));

  const tipDir  = p5.Vector.sub(b, ctrl).normalize();
  const tip     = p5.Vector.sub(b, p5.Vector.mult(tipDir, R));

  // main curve
  beginShape();
  vertex(tail.x, tail.y);
  quadraticVertex(ctrl.x, ctrl.y, tip.x, tip.y);
  endShape();

  // arrow-head
  push();
  translate(tip.x, tip.y);
  rotate(atan2(tip.y - ctrl.y, tip.x - ctrl.x));
  const h = 20;
  fill(...ARC_COL);   noStroke();
  triangle(0, 0, -h,  h/2, -h, -h/2);
  pop();
}

// ----------------------------------------------------------------
//  ⬇  Handle browser or container resize
function windowResized() {
  resizeCanvas(getContainerWidth(), canvasHeight);
  redraw();
}

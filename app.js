let video = document.getElementById("videoInput");
let canvasOutput = document.getElementById("canvasOutput");
let canvasOriginal = document.getElementById("original");
let canvasGray = document.getElementById("gray");
let canvasBlur = document.getElementById("blur");
let canvasThreshold = document.getElementById("threshold");

// Get threshold input elements
let thresh1 = document.getElementById("thresh1");
let thresh2 = document.getElementById("thresh2");
let thresh1Val = document.getElementById("thresh1-val");
let thresh2Val = document.getElementById("thresh2-val");

let ctxOutput = canvasOutput.getContext("2d");
let ctxOriginal = canvasOriginal.getContext("2d");
let ctxGray = canvasGray.getContext("2d");
let ctxBlur = canvasBlur.getContext("2d");
let ctxThreshold = canvasThreshold.getContext("2d");

navigator.mediaDevices.getUserMedia({ video: true }).then(function (stream) {
  video.srcObject = stream;
  video.play();
});

function processImage() {
  let src = cv.imread(video);
  let original = new cv.Mat();
  let gray = new cv.Mat();
  let blur = new cv.Mat();
  let threshold = new cv.Mat();

  // Original Image
  src.copyTo(original);
  cv.imshow(canvasOriginal, original);

  // Convert to grayscale
  cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
  cv.imshow(canvasGray, gray);

  // Apply Gaussian blur
  cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
  cv.imshow(canvasBlur, blur);

  // Get threshold values from the range sliders
  let th1 = parseInt(thresh1.value);
  let th2 = parseInt(thresh2.value);

  // Apply Canny edge detection with dynamic threshold values
  cv.Canny(blur, threshold, th1, th2);
  cv.imshow(canvasThreshold, threshold);

  // Display the original on output canvas
  cv.imshow(canvasOutput, src);

  // Clean up
  src.delete();
  original.delete();
  gray.delete();
  blur.delete();
  threshold.delete();
}

// Update threshold values when the slider changes
thresh1.addEventListener("input", () => {
  thresh1Val.textContent = thresh1.value;
});
thresh2.addEventListener("input", () => {
  thresh2Val.textContent = thresh2.value;
});

video.addEventListener("play", () => {
  setInterval(processImage, 100); // Process the image every 100ms
});

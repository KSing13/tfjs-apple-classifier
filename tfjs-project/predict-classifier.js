/**
 * CNN Prediction Script
 * Loads the trained model and predicts the class of a new, single image.
 */
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

// --- Configuration (MUST match training script) ---
const labels = ["Green Apple", "Red Apple"];
const imageWidth = 28;
const imageHeight = 28;
// const imageChannels = 1;

// Path to the saved model
const loadModelPath = `file://${path.join(
  __dirname,
  "apple-classifier-tfjs",
  "model.json"
)}`;

// Test image
// const testImagePath = path.join(__dirname, "..", "predict_g001.jpeg");
// const testImagePath = path.join(__dirname, "..", "lime.jpg");
const testImagePath = path.join(__dirname, "..", "predict-images/zucchini.jpg");

// --- Image Preprocessing Function ---
/**
 * Loads and preprocesses a single image for prediction using TF.js native decoding.
 * This must use the exact same preprocessing steps as the training data.
 * @param {string} filepath The path to the image file.
 * @returns {Promise<tf.Tensor4D>} The image tensor, normalized and reshaped [1, 28, 28, 1].
 */
async function preprocessImage(filepath) {
  // Use tf.tidy to clean up intermediate tensors automatically
  return tf.tidy(() => {
    try {
      // Read file buffer synchronously
      const imageBuffer = fs.readFileSync(filepath);

      // Decode the image buffer into a 3D tensor (height, width, channels)
      // Grayscale is the mean of R, G, B channels, creating a tensor of shape [28, 28, 1]
      const imageTensor = tf.node.decodeImage(imageBuffer, 3);

      // Resize the image to 28x28
      const resized = tf.image.resizeBilinear(imageTensor, [
        imageWidth,
        imageHeight,
      ]);

      // Convert to Grayscale (Average across the 3 color channels)
      const grayscale = resized.mean(2).expandDims(2);

      // Normalize (0-255 to 0-1)
      const normalized = grayscale.div(255.0);

      // Expand dimension for the batch axis: [1, 28, 28, 1]
      return normalized.expandDims(0).asType("float32");
    } catch (error) {
      console.error(`\nError loading test image ${filepath}: ${error.message}`);
      console.error(
        "Please verify the 'testImagePath' in the script is correct and the image file exists."
      );
      return null;
    }
  });
}

// --- Prediction Function ---
const predict = async function () {
  const modelDir = path.join(__dirname, "apple-classifier-tfjs");

  if (!fs.existsSync(path.join(modelDir, "model.json"))) {
    console.error(`\nModel not found at ${modelDir}.`);
    console.error(
      "Please ensure you have run 'node train_classifier.js' successfully first."
    );
    return;
  }

  console.log("Loading model from:", loadModelPath);
  const model = await tf.loadLayersModel(loadModelPath);
  console.log("Model loaded successfully.");

  const inputTensor = await preprocessImage(testImagePath);

  if (!inputTensor) {
    return; // Exit if image failed to load
  }

  console.log("\nRunning prediction on test image...");

  // Perform inference
  const output = model.predict(inputTensor);

  // Extract the probabilities
  const predictions = output.dataSync();

  // Find the class index with the highest probability
  const predictedIndex = predictions.indexOf(Math.max(...predictions));
  const predictedLabel = labels[predictedIndex];

  console.log(
    `Prediction Probabilities: [Green Apple: ${predictions[0].toFixed(
      4
    )}, Red Apple: ${predictions[1].toFixed(4)}]`
  );
  console.log("-----------------------------------------");
  console.log(`Predicted Class: ${predictedLabel}`);
  console.log("-----------------------------------------");

  // Clean up tensors to prevent memory leak
  inputTensor.dispose();
  output.dispose();
};

// --- Execution ---
// Explicitly catch any unhandled promise rejections during execution.
predict().catch((err) => {
  console.error("Uncaught execution error:", err.message);
  process.exit(1);
});

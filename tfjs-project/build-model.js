/**
 * Supervised Learning: Apple CNN Classifier
 *
 * Requirements:
 * 1. Node.js environment.
 * 2. `npm install @tensorflow/tfjs-node`.
 * 3. A directory structure for training data:
 * /train-images/
 * /Green Apple/
 * r0_3_100.jpg
 * ...
 * /Red Apple/
 * r0_3_100.jpg
 * ...
 */
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");

// --- Configuration ---
const trainDataDir = path.join(__dirname, "..", "train-images");

// Mapping of class indices to labels
const labels = ["Green Apple", "Red Apple"];
const numOfClasses = labels.length;

// Image and Model parameters
const imageWidth = 28;
const imageHeight = 28;
const imageChannels = 1;

// Training parameters
const batchSize = 32; // Number of samples processed at once during the training and evaluation steps, smaller batch size is often better for memory constraints
const epochsValue = 10; // Numeber of times the model will run through the entire training dataset
const trainTestSplit = 0.8; // 80% for training, 20% for testing

// --- Data Preparation Functions ---
/**
 * Recursively scans the directory to find image files and assigns labels
 * @returns {Array<{filepath: string, label: number}>} Array of file paths and their label indices.
 */
function getFilesAndLabels() {
  console.log(`Scanning data directory: ${trainDataDir}`);
  const data = [];

  // Check if the base directory exists
  if (!fs.existsSync(trainDataDir)) {
    throw new Error(
      `Directory not found: ${trainDataDir}. Please create it and populate it with subdirectories for each class.`
    );
  }

  // Iterate over the class labels
  for (let labelIndex = 0; labelIndex < labels.length; labelIndex++) {
    const labelName = labels[labelIndex];
    const classDir = path.join(trainDataDir, labelName);

    if (!fs.existsSync(classDir)) {
      console.warn(
        `Warning: Class directory not found: ${classDir}. Skipping this class.`
      );
      continue;
    }

    const classFiles = fs
      .readdirSync(classDir)
      .filter(
        (fname) =>
          fname.endsWith(".png") ||
          fname.endsWith(".jpg") ||
          fname.endsWith(".jpeg")
      ) // Filter for common image extensions
      .map((fname) => path.join(classDir, fname));

    classFiles.forEach((filepath) => {
      data.push({ filepath, label: labelIndex });
    });
  }

  if (data.length === 0) {
    throw new Error(
      "No images found. Check your file paths and directory structure."
    );
  }

  tf.util.shuffle(data); // Shuffle the entire dataset
  console.log(`Found ${data.length} total images.`);
  return data;
}

/**
 * Loads an image from a file path, preprocesses it, and returns a Tensor.
 * @param {string} filepath The path to the image file.
 * @returns {Promise<tf.Tensor3D>} The image tensor, normalized and reshaped.
 */
async function loadData(filepath) {
  try {
    // Read file buffer synchronously
    const imageBuffer = fs.readFileSync(filepath);

    // Decode the image buffer into a 3D tensor (height, width, channels)
    // Grayscale is the mean of R, G, B channels, creating a tensor of shape [28, 28, 1]
    const imageTensor = tf.node.decodeImage(imageBuffer, 3);

    // Resize to 28x28
    const resized = tf.image.resizeBilinear(imageTensor, [
      imageWidth,
      imageHeight,
    ]);

    // Convert to Grayscale (Average across the 3 color channels)
    const grayscale = resized.mean(2).expandDims(2);

    // Normalize (0-255 to 0-1)
    const normalized = grayscale.div(255.0);

    // Ensure the result is float32, which is required for CNN input
    return normalized.asType("float32");
  } catch (error) {
    console.error(`Error loading image ${filepath}: ${error.message}`);
    // Return a tensor of zeros if loading fails to prevent crash, though proper error handling is better.
    return tf.zeros([imageWidth, imageHeight, imageChannels], "float32");
  }
}

/**
 * Builds a TensorFlow Dataset from the array of file/label objects.
 * Uses tf.data.generator to load images asynchronously on demand.
 * @param {Array<{filepath: string, label: number}>} data The list of file paths and labels.
 * @returns {tf.data.Dataset} The batched and preprocessed dataset.
 */
function buildDataset(data) {
  return tf.data
    .generator(async function* () {
      for (let i = 0; i < data.length; i++) {
        const { filepath, label } = data[i];

        // Load image data (xs)
        const xs = await loadData(filepath);

        // Create one-hot encoded label (ys)
        const ys = tf.oneHot(label, numOfClasses);

        yield { xs, ys };
      }
    })
    .repeat() // Repeat indefinitely for training (fitDataset handles epoch termination)
    .shuffle(data.length) // Shuffle for better training
    .batch(batchSize)
    .prefetch(4); // Prefetch 4 batches to improve GPU utilization
}

// --- Model Definition, Training, and Evaluation ---
/**
 * Defines and compiles the CNN model architecture.
 * @returns {tf.Sequential} The compiled model.
 */
const buildModel = function () {
  const model = tf.sequential();

  // Input shape is [28, 28, 1]
  model.add(
    tf.layers.conv2d({
      inputShape: [imageWidth, imageHeight, imageChannels],
      filters: 8,
      kernelSize: 5,
      padding: "same",
      activation: "relu",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 2,
      strides: 2,
    })
  );
  model.add(
    tf.layers.conv2d({
      filters: 16,
      kernelSize: 5,
      padding: "same",
      activation: "relu",
    })
  );
  model.add(
    tf.layers.maxPooling2d({
      poolSize: 3,
      strides: 3,
    })
  );
  model.add(tf.layers.flatten());
  model.add(
    tf.layers.dense({
      units: numOfClasses,
      activation: "softmax",
    })
  );

  // Compile the model
  model.compile({
    optimizer: tf.train.adam(0.001), // Explicit learning rate is good practice
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  return model;
};

/**
 * Trains the model against the training data.
 */
const trainModel = async function (
  model,
  trainingData,
  numExamples,
  epochs = epochsValue
) {
  const stepsPerEpoch = Math.ceil(numExamples / batchSize);
  console.log(`\nStarting training for ${numExamples} examples...`);

  const options = {
    epochs: epochs,
    batchesPerEpoch: stepsPerEpoch, // Tell fitDataset how many batches make an epoch
    verbose: 1, // Use verbose 1 for detailed output during training
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          `  > Epoch ${epoch + 1}/${epochs}: Loss: ${logs.loss.toFixed(
            4
          )}, Acc: ${logs.acc.toFixed(4)}`
        );
      },
    },
  };

  return await model.fitDataset(trainingData, options);
};

/**
 * Evaluates the model against the test data.
 */
const evaluateModel = async function (model, testingData, numExamples) {
  const steps = Math.ceil(numExamples / batchSize);
  console.log(`\nStarting evaluation on ${numExamples} test examples...`);

  // We must ensure the test data does not repeat by explicitly calling once
  const result = await model.evaluateDataset(testingData.take(steps), {
    batches: steps,
  });

  const testLoss = result[0].dataSync()[0];
  const testAcc = result[1].dataSync()[0];

  console.log(`  Test Loss: ${testLoss.toFixed(4)}`);
  console.log(`  Test Accuracy: ${testAcc.toFixed(4)}`);
};

// --- Main Execution ---

const run = async function () {
  try {
    // 1. Get and split data
    const allData = getFilesAndLabels();
    const numTrain = Math.floor(allData.length * trainTestSplit);

    const trainDataArr = allData.slice(0, numTrain);
    const testDataArr = allData.slice(numTrain);

    console.log(
      `Train examples: ${trainDataArr.length}, Test examples: ${testDataArr.length}`
    );

    // 2. Build Datasets
    // Note: The datasets are built as generators which load images asynchronously during training.
    const trainDataset = buildDataset(trainDataArr);
    const testDataset = buildDataset(testDataArr);

    // 3. Build Model
    const model = buildModel();
    model.summary();

    // 4. Train Model
    await trainModel(model, trainDataset, trainDataArr.length);

    // 5. Evaluate Model
    console.log("\nEvaluating model...");
    await evaluateModel(model, testDataset, testDataArr.length);

    // 6. Save Model
    const saveModelPath = `file://${path.join(
      __dirname,
      "apple-classifier-tfjs"
    )}`;
    console.log("\nSaving model...");
    await model.save(saveModelPath);
    console.log(
      `Model successfully saved to ${saveModelPath.replace("file://", "")}`
    );
  } catch (e) {
    console.error("An error occurred during the CNN run:", e.message);

    console.log(
      "Please ensure you have run 'npm install @tensorflow/tfjs-node' and created the 'train-images' directory with class subfolders (e.g., 'train-images/Green Apple/')."
    );
  }
};

/// Initialize and run the application, explicitly catching any top-level errors.
run().catch((err) => {
  console.error("Uncaught execution error:", err.message);
  process.exit(1);
});

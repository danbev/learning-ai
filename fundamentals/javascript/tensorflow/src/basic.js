const tf = require('@tensorflow/tfjs-node');

async function run() {
  const model = tf.sequential();
  // a unit is a neuron.
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));

  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

  // Generate some synthetic data for training a regression task.
  // inputs values
  const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  // target values
  const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

  // Train the model using the data.
  await model.fit(xs, ys, {epochs: 10});

  // Use the model to do inference on a data point the model hasn't seen.
  model.predict(tf.tensor2d([5], [1, 1])).print();
}

run();


pub mod perceptron {
    use ndarray::{ArrayView, Ix1, s};

    /// Stores state of the binary classifier; perceptrons are also known as linear units.
    pub struct Perceptron {
       weights: ndarray::Array1<f64>,
       bias: f64,
    }

    impl Perceptron {
        pub fn new(num_features: usize) -> Perceptron {
            Perceptron {
                weights: ndarray::Array1::zeros(num_features),
                bias: 0.0,
            }
        }

        /// Given some perceptrons' weights and biases,
        /// gives a binary output (0 or 1) depending on the calculated output
        /// We use ArrayView to avoid copying the data and get some compile-time guarantees
        /// about input dimensionality.
        pub fn predict(&self, input: ArrayView<f64, Ix1>) -> f64 {
            let mut linear_unit_output = self.weights.dot(&input);
            linear_unit_output += self.bias;
            if linear_unit_output > 0f64 { 1f64 } else { 0f64 }
        }

        /// Given a set of data, normalizes the data to be between 0.0 and 1.0
        pub fn normalize(data: ndarray::Array2<f64>) -> ndarray::Array2<f64>{
            let mut highest = 0f64;
            let mut lowest = f64::INFINITY;
            let mut return_data = data.clone();
            for i in data.iter() {
                if *i > highest {
                    highest = *i;
                }
                if *i < lowest {
                    lowest = *i;
                }
            }
            for i in return_data.iter_mut() {
                *i = (*i - lowest) / (highest - lowest);
            }
            return return_data;
        }

        /// Performs the "Perceptron Algorithm" given some set of data "training_data"
        /// and tries to gain optimal weights.
        ///
        /// The perceptron algorithm is pretty simple:
        /// 0. We find the highest and lowest values, and normalize them to be between 0.0 and 1.0
        /// 1. Given N iterations, or until the weights don't change:
        ///     a.) Iterate over each training sample 'x', with ground truth 'a'
        ///         i.) if a - predict(x) == 0, continue onto next training sample
        ///         ii.) otherwise, we update weights by multiplying the feature by a - predict(x).
        ///
        /// The samples are organized as rows, with the last column being the ground truth.
        pub fn train(&mut self, training_data: ndarray::Array2<f64>, n_iterations: usize) {
            // we find the highest and lowest values, and normalize:
            let normalized_data = Self::normalize(training_data);

            // we iterate over the training data N times, or until the weights don't change
            for _ in 0..n_iterations {
                let prev_weights = self.weights.clone();
                // for each row/sample in the normalized data, we separate out the ground truth
                for row in normalized_data.rows() {
                    // cloning a slice turns the view into an owned array
                    let x = row.slice(s![0..-1]).clone();
                    let a = row[row.len() - 1];
                    let prediction = self.predict(x);
                    if a - prediction == 0f64 { // if the prediction is correct, we continue
                        continue;
                    }
                    // prediction was incorrect, update weights
                    for (i, weight) in self.weights.iter_mut().enumerate() {
                        *weight += (a - prediction) * x[i];
                    }
                }
                // if the weights haven't changed, we break out of the loop
                if prev_weights == self.weights {
                    break;
                }
            }
        }
    }
}
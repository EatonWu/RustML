pub mod perceptron {

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

        /// Given a perceptron's weights and biases,
        /// gives a binary output (0 or 1) depending on the calculated output
        pub fn predict(&self, input: ndarray::Array1<f64>) -> f64 {
            // ensure the input array is the same dimension as the weight array
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
        pub fn train(&self, training_data: ndarray::Array2<f64>) {
            // we find the highest and lowest values, and normalize:
            // let normalized_data = Self::normalize(training_data);
        }
    }

}
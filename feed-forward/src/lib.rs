pub mod perceptron {
    use ndarray::{ArrayView, Ix1, s};
    use rand::prelude::SliceRandom;

    /// Stores state of the binary classifier; perceptrons are also known as linear units.
    pub struct Perceptron {
       weights: ndarray::Array1<f64>,
       bias: f64,
    }

    pub fn correct_labels(labels: &Vec<u8>, class: u8) -> Vec<u8> {
        labels.iter().map(|&x| if x == class { 1 } else { 0 }).collect::<Vec<u8>>()
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
        /// Returns the prediction as well as the linear unit output (confidence?)
        pub fn predict(&self, input: ArrayView<f64, Ix1>) -> (f64, f64) {
            let mut linear_unit_output = self.weights.dot(&input);
            linear_unit_output += self.bias;
            return (if linear_unit_output > 0f64 { 1f64 } else { 0f64 }, linear_unit_output);
        }

        /// Given a set of data, normalizes the data to be between 0.0 and 1.0
        pub fn normalize(data: &ndarray::Array2<u8>) -> ndarray::Array2<f64>{
            let mut highest = u8::MIN;
            let mut lowest = u8::MAX;
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
            // convert to f64
            return_data.mapv(|x| x as f64)
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
        pub fn train(&mut self, training_data: &ndarray::Array2<u8>, training_labels: &Vec<u8>, n_iterations: usize) {
            // we find the highest and lowest values, and normalize:
            let normalized_data = Self::normalize(training_data);

            // we iterate over the training data N times, or until the weights don't change
            for i in 0..n_iterations {
                let prev_weights = self.weights.clone();
                for (idx, row) in normalized_data.outer_iter().enumerate() {
                    // cloning a slice turns the view into an owned array
                    // println!("Row {}: {:?}", idx, row.clone());
                    let x = row.clone(); // the sample
                    let a = training_labels[idx] as f64; // the ground truth
                    let (prediction, _) = self.predict(x);
                    if a - prediction == 0f64 { // if the prediction is correct, we continue
                        // println!("Prediction {} was correct, continuing", prediction);
                        continue;
                    }
                    // prediction was incorrect, update weights
                    if prediction == 0f64 && a == 1f64 { // false negative
                        self.weights += &x;
                    } else if prediction == 1f64 && a == 0f64 { // false positive
                        self.weights -= &x;
                    }
                }
                // if the weights haven't changed, we break out of the loop
                if prev_weights == self.weights {
                    println!("Converged at n = {}, breaking loop", i);
                    break;
                }
            }
        }

        /// Given some set of validation images, and their labels,
        /// and print out the predictions of the model, as well as the accuracy.
        pub fn validate(&self, validation_images: &ndarray::Array2<u8>, validation_labels: &Vec<u8>) {
            let normalized_validation_images = Self::normalize(validation_images);
            let mut correct_predictions = 0;
            for (idx, row) in normalized_validation_images.outer_iter().enumerate() {
                let (prediction, _) = self.predict(row);
                if prediction == validation_labels[idx] as f64 {
                    correct_predictions += 1;
                }
            }
            println!("Accuracy: {}", correct_predictions as f64 / validation_labels.len() as f64);
        }
    }

    // multi-class perceptrons
    pub struct MultiClassPerceptron {
        perceptrons: Vec<Perceptron>,
        classes: Vec<i32>
    }

    impl MultiClassPerceptron {
        pub fn new(classes: Vec<i32>, num_features: usize) -> MultiClassPerceptron {
            let mut perceptrons = vec![];
            for _ in 0..classes.len() {
                perceptrons.push(Perceptron::new(num_features));
            }
            MultiClassPerceptron {
                perceptrons,
                classes,
            }
        }

        pub fn train(&mut self, training_data: &ndarray::Array2<u8>, training_labels: &Vec<u8>, n_iterations: usize) {
            for i in 0..self.classes.len() {
                let mut perceptron = &mut self.perceptrons[i];
                // create the corrected labels
                let corrected_labels = correct_labels(training_labels, self.classes[i] as u8);
                // train the perceptron
                perceptron.train(training_data, &corrected_labels, n_iterations);
            }
        }

        pub fn predict(&self, input: ArrayView<f64, Ix1>) -> usize {
            let mut predictions: Vec<(f64, f64)> = vec![];
            for perceptron in &self.perceptrons {
                predictions.push(perceptron.predict(input));
            }
            // println!("Predictions: {:?}", predictions);

            let ones = predictions.iter().filter(|(prediction, confidence)| *prediction == 1f64).collect::<Vec<_>>();
            return if ones.len() == 0 {
                // return the class with the lowest confidence
                let (idx, _) = predictions.iter().enumerate().min_by(|(_, (_, confidence1)), (_, (_, confidence2))| confidence1.partial_cmp(confidence2).unwrap()).unwrap();
                self.classes[idx] as usize
            } else {
                // return the index of the 1 prediction with the highest confidence
                let (idx, _) = ones.iter().enumerate().max_by(|(_, (_, confidence1)), (_, (_, confidence2))| confidence1.partial_cmp(confidence2).unwrap()).unwrap();
                self.classes[idx] as usize
            }
        }

        pub fn validate(&self, validation_images: &ndarray::Array2<u8>, validation_labels: &Vec<u8>) {
            let normalized_validation_images = Perceptron::normalize(validation_images);
            let mut correct_predictions = 0;

            // iterate over each validation sample
            for (idx, row) in normalized_validation_images.outer_iter().enumerate() {
                // println!("Label: {}", validation_labels[idx] as usize);
                let prediction = self.predict(row);
                if prediction == validation_labels[idx] as usize {
                    correct_predictions += 1;
                }
            }

            println!("Accuracy: {}", correct_predictions as f64 / validation_labels.len() as f64);
        }

        pub fn validate_nth_perceptron(&self, class_idx: usize, validation_images: &ndarray::Array2<u8>, validation_labels: &Vec<u8>) {
            let perceptron = &self.perceptrons[class_idx];
            // correct the validation label
            let corrected_labels = correct_labels(validation_labels, self.classes[class_idx] as u8);
            perceptron.validate(validation_images, &corrected_labels);
        }
    }
}
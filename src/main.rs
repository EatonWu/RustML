use ndarray::{Array2, Axis};
use feed_forward::perceptron::{MultiClassPerceptron, Perceptron};

use mnist_data;
use mnist_data::mnist_data::*;

fn main() {
    let mnist = get_medium_mnist_data();
    match &mnist {
        Ok(_) => {
            // println!("Mini Mnist Data: {:?}", mnist);
        },
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }

    let mnist = mnist.unwrap();
    let images: Array2<u8> = convert_mnist_images_to_ndarray2(mnist.trn_img);
    let labels = mnist.trn_lbl;

    // print number of rows in images and number of labels
    println!("Number of rows in images: {}", images.shape()[0]);
    println!("Number of labels: {}", labels.len());

    let mut model = Perceptron::new(784);

    // set non-zero labels to 0 and zero labels to 1
    let corrected_labels = labels.iter().map(|&x| if x == 0 { 1 } else { 0 }).collect::<Vec<u8>>();

    // we train the model with the contents of the training set
    model.train(&images, &corrected_labels, 10);

    // predict some images
    let test_images: Array2<u8> = convert_mnist_images_to_ndarray2(mnist.tst_img);
    let test_labels = mnist.tst_lbl;
    let corrected_test_labels = test_labels.iter().map(|&x| if x == 0 { 1 } else { 0 }).collect::<Vec<u8>>();

    model.validate(&test_images, &corrected_test_labels);

    // -----------------------

    let mut multi_model = MultiClassPerceptron::new(vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9], 784);
    multi_model.train(&images, &labels, 10);

    for i in 0..10 {
        multi_model.validate_nth_perceptron(i, &test_images, &test_labels);
    }

   multi_model.validate(&test_images, &test_labels);
}
use ndarray::Array2;
use feed_forward::perceptron::Perceptron;

use mnist_data;
use mnist_data::mnist_data::*;

fn main() {
    let mnist = get_mini_mnist_data();
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

    // print the image and the associated label
    for (images_row, label) in images.outer_iter().zip(labels.iter()) {
        println!("Image: {:?}, Label: {:?}", images_row, label);
    }
    
    // normalize the image
    // let normalized_images = Perceptron::normalize(images.mapv(|x| x as f64));
    
    // train the model
    // let mut model = Perceptron::new(784);
    
    
    // we train the model with the contents of the training set
    // model.train(normalized_images, 1);
    
}
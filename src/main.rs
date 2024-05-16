use mnist_data;
use mnist_data::mnist_data::*;
use ndarray:: { Array2, Array3 };

fn main() {
    let mnist = get_mini_mnist_data();
    match &mnist{
        Ok(_) => {
            // println!("Mini Mnist Data: {:?}", mnist);
        },
        Err(e) => {
            println!("Error: {:?}", e);
        }
    }

    let mnist = mnist.unwrap();
    let stuff: Array2<u8> = convert_mnist_images_to_ndarray2(mnist.trn_img);
    println!("{:?}", stuff);
    
}

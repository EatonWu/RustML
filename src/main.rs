use mnist_data;

fn main() {
    mnist_data::mnist_data::download_mnist_dataset().expect("TODO: panic message");
   
}

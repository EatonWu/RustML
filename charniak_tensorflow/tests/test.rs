#[cfg(test)]
mod tests {
    use super::*;
    use tensorflow::{Session, Tensor};
    use tensorflow::ops::constant;
    #[test]
    fn addition_test() {
        // create input variables for addition
        let mut x = Tensor::new(&[1]);
        x[0] = 2;
        let mut y = Tensor::new(&[1]);
        y[0] = 3;

        println!("x: {:?}", x);
        println!("y: {:?}", y);
    }
}
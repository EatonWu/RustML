mod ff_tests {

    #[test]
    fn test_normalize() {
        let data = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let normalized_data= feed_forward::perceptron::Perceptron::normalize(data);
        assert_eq!(normalized_data, ndarray::array![[0f64, 1f64/3f64], [2f64/3f64,  1.0f64]]);
    }

    #[test]
    fn test_normalize_255_vals() {
        let data = ndarray::array![[0.0, 255.0], [127.5, 255.0]];
        let normalized_data= feed_forward::perceptron::Perceptron::normalize(data);
        assert_eq!(normalized_data, ndarray::array![[0f64, 1f64], [0.5f64, 1f64]]);
    }
}
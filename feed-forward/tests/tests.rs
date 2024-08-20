mod ff_tests {

    // #[test]
    // fn test_normalize() {
    //     let normalized_data= feed_forward::perceptron::Perceptron::normalize(data);
    //     assert_eq!(normalized_data, ndarray::array![[0f64, 1f64/3f64], [2f64/3f64,  1.0f64]]);
    // }

    // #[test]
    // fn test_normalize_255_vals() {
    //     let data = ndarray::array![[0.0, 255.0], [127.5, 255.0]];
    //     let normalized_data= feed_forward::perceptron::Perceptron::normalize(data);
    //     assert_eq!(normalized_data, ndarray::array![[0f64, 1f64], [0.5f64, 1f64]]);
    // }
}

mod softmax_tests {
    #[test]
    fn test_softmax() {
        let data = ndarray::array![1.0, 2.0, 3.0];
        let view = data.view();
        let softmaxed_data = feed_forward::cross_entropy::softmax(view);
        assert_eq!(softmaxed_data, ndarray::array![0.09003057317038046, 0.24472847105479767, 0.6652409557748219]);

        // check if the sum of the softmaxed data is approximately 1 (to account for fp errors)
        let sum: f64 = softmaxed_data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
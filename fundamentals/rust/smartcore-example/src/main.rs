use smartcore::dataset::iris::load_dataset;
//use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::metrics::accuracy;
use smartcore::neighbors::knn_classifier::KNNClassifier;

fn main() {
    let iris_data = load_dataset();
    println!("samples: {:?}", iris_data.num_samples);
    println!("features nr: {:?}", iris_data.num_features);
    println!("features: {:?}", iris_data.feature_names);
    println!("target names: {:?}", iris_data.target_names);
    // Turn Iris dataset into NxM matrix
    let x: DenseMatrix<f32> = DenseMatrix::new(
        iris_data.num_samples,  // num rows
        iris_data.num_features, // num columns
        iris_data.data,         // data as Vec
        false,                  // column_major
    );
    // These are our target class labels
    let y = iris_data.target;
    // Fit KNN classifier to Iris dataset
    let knn = KNNClassifier::fit(&x, &y, Default::default()).unwrap();
    let y_hat = knn.predict(&x).unwrap(); // Predict class labels
                                          // Calculate training error
    println!("accuracy: {}", accuracy(&y, &y_hat)); // Prints 0.96
}

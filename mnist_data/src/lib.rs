pub mod mnist_data {
    use std::fs::{File};
    use std::io;
    use std::io::Write;
    use mnist::MnistBuilder;

    /// Downloads the MNIST dataset from a Google Drive and stores it in the data directory.
    /// The downloaded file is initially a zip file, which needs to be extracted.
    /// The extracted files are the training and testing images and labels.
    pub fn download_mnist_dataset() -> Result<(), Box<dyn std::error::Error>>{
        // check if the data directory exists and contains the files
        let data_dir = std::path::Path::new("data");
        if !data_dir.exists() {
            std::fs::create_dir(data_dir).unwrap();
        }

        let train_images = data_dir.join("train-images.idx3-ubyte");
        let train_labels = data_dir.join("train-labels.idx1-ubyte");
        let test_images = data_dir.join("t10k-images.idx3-ubyte");
        let test_labels = data_dir.join("t10k-labels.idx1-ubyte");

        // only do this if the files do not exist
        if !train_images.exists() || !train_labels.exists() || !test_images.exists() || !test_labels.exists() {
            let url = "https://drive.usercontent.google.com/download?id=11ZiNnV3YtpZ7d9afHZg0rtDRrmhha-1E";

            // check if zip exists, if not, download it
            let zip_file = std::path::Path::new("data/mnist.zip");
            if !zip_file.exists() {
                println!("Downloading the MNIST dataset from Google Drive...");
                let response = reqwest::blocking::get(url);

                // check if the response is successful
                match &response {
                    Err(e) => {
                        println!("Error downloading the MNIST dataset: {}", e);
                    },
                    Ok(_) => {
                        println!("Successfully downloaded the MNIST dataset.");
                    }
                }

                let response = response.unwrap();
                if response.status() != 200 {
                    println!("Failed to download the MNIST dataset: {}", response.status());
                    return Ok(());
                }

                // write the response to a file
                let mut output_file = File::create("data/mnist.zip")?;

                // validate that the zip file exists
                let zip_file = std::path::Path::new("data/mnist.zip");
                if zip_file.exists() {
                    println!("MNIST dataset downloaded successfully.");
                    // println!("{:?}", response.bytes().unwrap());
                    output_file.write(response.bytes().unwrap().as_ref())?;
                }
                else {
                    println!("Error downloading the MNIST dataset.");
                }

                drop(output_file);
            }

            // open files with read permissions
            let output_file = File::open("data/mnist.zip").unwrap();

            // unzip the mnist dataset into the data directory
            let file_to_unzip_res = zip::ZipArchive::new(output_file);
            match file_to_unzip_res {
                Err(err) => {
                    println!("Failed to unzip the MNIST dataset: {}", err);
                },
                Ok(mut zip) => {
                    for i in 0..zip.len() {
                        let mut file = zip.by_index(i).unwrap();
                        let file_name = file.name();
                        let file_path = data_dir.join(file_name);
                        let mut output_file = File::create(file_path).unwrap();
                        io::copy(&mut file, &mut output_file).unwrap();
                    }
                }
            }
        }
        else {
            println!("MNIST dataset already exists in the data directory.");
        }

        Ok(())
    }

    /// uses the MNIST crate to extract the images and labels from the dataset
    /// returns an ndarray of the images and labels
    pub fn get_mnist_data() -> Result<ndarray::Array2::<f64>, Box<dyn std::error::Error>>{
        // download the mnist dataset
        let res = download_mnist_dataset()?;

        // if we get here, the dataset has been downloaded

        return Ok(ndarray::Array2::<f64>::zeros((1, 1)));
    }

}

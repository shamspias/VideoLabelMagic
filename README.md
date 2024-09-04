# VideoLabelMagic

VideoLabelMagic is a Streamlit-based application tailored for researchers and developers in the computer vision field.
It simplifies the process of extracting frames from videos, applying object detection using a YOLO model, and annotating
these frames to generate training data.

## Features

- **Video Upload**: Upload video files via the web interface.
- **Model Selection**: Utilize pre-trained YOLO models for object detection.
- **Frame Rate Control**: Adjust the frame rate for extracting images from the video.
- **Dynamic Class Configuration**: Use YAML files to define and utilize different class configurations for object
  detection.
- **Output Customization**: Configure output directories for storing extracted frames and annotations.
- **Transformation Options**: Apply transformations such as resizing, converting to grayscale, or rotating frames.
- **Flexible Storage**: Choose between local file system or cloud-based object storage for input/output operations.

## Usage

1. **Starting the Application**:
    - Launch the application and access it via `http://localhost:8501` on your browser.

2. **Uploading and Configuring**:
    - Upload a video file or select one from the configured object storage.
    - Choose the detection model, class configuration, and specify the output directory and frame rate.
    - Select desired transformations for the frames to be processed.

3. **Processing**:
    - Click "Extract Frames" to start the frame extraction and annotation process.
    - Once processing completes, the outputs can be found in the specified directory or uploaded to cloud storage.

4. **Viewing Results**:
    - Access extracted images and annotations directly from the output directory or your cloud storage interface.

## Creating Class Configuration Files

To customize object detection classes, you need to create a YAML file specifying each class and its corresponding ID.
Here's how to set up your YAML file for dynamic class configuration:

1. **File Format**: Each class entry should contain an `id` and `name`. For example:

    ```yaml
    classes:  
      - id: 0  
        name: person  
      - id: 1  
        name: car  
      - id: 2  
        name: truck  
      - id: 3  
        name: tank
    ```

2. **Saving the File**: Save the file with a `.yaml` extension in the `object_class/` directory.
3. **Using in Application**: When running the application, select your new class configuration file from the dropdown
   menu.

### Prerequisites

- Python 3.11+
- Pipenv or virtualenv (recommended for package management)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/shamspias/VideoLabelMagic.git
   cd VideoLabelMagic
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit application:
   ```bash
   streamlit run app/main.py
   ```

## Contributing

Contributions to VideoLabelMagic are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on
how to make contributions.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
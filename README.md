# VideoLabelMagic

VideoLabelMagic is a Streamlit-based application designed to facilitate the process of extracting frames from video files, detecting objects within those frames using a YOLO model, and annotating them accordingly. This tool is particularly useful for researchers and developers working on computer vision projects requiring training data preparation.

## Features

- **Video Upload**: Users can upload video files directly through the web interface.
- **Model Selection**: Choose from pre-trained YOLO models to detect objects.
- **Frame Rate Control**: Specify the frame rate for extracting images from the video.
- **Dynamic Class Configuration**: Utilize different class configurations for object detection.
- **Output Customization**: Set output directories for saving extracted frames and annotations.

## Getting Started

### Prerequisites

- Python 3.8+
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

### Usage

After launching the application, navigate to `http://localhost:8501` in your web browser. From there, you can:

- Upload a video file.
- Select a detection model and class configuration.
- Specify the output directory and frame rate.
- Click "Extract Frames" to process the video.

## Contributing

Contributions to VideoLabelMagic are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to make contributions.

## License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
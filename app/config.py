from pydantic_settings import BaseSettings


class Config(BaseSettings):
    streamlit_title: str = "VideoLabelMagic"
    models_directory: str = "models/"
    output_directory: str = "outputs/"
    object_class_directory: str = "object_class/"
    default_frame_rate: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

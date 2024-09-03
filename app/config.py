from pydantic_settings import BaseSettings


class Config(BaseSettings):
    streamlit_title: str = "VideoLabelMagic"
    model_directory: str = "models/"
    output_directory: str = "outputs/"
    default_frame_rate: float = 1.0

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

from pydantic_settings import BaseSettings
from typing import Optional


class Config(BaseSettings):
    streamlit_title: Optional[str] = "VideoLabelMagic"
    models_directory: Optional[str] = "models/"
    output_directory: Optional[str] = "outputs/"
    object_class_directory: Optional[str] = "object_class/"
    default_frame_rate: Optional[float] = 1.0

    # Object storage settings
    storage_use_s3: Optional[bool] = False
    s3_endpoint_url: Optional[str] = ""
    s3_access_key: Optional[str] = ""
    s3_secret_key: Optional[str] = ""
    s3_bucket_name: Optional[str] = ""
    s3_region_name: Optional[str] = ""

    # SAHI settings
    sahi_enabled: Optional[bool] = False
    sahi_model_type: Optional[str] = 'yolov8'
    sahi_device: Optional[str] = 'cpu'
    sahi_slice_size: Optional[tuple] = (256, 256)
    sahi_overlap_ratio: Optional[tuple] = (0.2, 0.2)

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'

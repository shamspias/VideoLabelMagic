import boto3
from botocore.exceptions import NoCredentialsError


class StorageManager:
    def __init__(self, config):
        self.config = config
        if self.config.storage_use_s3:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.config.s3_endpoint_url,
                aws_access_key_id=self.config.s3_access_key,
                aws_secret_access_key=self.config.s3_secret_key,
                region_name=self.config.s3_region_name
            )

    def upload_file_to_s3(self, file_name, object_name=None):
        """Upload a file to an S3 bucket"""
        if object_name is None:
            object_name = file_name
        try:
            response = self.s3_client.upload_file(file_name, self.config.s3_bucket_name, object_name)
            return response
        except FileNotFoundError:
            print("The file was not found")
            return None
        except NoCredentialsError:
            print("Credentials not available")
            return None

    def download_file_from_s3(self, object_name, file_name):
        """Download a file from an S3 bucket"""
        try:
            self.s3_client.download_file(self.config.s3_bucket_name, object_name, file_name)
            return file_name
        except NoCredentialsError:
            print("Credentials not available")
            return None

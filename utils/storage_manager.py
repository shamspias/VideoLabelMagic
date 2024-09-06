import boto3
import os
from botocore.exceptions import NoCredentialsError, ClientError


class StorageManager:
    """
    Handles interactions with S3-compatible object storage, including listing, downloading, and uploading files.
    """

    def __init__(self, config):
        """
        Initialize the S3 client with custom endpoint and credentials from the configuration.
        """
        self.config = config

        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.config.s3_endpoint_url,  # Custom S3-compatible endpoint URL
                aws_access_key_id=self.config.s3_access_key,
                aws_secret_access_key=self.config.s3_secret_key
            )
            # print(f"Connected to S3 endpoint: {self.config.s3_endpoint_url} (Region: {self.config.s3_region_name})")
        except Exception as e:
            raise RuntimeError(f"Error initializing S3 client: {str(e)}")

    def list_files_in_bucket(self):
        """
        List all files in the configured S3 bucket.
        Returns:
            List[str]: A list of file keys from the bucket or an empty list if no files exist or an error occurs.
        """
        try:
            # Check if the bucket exists
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket_name)
            response = self.s3_client.list_objects_v2(Bucket=self.config.s3_bucket_name)

            if 'Contents' in response:
                files = [item['Key'] for item in response['Contents']]
                return files
            else:
                return []
        except ClientError as e:
            error_message = e.response['Error']['Message']
            print(f"Error accessing S3 bucket: {error_message}")
            return []
        except Exception as e:
            print(f"Unexpected error accessing S3 bucket: {str(e)}")
            return []

    def download_file_from_s3(self, object_name, local_path):
        """
        Download a file from the S3 bucket to a local path. Ensures that the local directory exists.
        Args:
            object_name (str): The name of the object in the S3 bucket.
            local_path (str): The local file path to download the object to.
        """
        try:
            # Ensure the local directory exists
            local_dir = os.path.dirname(local_path)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
                print(f"Created local directory: {local_dir}")

            print(f"Attempting to download {object_name} from bucket '{self.config.s3_bucket_name}' to {local_path}...")
            self.s3_client.download_file(self.config.s3_bucket_name, object_name, local_path)
            print(f"Downloaded '{object_name}' to '{local_path}'")
        except NoCredentialsError:
            print("Credentials not available for downloading the file.")
        except ClientError as e:
            error_message = e.response['Error']['Message']
            print(f"Error downloading file from S3: {error_message}")
        except Exception as e:
            print(f"Unexpected error downloading file from S3: {str(e)}")

    def upload_file_to_s3(self, local_path, object_name):
        """
        Upload a file from the local file system to the S3 bucket.
        Args:
            local_path (str): The path to the local file to upload.
            object_name (str): The name to assign to the object in the S3 bucket.
        """
        try:
            self.s3_client.upload_file(local_path, self.config.s3_bucket_name, object_name)
            print(f"Uploaded '{local_path}' to '{object_name}' in bucket '{self.config.s3_bucket_name}'")
        except NoCredentialsError:
            print("Credentials not available for uploading the file.")
        except ClientError as e:
            error_message = e.response['Error']['Message']
            print(f"Error uploading file to S3: {error_message}")
        except Exception as e:
            print(f"Unexpected error uploading file to S3: {str(e)}")

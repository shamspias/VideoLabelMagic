import boto3
from botocore.exceptions import NoCredentialsError, ClientError


class StorageManager:
    def __init__(self, config):
        self.config = config
        # Initialize S3 client with proper endpoint, credentials, and region
        try:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.config.s3_endpoint_url,  # Custom S3-compatible endpoint URL
                aws_access_key_id=self.config.s3_access_key,
                aws_secret_access_key=self.config.s3_secret_key
            )

            print(f"Connected to S3 endpoint: {self.config.s3_endpoint_url} (Region: {self.config.s3_region_name})")
        except Exception as e:
            print(f"Error initializing S3 client: {str(e)}")

    def list_files_in_bucket(self):
        """
        List all files in the configured S3 bucket.
        """
        try:
            # Check if the bucket exists by calling head_bucket
            print(f"Checking if bucket '{self.config.s3_bucket_name}' exists...")
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket_name)
            print(f"Bucket '{self.config.s3_bucket_name}' exists, listing files...")

            response = self.s3_client.list_objects_v2(Bucket=self.config.s3_bucket_name)

            if 'Contents' in response:
                files = [item['Key'] for item in response['Contents']]
                print(f"Found {len(files)} files in bucket '{self.config.s3_bucket_name}'.")
                return files
            else:
                print(f"No files found in bucket '{self.config.s3_bucket_name}'.")
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
        Download a file from an S3 bucket to a local path.
        """
        try:
            print(f"Attempting to download {object_name} from bucket '{self.config.s3_bucket_name}' to {local_path}...")
            self.s3_client.download_file(self.config.s3_bucket_name, object_name, local_path)
            print(f"Downloaded {object_name} to {local_path}")
        except NoCredentialsError:
            print("Credentials not available")
        except ClientError as e:
            error_message = e.response['Error']['Message']
            print(f"Error downloading file from S3: {error_message}")
        except Exception as e:
            print(f"Unexpected error downloading file from S3: {str(e)}")

    def upload_file_to_s3(self, local_path, object_name):
        """
        Upload a file to an S3 bucket.
        """
        try:
            print(f"Attempting to upload {local_path} to bucket '{self.config.s3_bucket_name}' as {object_name}...")
            self.s3_client.upload_file(local_path, self.config.s3_bucket_name, object_name)
            print(f"Uploaded {local_path} to {object_name}")
        except NoCredentialsError:
            print("Credentials not available")
        except ClientError as e:
            error_message = e.response['Error']['Message']
            print(f"Error uploading file to S3: {error_message}")
        except Exception as e:
            print(f"Unexpected error uploading file to S3: {str(e)}")

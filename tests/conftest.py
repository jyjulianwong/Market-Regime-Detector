import os

# Settings derives an unset MRD_S3_BUCKET_NAME from the active AWS account via
# STS. Pin it for the test suite so modules that call get_settings() at import
# time never need real credentials.
os.environ.setdefault("MRD_S3_BUCKET_NAME", "test-jyjulianwong-mrd")

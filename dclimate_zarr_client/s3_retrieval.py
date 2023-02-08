import os
from s3fs import S3FileSystem, S3Map
import xarray as xr

BUCKET = "s3://zarr-dev"


def get_dataset_from_s3(dataset_name):
    if "aws_key" in os.environ and "aws_secret" in os.environ:
        s3fs = S3FileSystem(key=os.environ["aws_key"], secret=os.environ["aws_secret"])
    else:
        s3fs = S3FileSystem(anon=False)
    s3_map = S3Map(f"{BUCKET}/{dataset_name}", s3=s3fs)
    return xr.open_zarr(s3_map)

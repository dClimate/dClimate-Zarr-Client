import os
from s3fs import S3FileSystem, S3Map
import json
import xarray as xr

BUCKET = "s3://zarr-dev"


def get_s3_fs() -> S3FileSystem:
    """Gets an S3 filesystem based on provided credentials

    Returns:
        S3FileSystem:
    """
    if "aws_key" in os.environ and "aws_secret" in os.environ:
        return S3FileSystem(key=os.environ["aws_key"], secret=os.environ["aws_secret"])
    else:
        return S3FileSystem(anon=False)


def get_dataset_from_s3(dataset_name: str) -> xr.Dataset:
    """Get a dataset from s3 from its name

    Args:
        dataset_name (str): key for datasets

    Returns:
        xr.Dataset: dataset correponding to key
    """
    s3_map = S3Map(f"{BUCKET}/{dataset_name}.zarr", s3=get_s3_fs())
    return xr.open_zarr(s3_map)


def get_metadata_by_s3_key(key: str) -> dict:
    """Get metadata for specific dataset

    Args:
        key (str): dataset key

    Returns:
        dict: metadata corresponding to key
    """
    s3 = get_s3_fs()
    attr_text = s3.cat(f"{BUCKET}/{key}.zarr/.zattrs")
    return json.loads(attr_text)

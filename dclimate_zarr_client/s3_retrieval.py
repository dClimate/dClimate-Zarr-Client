import datetime
import os
from s3fs import S3FileSystem, S3Map
import typing
import json
import numpy as np
import xarray as xr

from dclimate_zarr_client.dclimate_zarr_errors import DatasetNotFoundError



def get_s3_fs() -> S3FileSystem:
    """Gets an S3 filesystem based on provided credentials

    Returns:
        S3FileSystem:
    """
    if "aws_key" in os.environ and "aws_secret" in os.environ:
        return S3FileSystem(key=os.environ["aws_key"], secret=os.environ["aws_secret"])
    else:
        return S3FileSystem(anon=False)


def get_dataset_from_s3(dataset_name: str, bucket_name: str, forecast_hour: int = None) -> xr.Dataset:
    """Get a dataset from s3 from its name

    Args:
        dataset_name (str): key for datasets
        bucket_name (str): bucket name from where the datasets are fetched

    Returns:
        xr.Dataset: dataset corresponding to key
    """
    try:
        s3_map = S3Map(f"s3://{bucket_name}/datasets/{dataset_name}.zarr", s3=get_s3_fs(), check=True)
        ds = xr.open_zarr(s3_map, chunks=None)
    except ValueError:
        raise DatasetNotFoundError("Invalid dataset name")
    if forecast_hour:
        ds = filter_forecast_dataset(ds, forecast_hour)
    if ds.update_in_progress:
        if ds.update_is_append_only:
            start, end = ds.attrs["date range"][0], ds.attrs["update_previous_end_date"]
        else:
            start, end = ds.attrs["date range"]
        if end is None:
            raise DatasetNotFoundError(
                "Dataset is undergoing initial parse, retry request later"
            )
        date_range = slice(
            *[datetime.datetime.strptime(t, "%Y%m%d%H") for t in (start, end)]
        )
        ds = ds.sel(time=date_range)

    return ds


def filter_forecast_dataset(ds: xr.Dataset, forecast_hour: int) -> xr.Dataset:
    """
    Filter a 4D forecast dataset to a 3D dataset ready for analysis

    Args:
        xr.Dataset: 4D Xarray dataset containing time (forecast_reference_time) and forecast hour (step) dimensions
        forecast_hor (int): integer representing the desired forecast hour

    Returns:
        xr.Dataset: 3D Xarray dataset with forecast hour added to time dimension
    """
    # select the desired forecast hour; must convert to the nanosecond unit used internally by timedelta64
    ds = ds.sel(step=np.timedelta64(forecast_hour * 3600000000000))
    # Keep time dim name consistent
    ds = ds.rename({"forecast_reference_time" : "time"})
    # Set time to equal the forecast time, not the forecast reference time. Assumes only one forecast step is returned
    ds = ds.assign_coords(time=ds.time.values + ds.step.values)
    # Remove forecast hour
    ds = ds.squeeze().drop('step')
    return ds


def list_s3_datasets(bucket_name: str) -> typing.List[str]:
    """List all datasets available over s3

     Args:
        bucket_name (str): bucket name from where the datasets are fetched

    Returns:
        list[str]: available datasets
    """
    s3 = get_s3_fs()
    root_keys = s3.ls(f"s3://{bucket_name}/datasets")
    file_names = [key.split("/")[-1] for key in root_keys]
    zarr_names = [name[:-5] for name in file_names if name.endswith(".zarr")]
    return zarr_names


def get_metadata_by_s3_key(key: str, bucket_name: str) -> dict:
    """Get metadata for specific dataset

    Args:
        key (str): dataset key
        bucket_name (str): bucket name from where the datasets are fetched

    Returns:
        dict: metadata corresponding to key
    """
    s3 = get_s3_fs()
    try:
        attr_text = s3.cat(f"s3://{bucket_name}/datasets/{key}.zarr/.zattrs")
    except FileNotFoundError:
        raise DatasetNotFoundError("Invalid dataset name")
    return json.loads(attr_text)

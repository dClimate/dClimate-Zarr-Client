import datetime
import typing

import numpy as np
import geopandas as gpd
import xarray as xr
from shapely.ops import unary_union

from dclimate_zarr_client.dclimate_zarr_errors import (
    InvalidAggregationMethodError,
    InvalidTimePeriodError,
    SelectionTooLargeError,
    NoDataFoundError,
)

# Users should not select more than this number of data points and coordinates
DEFAULT_POINT_LIMIT = 40 * 40 * 50_000


def check_dataset_size(ds: xr.Dataset, point_limit: int = DEFAULT_POINT_LIMIT):
    """Checks how many data points are in a dataset

    Args:
        ds (xr.Dataset): dataset to check size of
        point_limit (int, optional): limit for dataset size. Defaults to DEFAULT_POINT_LIMIT.

    Raises:
        SelectionTooLargeError: Raised when dataset size limit is violated
    """
    # Go through each of the dimensions and check whether they exist and if not set them to 1
    dim_lengths = []
    for dim in ["latitude", "longitude", "time"]:
        try:
            dim_lengths.append(len(ds[dim]))
        except (KeyError, TypeError):
            dim_lengths.append(1)
    num_points = np.prod(dim_lengths)
    # check number of points against the agreed limit
    if num_points > point_limit:
        raise SelectionTooLargeError(
            f"Selection of {num_points} data points is more than limit of {point_limit}"
        )


def check_has_data(ds: xr.Dataset):
    """Checks if data is all NA

    Args:
        ds (xr.Dataset): dataset to check

    Raises:
        NoDataFoundError: Raised when data is all NA
    """
    try:
        var_name = list(ds.data_vars)[0]
        if ds[var_name].isnull().all():
            raise NoDataFoundError("Selection is empty or all NA")
    except ValueError:
        raise NoDataFoundError("Selection is empty or all NA")


def _check_input_parameters(time_period=None, agg_method=None):
    """Checks whether input parameters align with permitted time periods and aggregation methods

    Args:
        time_period (str, optional): a string specifying the time period to resample a dataset by
        agg_method (str, optional): a string specifying the aggregation method to use on a dataset

    Raises:
        InvalidTimePeriodError: Raised when the specified time period is not accepted
        InvalidAggregationMethodError: Raised when the specified aggregation method is not accepted
    """
    if time_period and time_period not in [
        "hour",
        "day",
        "week",
        "month",
        "quarter",
        "year",
        "all",
    ]:
        raise InvalidTimePeriodError(
            f"Specified time period {time_period} not among permitted periods: \
                'hour', 'day', 'week', 'month', 'quarter', 'year', 'all'"
        )
    if agg_method and agg_method not in ["min", "max", "median", "mean", "std", "sum"]:
        raise InvalidAggregationMethodError(
            f"Specified method {agg_method} not among permitted methods: \
                'min', 'max', 'median', 'mean', 'std', 'sum'"
        )


def get_single_point(ds: xr.Dataset, lat: float, lon: float, snap_to_grid: bool = True) -> xr.Dataset:
    """Gets a dataset corresponding to the full time series for a single point in a dataset

    Args:
        ds (xr.Dataset): dataset to subset
        lat (float): latitude coordinate
        lon (float): longitude coordinate
        snap_to_grid (bool): when True, find nearest point to lat, lon in dataset.
            When false, error out when exact lat, lon is not on dataset grid.

    Returns:
        xr.Dataset: subsetted dataset
    """
    if snap_to_grid:
        return ds.sel(latitude=lat, longitude=lon, method="nearest")
    try:
        return ds.sel(latitude=lat, longitude=lon, method="nearest", tolerance=10e-5)
    except KeyError:
        raise NoDataFoundError("User requested not to snap_to_grid, but exact coord not in dataset")


def get_multiple_points(
    ds: xr.Dataset, points_mask: gpd.array.GeometryArray, epsg_crs: int, snap_to_grid: bool = True
) -> dict:
    mask = list(gpd.geoseries.GeoSeries(points_mask).set_crs(epsg_crs).to_crs(4326))
    lats, lons = [point.y for point in mask], [point.x for point in mask]
    lats, lons = xr.DataArray(lats, dims="point"), xr.DataArray(lons, dims="point")
    if snap_to_grid:
        return ds.sel(latitude=lats, longitude=lons, method="nearest")
    try:
        return ds.sel(latitude=lats, longitude=lons, method="nearest", tolerance=10e-5)
    except KeyError:
        raise NoDataFoundError(
            "User requested not to snap_to_grid, but at least one coord not in dataset")


def _haversine(
    lat1: typing.Union[np.ndarray, float],
    lon1: typing.Union[np.ndarray, float],
    lat2: typing.Union[np.ndarray, float],
    lon2: typing.Union[np.ndarray, float],
) -> typing.Union[np.ndarray, float]:
    """Calculates arclength distance in km between coordinate pairs,
        assuming the earth is a perfect sphere

    Args:
        lat1 (typing.Union[np.ndarray, float]): latitude coordinate for first point
        lon1 (typing.Union[np.ndarray, float]): longitude coordinate for first point
        lat2 (typing.Union[np.ndarray, float]): latitude coordinate for second point
        lon2 (typing.Union[np.ndarray, float]): longitude coordinate for second point

    Returns:
        typing.Union[np.ndarray, float]: distance between coordinate pairs in km
    """
    # convert decimal degrees to radians
    lon1 = np.deg2rad(lon1)
    lon2 = np.deg2rad(lon2)
    lat1 = np.deg2rad(lat1)
    lat2 = np.deg2rad(lat2)

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))

    # radius of earth in km
    r = 6371
    return c * r


def get_points_in_circle(
    ds: xr.Dataset,
    center_lat: float,
    center_lon: float,
    radius: float,
) -> xr.Dataset:
    """Subsets dataset to points within radius of given center coordinates

    Args:
        ds (xr.Dataset): dataset to subset
        center_lat (float): latitude coordinate of center
        center_lon (float): longitude coordinate of center
        radius (float): radius of circle in kilometers

    Returns:
        xr.Dataset: subsetted dataset
    """
    distances = _haversine(center_lat, center_lon, ds["latitude"], ds["longitude"])
    try:
        circle_ds = ds.where(distances < radius, drop=True)
    except ValueError:
        raise NoDataFoundError("Selection is empty or all NA")
    return circle_ds


def get_points_in_rectangle(
    ds: xr.Dataset,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
) -> xr.Dataset:
    """Subsets dataset to points in rectangle

    Args:
        ds (xr.Dataset): dataset to subset
        min_lat (float): southern limit of rectangle
        min_lon (float): western limit of rectangle
        max_lat (float): northern limit of rectangle
        max_lon (float): eastern limit of rectangle

    Returns:
        xr.Dataset: subsetted dataset
    """
    try:
        rectangle_ds = ds.where(
            (ds.latitude >= min_lat)
            & (ds.latitude <= max_lat)
            & (ds.longitude >= min_lon)
            & (ds.longitude <= max_lon),
            drop=True,
        )
    except ValueError:
        raise NoDataFoundError("Selection is empty or all NA")
    return rectangle_ds


def get_points_in_polygons(
    ds: xr.Dataset,
    polygons_mask: gpd.array.GeometryArray,
    epsg_crs: int = 4326,
    point_limit=DEFAULT_POINT_LIMIT,
) -> xr.Dataset:
    """Subsets dataset to points within arbitrary shape. Requires rioxarray to be installed

    Args:
        ds (xr.Dataset): dataset to subset
        polygons_mask (gpd.array.GeometryArray[shapely.geometry.multipolygon.MultiPolygon]):
            list (GeometryArray) of MultiPolygon shapes defining the area of interest
        epsg_crs (int): epsg code for polygons_mask (see https://en.wikipedia.org/wiki/EPSG_Geodetic_Parameter_Dataset)

    Returns:
        xr.Dataset: subsetted dataset
    """
    # If the polygon(s) are collectively smaller than the size of one grid cell, clipping will return no data
    # In this case return data from the grid cell nearest to the center of the polygon
    if ds.attrs["spatial resolution"]**2 > polygons_mask.unary_union().area:
        rep_point_ds = reduce_polygon_to_point(ds, polygons_mask)
        return rep_point_ds
    # return clipped data as normal if the polygons are large enough
    else:
        ds.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
        ds.rio.write_crs("epsg:4326", inplace=True)
        mask = gpd.geoseries.GeoSeries(polygons_mask).set_crs(epsg_crs).to_crs(4326)
        min_lon, min_lat, max_lon, max_lat = mask.total_bounds
        box_ds = get_points_in_rectangle(ds, min_lat, min_lon, max_lat, max_lon)
        check_dataset_size(box_ds, point_limit=point_limit)
        shaped_ds = box_ds.rio.clip(mask, 4326, drop=True)
        data_var = list(shaped_ds.data_vars)[0]
        if "grid_mapping" in shaped_ds[data_var].attrs:
            del shaped_ds[data_var].attrs["grid_mapping"]
        return shaped_ds


def get_data_in_time_range(
    ds: xr.Dataset, start_time: datetime.datetime, end_time: datetime.datetime
) -> xr.Dataset:
    """Subset dataset to be within a contiguous time range. Can be combined with spatial subsetters defined above

    Args:
        ds (xr.Dataset): dataset to subset
        start_time (datetime.datetime): beginning of time range
        end_time (datetime.datetime): end of time range

    Returns:
        xr.Dataset: subsetted dataset
    """
    try:
        return ds.sel(time=slice(str(start_time), str(end_time)))
    except ValueError:
        raise NoDataFoundError("No data found in time range.")


def reduce_polygon_to_point(
    ds: xr.Dataset, polygons_mask: gpd.array.GeometryArray
) -> xr.Dataset:
    """Subsets data to a representative point approximately at the center of an arbitrary shape. 
        Returns data from the nearest grid cell to this pont.
        NOTE a more involved alternative would be to return the average for values in the entire polygon at this point

    Args:
        ds (xr.Dataset): dataset to subset
        polygons_mask (gpd.array.GeometryArray[shapely.geometry.multipolygon.MultiPolygon]):
            list (GeometryArray) of MultiPolygon shapes defining the area of interest

    Returns:
        xr.Dataset: subsetted dataset
    """
    pt = unary_union(polygons_mask).representative_point()
    ds = ds.sel(latitude=pt.y, longitude=pt.x, method="nearest")
    return ds


def spatial_aggregation(
    ds: xr.Dataset,
    agg_method: str,
) -> xr.Dataset:
    """Subsets data for all points for every time period according to the specified aggregation method.
       For a more nuanced treatment of spatial units use the `get_points_in_polygons` method.

    Args:
        ds (xr.Dataset): dataset to subset
        agg_method (str): method to aggregate by

    Returns:
        xr.Dataset: subsetted dataset
    """
    _check_input_parameters(agg_method=agg_method)
    spatial_dims = [dim for dim in ds.dims if dim != "time"]
    # Aggregate by the specified method across all time periods
    aggregator = getattr(xr.Dataset, agg_method)
    return aggregator(ds, spatial_dims, keep_attrs=True)


def temporal_aggregation(
    ds: xr.Dataset,
    time_period: str,
    agg_method: str,
    time_unit: int = 1,
) -> xr.Dataset:
    """Subsets data according to a specified combination of time period, units of time, \
        aggregation method, and/or desired spatial unit.
       Time-based inputs defualt to the entire time range and 1 unit of time, respectively.
       Spatial units default to points, i.e. every combination of latitudes/longitudes. The only alternative is "all".
       For a more nuanced treatment of spatial units use the `get_points_in_polygons` method.

    Args:
        ds (xr.Dataset): dataset to subset
        time_period (str): time period to aggregate by, parsed into DateOffset objects as per \
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
        agg_method (str): method to aggregate by
        time_unit (int): number of time periods to aggregate by. Default is 1. Ignored if "all" time periods specified.

    Returns:
        xr.Dataset: subsetted dataset
    """
    _check_input_parameters(time_period=time_period, agg_method=agg_method)
    period_strings = {
        "hour": f"{time_unit}H",
        "day": f"{time_unit}D",
        "week": f"{time_unit}W",
        "month": f"{time_unit}M",
        "quarter": f"{time_unit}Q",
        "year": f"{time_unit}Y",
        "all": f"{len(set(ds.time.dt.year.values))}Y",
    }
    # Resample by the specified time period and aggregate by the specified method
    resampled = ds.resample(time=period_strings[time_period])
    aggregator = getattr(xr.core.resample.DatasetResample, agg_method)
    resampled_agg = aggregator(resampled, keep_attrs=True)

    return resampled_agg


def rolling_aggregation(
    ds: xr.Dataset,
    window_size: int,
    agg_method: str,
) -> xr.Dataset:
    """Subsets data to a rolling aggregate of data values along a dataset's "time" dimension.
        The size of the window and the aggregation method are specified by the user.
        Method must be one of "min", "max", "median", "mean", "std", or "sum".

    Args:
        ds (xr.Dataset): dataset to subset
        window_size (int): size of rolling window to apply
        agg_method (str): method to aggregate by

    Returns:
        xr.Dataset: subsetted dataset
    """
    _check_input_parameters(agg_method=agg_method)
    # Aggregate by the specified method over the specified rolling window length
    rolled = ds.rolling(time=window_size)
    aggregator = getattr(xr.core.rolling.DatasetRolling, agg_method)
    rolled_agg = aggregator(rolled, keep_attrs=True).dropna(
        "time"
    )  # remove NAs at beginning/end of array where window size is not large enough to compute a value

    return rolled_agg

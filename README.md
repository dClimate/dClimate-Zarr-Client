# dClimate-Zarr-Client
Retrieve zarrs stored on IPLD.

Uses [ipldstore](https://github.com/dClimate/ipldstore) to actually access zarrs, then provides
filtering and aggregation functionality to these zarrs using `xarray` native methods wherever possible.
Filtering and aggregation are packaged into a minimal number of convenience functions optimized for flexbility
and performance.

A limit is imposed on the total number of points users can request to prevent overwhelming the API. This limit
can be manually overridden in exceptional cases.

The main entrypoint to the repo's code is `dclimate_zarr_client.client.geo_temporal_query`


## File breakdown:

### client.py

Entrypoint to code, contains `geo_temporal_query`, which combines all possible subsetting
and aggregation logic in a single function. Can output the data as either a `dict`
or `bytes` representing an `xarray` dataset.

---

### dclimate_zarr_errors.py

Various exceptions to be raised for bad or invalid user input.

---

### geo_utils.py

Functions to manipulate `xarray` datasets. Contains polygon, rectangle, circle and point spatial
subsetting options, as well as temporal subsetting. Also allows for both spatial and temporal
aggregations.

---

### ipfs_retrieval.py

Functions for accessing zarrs over IPFS/IPNS. Functionality includes resolving IPNS keys to IPFS hashes
based on key names, as well as using `ipldstore` to open the zarrs that those IPFS hashes point to.


##  Usage:

While in virtual environment and at root of repo, run `pip install -e .` to install the package's core functionality.
To install with extra packages for development and testing, run `pip install -e .\[testing,dev]`. Then you can run python
code like:

```python
# Singleton Function Interface
from datetime import datetime
import xarray as xr
import dclimate_zarr_client as client
ds_name = "era5_wind_100m_u-hourly"
ds_bytes = client.geo_temporal_query(
    ds_name,
    point_kwargs={"lat": 40, "lon": -120},
    time_range=[datetime(2021, 1, 1), datetime(2022, 12, 31)],
    output_format="netcdf"
)
ds = xr.open_dataset(ds_bytes)

# Pythonic Interface
dataset = client.load_ipns(ds_name)
dataset = dataset.point(lat=40, lon=-120)
dataset = dataset.time_range(datetime(2021, 1, 1), datetime(2022, 12, 31))
ds_bytes = dataset.to_netcdf()

ds = xr.open_dataset(ds_bytes)
```
## Run tests for your local environment:
```shell
cd
pytest tests
```

## Run all acceptance tests:
```shell
nox
```

## Environment requirements:

- Running IPFS daemon
- Dataset parsed with [gridded-etl-tools](https://github.com/Arbol-Project/gridded-etl-tools/) with name `ds_name`
- Up-to-date IPNS table (IPNS key for `ds_name` can't be expired).
  If `ipfs name resolve <ipns key>` stalls out, the IPNS key is expired and `publish_metadata` step of ETL must be rerun.

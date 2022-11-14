# dClimate-Zarr-Client
Retrieve zarrs stored on IPLD

### Example usage:

```python
from datetime import datetime
import xarray as xr
from dclimate_zarr_client.client import geo_temporal_query
ds_name = "era5_wind_100m_u-hourly"
ds_bytes = geo_temporal_query(
    ds_name,
    point_kwargs={"lat": 40, "lon": -120},
    time_range=[datetime(2021, 1, 1), datetime(2022, 12, 31)],
    output_format="netcdf"
)
ds = xr.open_dataset(ds_bytes)
```
### Run tests:
```shell
cd test
pytest
```

### Environment requirements:

- Running IPFS daemon
- Dataset parsed with `climate_ipfs` branch `zarr-main` with name `ds_name`
- Up-to-date IPNS table (IPNS key for `ds_name` can't be expired)

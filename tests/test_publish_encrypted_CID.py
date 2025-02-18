import os
import shutil
import tempfile

from multiformats import CID
import numpy as np
import pandas as pd
import xarray as xr
import pytest
from numcodecs import register_codec

from Crypto.Random import get_random_bytes

from py_hamt import HAMT, IPFSStore

from dclimate_zarr_client.encryption_codec import EncryptionCodec


@pytest.fixture
def random_zarr_dataset():
    """Creates a random xarray Dataset and saves it to a temporary zarr store.
    Returns:
        tuple: (dataset_path, expected_data)
            - dataset_path: Path to the zarr store
            - expected_data: The original xarray Dataset for comparison
    """
    # Create temporary directory for zarr store
    temp_dir = tempfile.mkdtemp()
    zarr_path = os.path.join(temp_dir, "test.zarr")

    # Coordinates of the random data
    times = pd.date_range("2024-01-01", periods=100)
    lats = np.linspace(-90, 90, 18)
    lons = np.linspace(-180, 180, 36)

    # Create random variables with different shapes
    temp = np.random.randn(len(times), len(lats), len(lons))
    precip = np.random.gamma(2, 0.5, size=(len(times), len(lats), len(lons)))
    print(f"Random precip: {precip[0][0][0]}")

    # Create the dataset
    ds = xr.Dataset(
        {
            "precip": (
                ["time", "latitude", "longitude"],
                precip,
                {"units": "mm/day", "long_name": "Daily Precipitation"},
            ),
        },
        coords={
            "time": times,
            "latitude": ("latitude", lats, {"units": "degrees_north"}),
            "longitude": ("longitude", lons, {"units": "degrees_east"}),
        },
        attrs={"description": "Test dataset with random weather data"},
    )

    # Generate Random Key
    encryption_key = get_random_bytes(32).hex()
    print(f"Encryption Key: {encryption_key}")
    # Set the encryption key for the class
    EncryptionCodec.set_encryption_key(encryption_key)
    # Register the codec
    register_codec(EncryptionCodec(header="dClimate-Zarr"))

    # Apply the encryption codec to the dataset with a selected header
    encoding = {
        "precip": {
            "compressor": 
                EncryptionCodec(header="dClimate-Zarr")
            ,  # Add the Delta filter
        }
    }
    print(f"Encoding: {encoding}")
    # Write the dataset to the zarr store with the encoding on the temp
    ds.to_zarr(zarr_path, mode="w", encoding=encoding)

    yield zarr_path, ds

    # Cleanup
    shutil.rmtree(temp_dir)


def test_bad_encryption_keys():
    # Assert failure Encryption key must be set before using EncryptionCodec
    with pytest.raises(ValueError):
        EncryptionCodec(header="dClimate-Zarr")
    # Assert failure Encryption key must be a string
    with pytest.raises(ValueError):
        EncryptionCodec.set_encryption_key(123)
    # Assert failure Encryption key must be a hexadecimal string
    with pytest.raises(ValueError):
        EncryptionCodec.set_encryption_key("123Z")
    # Assert failure Encryption key must be 32 bytes (64 hex characters)
    with pytest.raises(ValueError):
        EncryptionCodec.set_encryption_key("1234567890")


def test_upload_then_read(random_zarr_dataset: tuple[str, xr.Dataset]):
    zarr_path, expected_ds = random_zarr_dataset

    # Open the zarr store
    test_ds = xr.open_zarr(zarr_path)

    # Check if encryption applied to temp but not to precip
    assert test_ds["precip"].encoding["compressor"] is not None

    # Prepare Writing to IPFS
    hamt1 = HAMT(
        store=IPFSStore(pin_on_add=False),
    )

    # Reusing the same encryption key as its still stored in the class in numcodecs
    test_ds.to_zarr(
        store=hamt1,
        mode="w",
    )

    hamt1_root: CID = hamt1.root_node_id  # type: ignore
    print(f"IPFS CID: {hamt1_root}")

    # Read the dataset from IPFS
    hamt1_read = HAMT(
        store=IPFSStore(),
        root_node_id=hamt1_root,
        read_only=True,
    )

    # Open the zarr store thats encrypted on IPFS
    loaded_ds1 = xr.open_zarr(store=hamt1_read)

    # Assert the values are the same
    assert np.array_equal(loaded_ds1["precip"].values, expected_ds["precip"].values), (
        "Precip values in loaded_ds1 and expected_ds are not identical!"
    )

    # Create new encryption filter but with a different encryption key
    encryption_key = get_random_bytes(32).hex()
    EncryptionCodec.set_encryption_key(encryption_key)
    register_codec(EncryptionCodec(header="dClimate-Zarr"))

    loaded_failure = xr.open_zarr(store=hamt1_read)
    # Accessing data should raise an exception since we don't have the correct encryption key
    with pytest.raises(Exception):
        _ = loaded_failure["precip"].values

    assert "precip" in loaded_ds1
    assert loaded_ds1.precip.attrs["units"] == "mm/day"

    assert loaded_ds1.precip.shape == expected_ds.precip.shape

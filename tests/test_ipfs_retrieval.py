import datetime
import json
import pathlib

import src.ipfs_retrieval as ipfs_retrieval
import pytest
import xarray as xr
import zarr
from src.dclimate_zarr_errors import NoMetadataFoundError

IPNS_NAME_HASH = "k2k4r8niyotlqqqvqoh7jr4gp6zp0b0975k88zmak151chv87w2p11qz"


def patched_get_single_metadata(ipfs_hash):
    with open(pathlib.Path(__file__).parent / "etc" / "stac_metadata" / f"{ipfs_hash}.json") as f:
        return json.load(f)


def patched_resolve_ipns_name_hash(ipns_name_hash):
    return "bafyreibtdfcfyyineq7pv2xunl4sxq6w6ziibswflmelaiydgbqwjk2sku"


def patched_get_dataset_by_ipfs_hash(ipfs_hash):
    with zarr.ZipStore(
        pathlib.Path(__file__).parent / "etc" / "sample_zarrs" / f"{ipfs_hash}.zip",
        mode="r",
    ) as in_zarr:
        return xr.open_zarr(in_zarr).compute()


@pytest.fixture(scope="module", autouse=True)
def default_session_fixture(module_mocker):
    """
    Patch metadata and Zarr retrieval functions in this test
    """
    module_mocker.patch(
        "src.ipfs_retrieval._get_single_metadata",
        patched_get_single_metadata,
    )
    module_mocker.patch(
        "src.ipfs_retrieval._resolve_ipns_name_hash",
        patched_resolve_ipns_name_hash,
    )
    module_mocker.patch(
        "src.ipfs_retrieval.get_dataset_by_ipfs_hash",
        patched_get_dataset_by_ipfs_hash,
    )


def test_get_dataset_by_ipns_hash_no_as_of():
    """
    Test that `get_dataset_by_ipns_hash` can select without a specified as_of time
    """
    ds = ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH)
    assert ds.attrs["order_created"] == 4


def test_get_dataset_by_ipns_hash_with_as_of():
    """
    Test that `get_dataset_by_ipns_hash` can select by specified as_of times
    """
    creation_times = [
        datetime.datetime(2022, 7, 26, 19, 17, 55),
        datetime.datetime(2022, 7, 26, 19, 19, 41),
        datetime.datetime(2022, 7, 26, 19, 20, 45),
        datetime.datetime(2022, 7, 26, 19, 22, 46),
    ]
    for i, time in enumerate(creation_times):
        ds = ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH, as_of=time)
        assert ds.attrs["order_created"] == i + 1


def test_get_dataset_by_ipns_hash_with_bad_as_of():
    """
    Test that `get_dataset_by_ipns_hash` fails when provided an invalid `as_of` parameter
    """
    creation_time = datetime.datetime(2022, 7, 26, 19, 17, 53)
    with pytest.raises(NoMetadataFoundError):
        ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH, as_of=creation_time)

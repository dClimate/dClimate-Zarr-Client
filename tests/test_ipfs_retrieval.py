import datetime
import json
import pathlib

import pytest
import requests
from unittest.mock import patch, mock_open
from src.ipfs_retrieval import get_ipns_name_hash, DatasetNotFoundError

import src.ipfs_retrieval as ipfs_retrieval

# import xarray as xr
# import zarr
from src.client import geo_temporal_query

IPNS_NAME_HASH = "k2k4r8niyotlqqqvqoh7jr4gp6zp0b0975k88zmak151chv87w2p11qz"


def patched_get_single_metadata(ipfs_hash):
    with open(
        pathlib.Path(__file__).parent / "etc" / "stac_metadata" / f"{ipfs_hash}.json"
    ) as f:
        return json.load(f)


def patched_resolve_ipns_name_hash(ipns_name_hash):
    return "bafyreibtdfcfyyineq7pv2xunl4sxq6w6ziibswflmelaiydgbqwjk2sku"


# def patched_get_dataset_by_ipfs_hash(ipfs_hash):
#     with zarr.ZipStore(
#         pathlib.Path(__file__).parent / "etc" / "sample_zarrs" / f"{ipfs_hash}.zip",
#         mode="r",
#     ) as in_zarr:
#         return xr.open_zarr(in_zarr).compute()


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
    # module_mocker.patch(
    #     "src.ipfs_retrieval.get_dataset_by_ipfs_hash",
    #     patched_get_dataset_by_ipfs_hash,
    # )


def test_get_ipns_name_hash():
    """
    Test that `get_ipns_name_hash`  returns json of dataset names and hashes
    """
    ipns_name_hash = ipfs_retrieval.get_ipns_name_hash("cpc-precip-conus")
    # Assert that it starts with ba
    assert ipns_name_hash.startswith("ba")


# Success from Local Cache
def test_get_ipns_name_hash_fallback_success():
    """
    Test that when the remote CID endpoint is unreachable OR the key isn't found,
    we successfully fall back to the local cids_cache.json.
    """
    # We'll simulate "requests.get" raising a RequestException,
    # so we go straight to the fallback code.
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException(
            "Simulated RequestException"
        )

        # Next, we mock "os.path.exists" to return True, as if cids_cache.json does exist
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = True

            # Provide a cids_cache.json that DOES have the desired entry
            fake_json = '{"cpc-precip-conus":"bafkreihashfromlocalfile"}'
            with patch("builtins.open", mock_open(read_data=fake_json)):
                ipns_name_hash = get_ipns_name_hash("cpc-precip-conus")
                assert ipns_name_hash == "bafkreihashfromlocalfile"


# Failure When Local Cache Does Not Exist
def test_get_ipns_name_hash_fallback_no_file():
    """
    Test that if requests.get fails and the local fallback file does NOT exist,
    we raise DatasetNotFoundError.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException(
            "Simulated RequestException"
        )

        # cids_cache.json does NOT exist
        with patch("os.path.exists") as mock_exists:
            mock_exists.return_value = False

            with pytest.raises(DatasetNotFoundError):
                get_ipns_name_hash("some-nonexistent-key")


# Failure When Local Cache Exists But Missing Key
def test_get_ipns_name_hash_fallback_key_not_found_in_file():
    """
    Even if cids_cache.json exists, if the key is not found,
    raise DatasetNotFoundError.
    """
    with patch("requests.get") as mock_requests_get:
        mock_requests_get.side_effect = requests.RequestException(
            "Simulated RequestException"
        )

        with patch("os.path.exists", return_value=True):
            fake_json = '{"some-other-key":"bafkreihashfromlocalfile"}'
            with patch("builtins.open", mock_open(read_data=fake_json)):
                with pytest.raises(DatasetNotFoundError) as exc:
                    get_ipns_name_hash("cpc-precip-conus")

                assert "Invalid dataset name" in str(exc.value)


def test_list_datasets():
    """
    Test that `list_datasets` returns a list of dataset keys
    """
    datasets = ipfs_retrieval.list_datasets()
    assert len(datasets) == 3
    assert "cpc-precip-conus" in datasets


def test_geo_temporal_query():
    ds_bytes = geo_temporal_query(
        "chirps-final-p05",
        point_kwargs={"latitude": 40.726446, "longitude": -95.937581},
        time_range=[datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 1)],
    )
    assert ds_bytes["data"][0] == 2.8954310417175293


# def test_get_dataset_by_ipns_hash_no_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` can select without a specified as_of time
#     """
#     ds = ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH)
#     assert ds.attrs["order_created"] == 4


# def test_get_dataset_by_ipns_hash_with_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` can select by specified as_of times
#     """
#     creation_times = [
#         datetime.datetime(2022, 7, 26, 19, 17, 55),
#         datetime.datetime(2022, 7, 26, 19, 19, 41),
#         datetime.datetime(2022, 7, 26, 19, 20, 45),
#         datetime.datetime(2022, 7, 26, 19, 22, 46),
#     ]
#     for i, time in enumerate(creation_times):
#         ds = ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH, as_of=time)
#         assert ds.attrs["order_created"] == i + 1


# def test_get_dataset_by_ipns_hash_with_bad_as_of():
#     """
#     Test that `get_dataset_by_ipns_hash` fails when provided an invalid `as_of` parameter
#     """
#     creation_time = datetime.datetime(2022, 7, 26, 19, 17, 53)
#     with pytest.raises(NoMetadataFoundError):
#         ipfs_retrieval.get_dataset_by_ipns_hash(IPNS_NAME_HASH, as_of=creation_time)

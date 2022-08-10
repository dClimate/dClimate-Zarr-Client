import datetime
import typing

import requests
import xarray as xr
from ipldstore import get_ipfs_mapper

from .dclimate_zarr_errors import DatasetNotFoundError, NoMetadataFoundError

DEFAULT_HOST = "http://127.0.0.1:5001/api/v0"


def _get_single_metadata(ipfs_hash: str) -> dict:
    """Get metadata for given ipfs hash over ipld

    Args:
        ipfs_hash (str): ipfs hash for which to get metadata

    Returns:
        dict: dict of metadata for hash
    """
    r = requests.post(f"{DEFAULT_HOST}/dag/get", params={"arg": ipfs_hash})
    r.raise_for_status()
    return r.json()


def _get_previous_hash_from_metadata(metadata: dict) -> typing.Optional[str]:
    """Pull in last updated hash from STAC metadata

    Args:
        metadata (dict): STAC metadata

    Returns:
        str: Previous hash, or None if given root metadata
    """
    links = metadata["links"]
    try:
        link_to_previous = [link for link in links if link["rel"] == "previous"][0]
    except IndexError:
        return None
    return link_to_previous["metadata href"]["/"]


def _resolve_ipns_name_hash(ipns_name_hash: str) -> str:
    """Find the latest IPFS hash corresponding to a stable ipns name hash

    Args:
        ipfs_name_hash (str): stable IPNS name hash

    Returns:
        str: ipfs hash corresponding to this ipns name hash
    """
    r = requests.post(f"{DEFAULT_HOST}/name/resolve", params={"arg": ipns_name_hash})
    r.raise_for_status()
    return r.json()["Path"].split("/")[-1]


def get_ipns_name_hash(ipns_key_str) -> str:
    """Find the latest IPNS name hash corresponding to a string (key)

    Args:
        ipfs_key_str (str): a string (key) identifying a dataset

    Raises:
        KeyError: raised if no IPNS key string is found in the IPNS keys list

    Returns:
        str: ipfsname hash corresponding to the provided string
    """
    r = requests.post(f"{DEFAULT_HOST}/key/list", params={"decoder": "json"})
    ipns_rec_dict, ipns_records = {}, []
    for name_hash_pair in r.json()["Keys"]:
        ipns_records.append(tuple([vals for vals in name_hash_pair.values()]))
    ipns_rec_dict.update(ipns_records)
    try:
        return ipns_rec_dict[ipns_key_str]
    except KeyError:
        raise DatasetNotFoundError


def _get_relevant_metadata(ipfs_head_hash: str, as_of: datetime.datetime) -> dict:
    """Iterates through STAC metadata until metadata generated before as_of is found

    Args:
        ipfs_head_hash (str): first hash in chain
        as_of (datetime.datetime): cutoff date for finding metadata

    Raises:
        NoMetadataFoundError: raised if no metadata older than cutoff date is found

    Returns:
        dict: relevant metadata
    """
    cur_metadata = _get_single_metadata(ipfs_head_hash)
    while True:
        time_generated = datetime.datetime.strptime(
            cur_metadata["properties"]["updated"], "%Y-%m-%dT%H:%M:%SZ"
        )
        if time_generated <= as_of:
            return cur_metadata
        prev_hash = _get_previous_hash_from_metadata(cur_metadata)
        if prev_hash is None:
            raise NoMetadataFoundError(f"No metadata found after as_of: {as_of}")
        cur_metadata = _get_single_metadata(prev_hash)


def get_dataset_by_ipfs_hash(ipfs_hash: str) -> xr.Dataset:
    """Gets xarray dataset using changing ipfs hash

    Args:
        ipfs_hash (str): ipfs hash that changes between updates

    Returns:
        xr.Dataset: dataset corresponding to hash
    """
    ipfs_mapper = get_ipfs_mapper()
    ipfs_mapper.set_root(ipfs_hash)
    return xr.open_zarr(ipfs_mapper)


def get_dataset_by_ipns_hash(
    ipns_name_hash: str, as_of: typing.Optional[datetime.datetime] = None
) -> xr.Dataset:
    """Gets xarray dataset using fixed ipns name hash

    Args:
        ipns_name_hash (str): ipns hash that will remain fixed between updates
        as_of (typing.Optional[datetime.datetime], optional): cutoff date for finding metadata. Defaults to None.
            if None, function will return most recent dataset

    Returns:
        xr.Dataset: dataset corresponding to hash and as_of date
    """
    ipfs_head_hash = _resolve_ipns_name_hash(ipns_name_hash)
    if as_of:
        metadata = _get_relevant_metadata(ipfs_head_hash, as_of=as_of)
    else:
        metadata = _get_single_metadata(ipfs_head_hash)
    return get_dataset_by_ipfs_hash(metadata["assets"]["analytic"]["href"]["/"])

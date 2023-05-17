from dclimate_zarr_client.dclimate_zarr_errors import BucketNotFoundError, PathNotFoundError, \
    ZarrClientError
from dclimate_zarr_client.s3_retrieval import get_s3_fs
import typing
import os
import json
from ast import literal_eval


def get_collections(bucket_name: str) -> typing.List[str]:
    s3 = get_s3_fs()
    _validate_bucket_name(bucket_name)
    collections = s3.ls(f'{bucket_name}/metadata/collections', detail=False)
    return [_extract_file_name_from_path(collection) for collection in collections]


def get_collection_metadata(bucket_name: str, collection_name: str):
    s3 = get_s3_fs()
    _validate_bucket_name(bucket_name)
    collection_metadata_path = f"{bucket_name}/metadata/collections/{collection_name}.json"
    _validate_path(collection_metadata_path)
    collection_metadata = s3.cat_file(collection_metadata_path)
    return json.loads(collection_metadata)


def get_collection_datasets(bucket_name: str, collection_name: str):
    s3 = get_s3_fs()
    _validate_bucket_name(bucket_name)
    collection_file_path = f'{bucket_name}/metadata/collections/{collection_name}.json'
    _validate_path(collection_file_path)
    collection_file_content = s3.cat_file(collection_file_path)
    try:
        collection_file_content_as_dict = literal_eval(collection_file_content.decode())
        links = collection_file_content_as_dict.get("links") or []
        valid_items = filter(lambda link: (link.get("rel") == "item" and link.get("href")), links)
        return [_extract_file_name_from_path(item.get("href")) for item in valid_items]

    except ValueError as e:
        raise ZarrClientError(f"There is an error reading the file: {collection_file_path}")


def get_dataset_metadata(bucket_name: str, dataset_name: str):
    s3 = get_s3_fs()
    _validate_bucket_name(bucket_name)
    dataset_metadata_file_path = f"{bucket_name}/metadata/datasets/{dataset_name}.json"
    _validate_path(dataset_metadata_file_path)
    dataset_metadata = s3.cat_file(dataset_metadata_file_path)
    return json.loads(dataset_metadata)


def _validate_bucket_name(bucket_name: str):
    try:
        _validate_path(bucket_name)
    except PathNotFoundError:
        raise BucketNotFoundError(f"Bucket {bucket_name} does not exist")


def _validate_path(path: str):
    s3 = get_s3_fs()
    exists = s3.exists(f'{path}')
    if not exists:
        raise PathNotFoundError(f"Path {path} does not exist")


def _extract_file_name_from_path(path: str):
    return os.path.splitext(os.path.basename(path))[0]

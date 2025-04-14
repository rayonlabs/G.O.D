import os
import random
import shutil
import tempfile
import uuid
import zipfile
from math import ceil
from pathlib import Path
from typing import List

from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from fiber import Keypair

import validator.core.constants as cst
from core.models.payload_models import DpoDatasetColumnsResponse
from core.models.payload_models import ImageTextPair
from core.models.utility_models import FileFormat
from core.models.utility_models import TaskType
from core.utils import download_s3_file
from validator.augmentation.augmentation import generate_augmented_text_dataset
from validator.core.models import DpoRawTask
from validator.core.models import InstructTextRawTask
from validator.evaluation.utils import get_default_dataset_config
from validator.utils.cache_clear import delete_dataset_from_cache
from validator.utils.logging import get_logger
from validator.utils.util import save_json_to_temp_file
from validator.utils.util import upload_file_to_minio


logger = get_logger(__name__)


def create_zip_for_image_dataset(split_keys: set, zip_name: str, entries: dict, dataset_root: Path) -> Path:
    subfolder_name = Path(zip_name).stem
    zip_path = dataset_root / zip_name

    if zip_path.exists():
        logger.error(f"Zip path {zip_path} exists. This should not happen. Deleting it.")
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for key in split_keys:
            img_file, txt_file = entries[key]
            with open(txt_file, "r") as f:
                logger.info(f"Adding the following prompt to the zip: {f.read()}")
            zipf.write(img_file, Path(subfolder_name) / img_file.relative_to(dataset_root))
            zipf.write(txt_file, Path(subfolder_name) / txt_file.relative_to(dataset_root))

    return zip_path


def unzip_to_temp_path(zip_file_path: str) -> str:
    random_tmp_id = uuid.uuid4()
    tmp_dir = f"{cst.TEMP_PATH_FOR_IMAGES}/{random_tmp_id}"
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    return tmp_dir


async def load_dataset_from_s3(dataset_url: str, max_file_size_bytes: int = None) -> Dataset | DatasetDict:
    """Load a dataset from S3 storage."""
    try:
        # TODO: check what is going on here
        with tempfile.TemporaryDirectory() as temp_dir:
            local_file_path = await download_s3_file(dataset_url)
            if max_file_size_bytes:
                file_size = os.path.getsize(local_file_path)
                if file_size > max_file_size_bytes:
                    raise ValueError(f"File size {file_size} exceeds max file size {max_file_size_bytes}")
            filename = os.path.basename(local_file_path)
            new_path = os.path.join(temp_dir, filename)

            os.rename(local_file_path, new_path)
            dataset = load_dataset("json", data_files=new_path, split="train", trust_remote_code=False)

            return dataset
    except Exception as e:
        logger.exception(f"Failed to load dataset from S3: {e}")
        raise e


async def train_test_split(dataset: Dataset, test_size: float = cst.TRAIN_TEST_SPLIT_PERCENTAGE) -> DatasetDict:
    logger.info(f"Splitting dataset into train and test with test size {test_size}")

    test_size = min(
        int(len(dataset) * test_size),
        cst.MAX_TEST_DATA_POINTS,
    )
    split_dataset = dataset.train_test_split(test_size=test_size, shuffle=True, seed=42)
    logger.info(f"Train set size: {len(split_dataset['train'])}")
    logger.info(f"Test set size: {len(split_dataset['test'])}")

    return split_dataset


def train_test_split_image(dataset_path: str) -> tuple[str, str]:
    """
    Dataset path is a folder containing the images and text files.
    """
    dataset_path = Path(dataset_path)

    has_images = any(dataset_path.glob(f"*.{ext.lstrip('.')}") for ext in cst.SUPPORTED_IMAGE_FILE_EXTENSIONS)

    if not has_images:
        sub_folder = [
            folder
            for folder in dataset_path.iterdir()
            if folder.is_dir() and any(folder.glob(f"*.{ext.lstrip('.')}") for ext in cst.SUPPORTED_IMAGE_FILE_EXTENSIONS)
        ]
        if not sub_folder:
            raise ValueError(f"No folder containing images found in: {dataset_path}")
        dataset_path = sub_folder[0]

    dataset_entries = {}
    for file in dataset_path.iterdir():
        if file.suffix in cst.SUPPORTED_IMAGE_FILE_EXTENSIONS:
            txt_file = file.with_suffix(".txt")
            if txt_file.exists():
                dataset_entries[file.stem] = (file, txt_file)

    keys = list(dataset_entries.keys())
    random.shuffle(keys)
    split_idx = ceil(len(keys) * cst.TRAIN_TEST_SPLIT_PERCENTAGE)
    test_keys = set(keys[:split_idx])
    train_keys = set(keys[split_idx:])

    test_zip_path = create_zip_for_image_dataset(
        split_keys=test_keys, zip_name=cst.IMAGE_TEST_SPLIT_ZIP_NAME, entries=dataset_entries, dataset_root=dataset_path
    )
    train_zip_path = create_zip_for_image_dataset(
        split_keys=train_keys, zip_name=cst.IMAGE_TRAIN_SPLIT_ZIP_NAME, entries=dataset_entries, dataset_root=dataset_path
    )

    return test_zip_path, train_zip_path


async def get_additional_synth_data(
    dataset: Dataset, columns_to_sample: List[str], keypair: Keypair, is_dpo: bool = False
) -> List[dict] | List[DpoDatasetColumnsResponse]:
    num_samples = min(
        cst.MAX_SYNTH_DATA_POINTS,
        int(len(dataset) * cst.ADDITIONAL_SYNTH_DATA_PERCENTAGE),
    )
    logger.info(f"Generating {num_samples} additional synthetic data points")
    sampled_data = dataset.shuffle(seed=42).select(range(num_samples))

    sampled_data = sampled_data.remove_columns([col for col in sampled_data.column_names if col not in columns_to_sample])
    # NOTE: Need to do something if errors, without trying to then generate synthetic data
    try:
        sampled_data_list = list(sampled_data)
    except Exception as e:
        logger.info(f"There is an issue with this sample data for some reason. dataset: {sampled_data}; error: {e}")
        return None

    synthetic_data = await generate_augmented_text_dataset(sampled_data_list, keypair=keypair, is_dpo=is_dpo)

    return synthetic_data


async def download_and_load_dataset(
    dataset_name: str, file_format: FileFormat, max_file_size_bytes: int = cst.MAX_FILE_SIZE_BYTES
) -> Dataset:
    if file_format == FileFormat.S3:
        dataset = await load_dataset_from_s3(dataset_name, max_file_size_bytes)
    else:
        config_name = get_default_dataset_config(dataset_name)
        dataset = load_dataset(dataset_name, config_name, trust_remote_code=True)

    if isinstance(dataset, DatasetDict):
        combined_dataset = concatenate_datasets([split for split in dataset.values()])
    else:
        combined_dataset = dataset

    return combined_dataset


def change_to_json_format(dataset: Dataset, columns: List[str]):
    try:
        result = []
        for row in dataset:
            row_dict = {}
            for col in columns:
                if col in row:
                    value = row[col]
                    row_dict[col] = str(value) if value is not None else ""
            result.append(row_dict)
        return result
    except Exception as e:
        logger.error(f"Error converting to JSON format: {str(e)}")
        return []


def assign_some_of_the_train_to_synth(train_dataset: Dataset):
    if not isinstance(train_dataset, Dataset):
        raise TypeError("train_dataset must be an instance of datasets.Dataset")
    if len(train_dataset) == 0:
        raise ValueError("Cannot split an empty dataset")
    try:
        num_synthetic_samples = min(cst.MAX_SYNTH_DATA_POINTS, int(len(train_dataset) * cst.ADDITIONAL_SYNTH_DATA_PERCENTAGE))
        dataset_length = len(train_dataset)
        split_index = dataset_length - num_synthetic_samples
        synthetic_dataset = train_dataset.select(range(split_index, dataset_length))
        remaining_train_dataset = train_dataset.select(range(split_index))
    except Exception as e:
        logger.info(f"There was an issue with the split {e} ")

    logger.info(
        f"Taking {num_synthetic_samples} samples from the train set to be synthetic data. "
        f"Original size: {dataset_length}, "
        f"Training size: {len(remaining_train_dataset)}, "
        f"Synthetic size: {len(synthetic_dataset)}"
    )

    return remaining_train_dataset, synthetic_dataset


async def _process_and_upload_datasets(
    train_dataset, test_dataset, synthetic_data, columns_to_sample, should_reupload_train, should_reupload_test, ds_hf_name=None
):
    files_to_delete = []
    try:
        if should_reupload_train:
            train_data_json = change_to_json_format(train_dataset, columns_to_sample)
            train_json_path, train_json_size = await save_json_to_temp_file(train_data_json, prefix="train_data_")
            files_to_delete.append(train_json_path)
            await _check_file_size(train_json_size, "train_data")
            train_json_url = await upload_file_to_minio(
                train_json_path, cst.BUCKET_NAME, f"{os.urandom(8).hex()}_train_data.json"
            )
        else:
            train_json_url = train_dataset
        if should_reupload_test:
            test_data_json = change_to_json_format(test_dataset, columns_to_sample)
            test_json_path, test_json_size = await save_json_to_temp_file(test_data_json, prefix="test_data_")
            files_to_delete.append(test_json_path)
            await _check_file_size(test_json_size, "test_data")
            test_json_url = await upload_file_to_minio(test_json_path, cst.BUCKET_NAME, f"{os.urandom(8).hex()}_test_data.json")
        else:
            test_json_url = test_dataset
        if synthetic_data:
            synthetic_data_json = change_to_json_format(synthetic_data, columns_to_sample)
            synth_json_path, synth_json_size = await save_json_to_temp_file(synthetic_data_json, prefix="synth_data_")
            files_to_delete.append(synth_json_path)
            await _check_file_size(synth_json_size, "synth_data")
            synth_json_url = await upload_file_to_minio(
                synth_json_path, cst.BUCKET_NAME, f"{os.urandom(8).hex()}_synth_data.json"
            )
        else:
            synth_json_url = None
    except Exception as e:
        logger.error(f"There was a problem going to json {e}")
        raise e

    logger.info(f"Train json url: {train_json_url}\nTest json url: {test_json_url}\nSynth json url: {synth_json_url}")

    if not train_json_url:
        raise Exception("Failed to upload training data to MinIO storage")
    if not test_json_url:
        raise Exception("Failed to upload test data to MinIO storage")
    if not synth_json_url and synthetic_data:
        raise Exception("Failed to upload synthetic data to MinIO storage")

    for file_path in files_to_delete:
        if os.path.exists(file_path):
            os.remove(file_path)

    if ds_hf_name:
        delete_dataset_from_cache(ds_hf_name)

    return (
        test_json_url.strip('"'),
        synth_json_url.strip('"') if synth_json_url else None,
        train_json_url.strip('"'),
    )


def pick_columns_to_sample(task: InstructTextRawTask | DpoRawTask) -> list[str]:
    if task.task_type == TaskType.INSTRUCTTEXTTASK:
        columns_to_sample = [
            i for i in [task.field_system, task.field_instruction, task.field_input, task.field_output] if i is not None
        ]
    elif task.task_type == TaskType.DPOTASK:
        columns_to_sample = [task.field_system, task.field_prompt, task.field_chosen, task.field_rejected]
    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")
    return columns_to_sample


def validate_and_transform_dpo(data, task: DpoRawTask):
    assert isinstance(data, DpoDatasetColumnsResponse)
    return {task.field_prompt: data.field_prompt, task.field_chosen: data.field_chosen, task.field_rejected: data.field_rejected}


async def prepare_text_task(task: InstructTextRawTask | DpoRawTask, keypair: Keypair) -> tuple[str, str, str]:
    should_reupload_train = FileFormat.S3 == task.file_format
    should_reupload_test = task.test_data is None or task.file_format != FileFormat.S3

    train_dataset_name = task.training_data if task.training_data else task.ds

    if not task.test_data:
        logger.info(f"Preparing {train_dataset_name}")
        try:
            dataset = await download_and_load_dataset(train_dataset_name, task.file_format)
        except Exception as e:
            logger.info(f"There was an issue loading the dataset: {e}")
            raise e
        dataset_dict = await train_test_split(dataset)

        train_ds = dataset_dict["train"]
        test_ds = dataset_dict["test"]
        should_reupload_train = True
        should_reupload_test = True
    else:
        logger.info(f"Preparing train and test datasets. Train: {task.training_data}, Test: {task.test_data}")
        try:
            train_ds = await download_and_load_dataset(task.training_data, task.file_format)
            test_ds = await download_and_load_dataset(task.test_data, task.file_format)
        except Exception as e:
            logger.info(f"There was an issue loading the dataset: {e}")
            raise e

    total_size = len(train_ds) + len(test_ds)
    check_ds_num_rows(total_size)

    if isinstance(task, InstructTextRawTask):
        columns_to_sample = pick_columns_to_sample(task)
    else:
        columns_to_sample = [task.field_prompt, task.field_chosen, task.field_rejected]

    if any(col not in train_ds.column_names for col in columns_to_sample):
        raise ValueError(f"Column {columns_to_sample} not found in train dataset")

    synthetic_ds = []
    try:
        if cst.GET_SYNTH_DATA:
            logger.info("Generating additional synthetic data")
            if isinstance(task, DpoRawTask):
                # we only need the field prompt to generate new data here
                synthetic_ds = await get_additional_synth_data(test_ds, [task.field_prompt], keypair, is_dpo=True)
                synthetic_ds = [validate_and_transform_dpo(data, task) for data in synthetic_ds]

            else:
                synthetic_ds = await get_additional_synth_data(test_ds, columns_to_sample, keypair, is_dpo=False)
        else:
            logger.info("Skipping synthetic data generation")
    except Exception as e:
        # if for some reason the api is down, we move some of the train over to be synth

        logger.info(f"Synthetic dataset gen is down, moving part of the train over: {e}")
        train_ds, synthetic_ds = assign_some_of_the_train_to_synth(train_ds)

    if synthetic_ds is None:
        logger.info("There was not enough synthetic data created we are instead grabbing from train ")
        train_ds, synthetic_ds = assign_some_of_the_train_to_synth(train_ds)

    return await _process_and_upload_datasets(
        train_ds,
        test_ds,
        synthetic_ds,
        columns_to_sample,
        should_reupload_train,
        should_reupload_test,
        train_dataset_name if task.file_format == FileFormat.HF else None,
    )


async def prepare_image_task(image_text_pairs: list[ImageTextPair]) -> tuple[str, str]:
    Path(cst.TEMP_PATH_FOR_IMAGES).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cst.TEMP_PATH_FOR_IMAGES) as source_dir:
        for i, pair in enumerate(image_text_pairs):
            txt_path = Path(source_dir) / f"{i}.txt"
            await download_s3_file(pair.text_url, str(txt_path))

            tmp_img_path = Path(await download_s3_file(pair.image_url))
            img_extension = tmp_img_path.suffix
            img_path = Path(source_dir) / f"{i}{img_extension}"
            shutil.move(tmp_img_path, img_path)

        test_zip_path, train_zip_path = train_test_split_image(dataset_path=source_dir)

        test_url = await upload_file_to_minio(
            file_path=str(test_zip_path), bucket_name=cst.BUCKET_NAME, object_name=f"{os.urandom(8).hex()}_test_data.zip"
        )
        train_url = await upload_file_to_minio(
            file_path=str(train_zip_path), bucket_name=cst.BUCKET_NAME, object_name=f"{os.urandom(8).hex()}_train_data.zip"
        )

        return (test_url.strip('"'), train_url.strip('"'))


async def _check_file_size(file_size: int, file_type: str) -> None:
    if file_size > cst.MAX_FILE_SIZE_BYTES:
        raise ValueError(
            f"{file_type} data size ({file_size} bytes) exceeds maximum allowed size of {cst.MAX_FILE_SIZE_BYTES} bytes"
        )


def check_ds_num_rows(num_rows: int) -> int:
    if num_rows < cst.MINIMUM_DATASET_ROWS:
        error_msg = f"Dataset has only {num_rows} rows, minimum required is {cst.MINIMUM_DATASET_ROWS}"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return num_rows

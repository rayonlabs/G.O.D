import ast
import asyncio
import json
import os
import random
import shutil
import tempfile
import uuid
import zipfile
from math import ceil
from pathlib import Path

from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset
from fiber import Keypair

import validator.core.constants as cst
from core.models.payload_models import DpoDatasetColumnsResponse
from core.models.payload_models import ImageTextPair
from core.models.utility_models import FileFormat
from core.utils import download_s3_file
from validator.augmentation.augmentation import generate_augmented_text_dataset
from validator.augmentation.augmentation import generate_dpo_reformulation
from validator.augmentation.augmentation import load_prompts
from validator.core.models import AnyTextTypeRawTask
from validator.core.models import DpoRawTask
from validator.core.models import GrpoRawTask
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


def adapt_synthetic_columns(
    synthetic_data: list[dict] | list[DpoDatasetColumnsResponse], task: AnyTextTypeRawTask
    ) -> list[dict]:
    """
    Transform synthetic data based on task type.

    Args:
        synthetic_data: List of synthetic data points
        task: The task instance that determines how to transform the data

    Returns:
        Transformed synthetic data
    """
    if isinstance(task, InstructTextRawTask):
        return synthetic_data
    elif isinstance(task, DpoRawTask):
        return [validate_and_transform_dpo(data, task) for data in synthetic_data]
    elif isinstance(task, GrpoRawTask):
        return synthetic_data
    else:
        raise ValueError(f"Unsupported task type: {type(task).__name__}")


async def get_additional_synth_data(
    dataset: Dataset, columns_to_sample: list[str], keypair: Keypair, task: AnyTextTypeRawTask
) -> list[dict]:
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

    synthetic_data = await generate_augmented_text_dataset(sampled_data_list, keypair=keypair, task_type=task.task_type)
    synthetic_data = adapt_synthetic_columns(synthetic_data, task)
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


def change_to_json_format(dataset: Dataset, columns: list[str]):
    result = []
    total_rows = 0
    fully_empty_rows = 0

    for row in dataset:
        row_dict = {}
        is_row_empty = True
        for col in columns:
            if col in row:
                value = row[col]
                str_value = str(value) if value is not None else ""
                row_dict[col] = str_value
                if str_value != "":
                    is_row_empty = False
        result.append(row_dict)
        total_rows += 1
        if is_row_empty:
            fully_empty_rows += 1

    if total_rows > 0 and (fully_empty_rows / total_rows) > 0.8:
        raise ValueError(f"More than 80% of rows are fully empty ({fully_empty_rows}/{total_rows} rows)")

    return result


def assign_some_of_the_train_to_synth(train_dataset: Dataset, is_dpo: bool = False):
    if not isinstance(train_dataset, Dataset):
        raise TypeError("train_dataset must be an instance of datasets.Dataset")
    if len(train_dataset) == 0:
        raise ValueError("Cannot split an empty dataset")

    try:
        dataset_length = len(train_dataset)

        if is_dpo:
            # For both DPO and instruct text tasks, use SYNTH_EXAMPLES_FROM_TRAIN constant
            num_synthetic_samples = cst.SYNTH_EXAMPLES_FROM_TRAIN
            synthetic_dataset = train_dataset.shuffle(seed=42).select(range(num_synthetic_samples))
            remaining_train_dataset = train_dataset

            logger.info(
                f"Using max(synth,test) approach: Sampling {num_synthetic_samples} examples WITH REPLACEMENT from train set. "
                f"Original train size: {dataset_length}, "
                f"Training size (unchanged): {len(remaining_train_dataset)}, "
                f"Synthetic size: {len(synthetic_dataset)}"
            )
        else:
            num_synthetic_samples = min(
                cst.MAX_SYNTH_DATA_POINTS,
                int(len(train_dataset) * cst.ADDITIONAL_SYNTH_DATA_PERCENTAGE),
            )
            split_index = dataset_length - num_synthetic_samples
            synthetic_dataset = train_dataset.select(range(split_index, dataset_length))
            remaining_train_dataset = train_dataset.select(range(split_index))

            logger.info(
                f"Taking {num_synthetic_samples} samples from the train set to be synthetic data. "
                f"Original size: {dataset_length}, "
                f"Training size: {len(remaining_train_dataset)}, "
                f"Synthetic size: {len(synthetic_dataset)}"
            )
    except Exception as e:
        logger.info(f"There was an issue with the split {e} ")
        raise e

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


def extract_grpo_extra_columns(task: GrpoRawTask) -> list[str]:
    """
    Extract all unique arguments from reward functions excluding field_prompt.
    """
    all_args = set()

    for reward_function in task.reward_functions:
        parsed = ast.parse(reward_function.reward_func)

        for node in ast.walk(parsed):
            if isinstance(node, ast.FunctionDef):
                all_args |= {arg.arg for arg in node.args.args}
                break

    return list(all_args - {task.field_prompt, "completions"})


def pick_columns_to_sample(task: AnyTextTypeRawTask) -> list[str]:
    if isinstance(task, InstructTextRawTask):
        columns_to_sample = [
            i for i in [task.field_system, task.field_instruction, task.field_input, task.field_output] if i is not None
        ]
    elif isinstance(task, DpoRawTask):
        columns_to_sample = [
            i for i in [task.field_system, task.field_prompt, task.field_chosen, task.field_rejected] if i is not None
            ]
    elif isinstance(task, GrpoRawTask):
        columns_to_sample = [task.field_prompt] + extract_grpo_extra_columns(task)
    else:
        raise ValueError(f"Unsupported task type: {task.task_type}")
    return columns_to_sample


def validate_and_transform_dpo(data: DpoDatasetColumnsResponse, task: DpoRawTask) -> dict:
    assert isinstance(data, DpoDatasetColumnsResponse)
    return {task.field_prompt: data.field_prompt, task.field_chosen: data.field_chosen, task.field_rejected: data.field_rejected}


async def generate_synthetic_dpo_data(dataset: Dataset, keypair: Keypair, task: DpoRawTask) -> list[dict]:
    prompt_field = task.field_prompt
    logger.info(f"Generating synthetic DPO data from the field {prompt_field}")

    num_samples = min(cst.SYNTHETIC_TOTAL_SIZE * 2, len(dataset))

    sampled_data = dataset.shuffle(seed=42).select(range(num_samples))
    prompts = sampled_data[prompt_field]

    prompts_for_gen = [{prompt_field: prompt} for prompt in prompts]

    prompts_obj = load_prompts()
    logger.info(f"Attempting to generate {cst.SYNTHETIC_TOTAL_SIZE} synthetic DPO samples")

    synthetic_dataset = []
    json_errors = 0
    generic_errors = 0
    consecutive_errors = 0
    max_consecutive_errors = 10
    max_retry_attempts = 3
    max_batch_retries = 2

    total_batches = (len(prompts_for_gen) + cst.SYNTH_GEN_BATCH_SIZE - 1) // cst.SYNTH_GEN_BATCH_SIZE
    batch_idx = 0

    while len(synthetic_dataset) < cst.SYNTHETIC_TOTAL_SIZE and batch_idx < len(prompts_for_gen):
        end_idx = min(batch_idx + cst.SYNTH_GEN_BATCH_SIZE, len(prompts_for_gen))
        batch = prompts_for_gen[batch_idx: end_idx]
        current_batch = (batch_idx // cst.SYNTH_GEN_BATCH_SIZE) + 1

        for retry in range(max_batch_retries):
            if retry > 0:
                logger.info(f"Retrying batch {current_batch}/{total_batches} (attempt {retry+1}/{max_batch_retries})")

            logger.info(f"Processing batch {current_batch}/{total_batches} ({len(batch)} samples)")

            batch_tasks = []
            for prompt in batch:
                async def process_with_retry(p):
                    for attempt in range(max_retry_attempts):
                        try:
                            result = await generate_dpo_reformulation(p, prompts_obj, keypair)
                            return result
                        except Exception as e:
                            if attempt < max_retry_attempts - 1:
                                logger.info(f"Retrying prompt after error: {e}")
                            else:
                                raise
                    return None

                batch_tasks.append(process_with_retry(prompt))

            results = await asyncio.gather(*batch_tasks, return_exceptions=True)

            batch_results = []
            batch_error_count = 0

            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    if isinstance(result, json.JSONDecodeError):
                        json_errors += 1
                    else:
                        generic_errors += 1
                    consecutive_errors += 1
                    batch_error_count += 1

                    if consecutive_errors >= max_consecutive_errors:
                        logger.error(f"Maximum consecutive errors reached when generating synthetic DPO dataset. Error: {result}")
                        consecutive_errors = 0
                else:
                    if idx == 0:
                        logger.info(f"Batch {current_batch} example - Input: {batch[idx]}")
                        logger.info(f"Batch {current_batch} example - Output: {result}")
                        logger.info(f"DPO Fields - prompt: '{result.field_prompt[:100]}...'")
                        logger.info(f"DPO Fields - chosen: '{result.field_chosen[:100]}...'")
                        logger.info(f"DPO Fields - rejected: '{result.field_rejected[:100]}...'")
                    consecutive_errors = 0

                    dpo_item = validate_and_transform_dpo(result, task)
                    batch_results.append(dpo_item)

            synthetic_dataset.extend(batch_results)

            if batch_results:
                logger.info(
                    f"Batch {current_batch}/{total_batches} complete. "
                    f"Generated {len(batch_results)}/{len(batch)} samples successfully"
                )

                if len(batch_results) >= len(batch) * 0.7 or batch_error_count < 3:
                    break

        batch_idx = end_idx
        logger.info(f"Progress: {len(synthetic_dataset)}/{cst.SYNTHETIC_TOTAL_SIZE} synthetic samples generated")

    logger.info(
        f"Finished generating synthetic data. Got {len(synthetic_dataset)} samples total. "
        f"JSON errors: {json_errors}, Other errors: {generic_errors}"
    )

    if len(synthetic_dataset) < cst.SYNTHETIC_TOTAL_SIZE:
        missing_samples = cst.SYNTHETIC_TOTAL_SIZE - len(synthetic_dataset)
        logger.warning(
            f"Could only generate {len(synthetic_dataset)}/{cst.SYNTHETIC_TOTAL_SIZE} "
            f"synthetic samples. Adding {missing_samples} samples from training data."
        )

        additional_train_samples = dataset.shuffle(seed=84).select(range(missing_samples))
        additional_train_list = [sample for sample in additional_train_samples]

        logger.info(f"Added {len(additional_train_list)} samples from training data to supplement synthetic dataset")
        synthetic_dataset.extend(additional_train_list)

    return synthetic_dataset


async def prepare_text_task(task: AnyTextTypeRawTask, keypair: Keypair) -> tuple[str, str, str]:
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

    columns_to_sample = pick_columns_to_sample(task)

    if any(col not in train_ds.column_names for col in columns_to_sample):
        raise ValueError(f"Column {columns_to_sample} not found in train dataset")

    synthetic_ds = []
    try:
        if cst.GET_SYNTH_DATA:
            logger.info("Generating additional synthetic data")
            if isinstance(task, DpoRawTask):
                logger.info("DPO task: Generating synthetic dataset using TEXT_SYNTH_MODEL and TEXT_SYNTH_WEAKER_MODEL")
                synthetic_data_list = await asyncio.wait_for(
                    generate_synthetic_dpo_data(train_ds, keypair, task),
                    timeout=300
                )
                if synthetic_data_list and len(synthetic_data_list) >= cst.SYNTHETIC_TOTAL_SIZE:
                    synth_for_training = synthetic_data_list[:cst.SYNTHETIC_FOR_TRAINING]
                    synth_for_eval = synthetic_data_list[cst.SYNTHETIC_FOR_TRAINING:]

                    synth_train_dataset = Dataset.from_list(synth_for_training)
                    train_ds = concatenate_datasets([train_ds, synth_train_dataset])
                    logger.info(f"Training dataset size after adding {len(synth_for_training)} synthetic examples: {len(train_ds)}")

                    train_samples = train_ds.shuffle(seed=42).select(range(cst.SYNTH_EXAMPLES_FROM_TRAIN))
                    train_samples_list = [sample for sample in train_samples]

                    synthetic_ds = synth_for_eval + train_samples_list
                    logger.info(f"Created synthetic evaluation dataset with {len(synth_for_eval)} synthetic examples and {len(train_samples_list)} examples from training data")
                else:
                    logger.info("Not enough synthetic data generated, falling back to sampling from train")
                    _, synthetic_ds = assign_some_of_the_train_to_synth(train_ds, is_dpo=True)
            elif isinstance(task, InstructTextRawTask):
                logger.info("INSTRUCT TEXT task: Using same approach as DPO for synthetic data")
                # For instruct text tasks, use the same pattern as DPO tasks
                logger.info("Generating synthetic data for instruct text task")
                synthetic_ds = await get_additional_synth_data(test_ds, columns_to_sample, keypair, task=task)

                # Print an example from synthetic data for inspection
                if synthetic_ds and len(synthetic_ds) > 0:
                    example = synthetic_ds[0]
                    logger.info(f"Example synthetic data point for INSTRUCT task: {example}")

                if synthetic_ds and len(synthetic_ds) >= cst.SYNTHETIC_TOTAL_SIZE:
                    # Split synthetic data into training and evaluation sets
                    synth_for_training = synthetic_ds[:cst.SYNTHETIC_FOR_TRAINING]
                    synth_for_eval = synthetic_ds[cst.SYNTHETIC_FOR_TRAINING:]

                    synth_train_dataset = Dataset.from_list(synth_for_training)
                    train_ds = concatenate_datasets([train_ds, synth_train_dataset])
                    logger.info(f"Training dataset size after adding {len(synth_for_training)} synthetic examples: {len(train_ds)}")

                    # Sample from training data for synthetic evaluation
                    train_samples = train_ds.shuffle(seed=42).select(range(cst.SYNTH_EXAMPLES_FROM_TRAIN))
                    train_samples_list = [sample for sample in train_samples]

                    synthetic_ds = synth_for_eval + train_samples_list
                    logger.info(f"Created synthetic evaluation dataset with {len(synth_for_eval)} synthetic examples and {len(train_samples_list)} examples from training data")
                else:
                    logger.info("Not enough synthetic data generated, falling back to sampling from train")
                    _, synthetic_ds = assign_some_of_the_train_to_synth(train_ds, is_dpo=True)
            else:
                synthetic_ds = await get_additional_synth_data(test_ds, columns_to_sample, keypair, task=task)
        else:
            logger.info("Skipping synthetic data generation")
    except Exception as e:
        logger.info(f"Synthetic dataset gen is down, moving part of the train over: {e}")
        is_dpo = isinstance(task, DpoRawTask) or isinstance(task, InstructTextRawTask)
        train_ds, synthetic_ds = assign_some_of_the_train_to_synth(train_ds, is_dpo=is_dpo)

    if synthetic_ds is None:
        logger.info("There was not enough synthetic data created, using train samples instead")
        is_dpo = isinstance(task, DpoRawTask) or isinstance(task, InstructTextRawTask)
        train_ds, synthetic_ds = assign_some_of_the_train_to_synth(train_ds, is_dpo=is_dpo)

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

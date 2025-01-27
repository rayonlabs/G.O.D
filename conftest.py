import json
import subprocess

import minio
import pytest

from validator.core.config import load_config


def run(
    command: str,
    check=True,
) -> subprocess.CompletedProcess[str]:
    """
    Run a command outside pytest, useful for running external tooling.
    """
    try:
        print(f"RUNNING COMMAND: {command}")
        return subprocess.run(command.split(), check=check, capture_output=True, text=True)

    except subprocess.CalledProcessError as ex:
        # reraise using ex.stderr. this gives you the actual issue raised from
        # the command in pytest's test summary
        raise Exception(ex.stderr) from ex


@pytest.fixture(scope="session")
def test_config():
    return load_config()


@pytest.fixture(autouse=True)
def mock_nineteen_api(monkeypatch):
    def mock_nineteen():
        return '{"answer": "This is a test"}'

    def mock_process_row():
        return json.loads(mock_nineteen())

    monkeypatch.setattr("validator.utils.call_endpoint.post_to_nineteen_ai", mock_nineteen)
    monkeypatch.setattr("validator.augmentation.augmentation.process_row", mock_process_row)


@pytest.fixture(scope="session")
def container_runtime():
    """
    If the system has podman, use that, otherwise default to docker.
    """
    try:
        run("which podman")
        return "podman"

    except Exception:
        return "docker"


@pytest.fixture(scope="session", autouse=True)
def setup_storage(
    container_runtime,
    test_config,
):
    try:
        run(
            (
                "{} run --rm -d -p 19000:9000 "
                "--name pytest-god-minio "
                "-e MINIO_ROOT_USER={} "
                "-e MINIO_ROOT_PASSWORD={} "
                "minio/minio:latest "
                "server /data "
            ).format(
                container_runtime,
                test_config.s3_compatible_access_key,
                test_config.s3_compatible_secret_key,
            )
        )

        # create an images bucket and allow public, anonymous reads
        mn = minio.Minio(
            test_config.s3_compatible_endpoint,
            test_config.s3_compatible_access_key,
            test_config.s3_compatible_secret_key,
            secure=False,
        )
        if not mn.bucket_exists(test_config.s3_bucket_name):
            mn.make_bucket(test_config.s3_bucket_name)

        yield

    finally:
        run(f"{container_runtime} stop pytest-god-minio", False)

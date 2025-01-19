import asyncio
import re
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Dict
from typing import List

from asyncpg import Record

from validator.core.config import Config
from validator.core.models import AllModelSizes
from validator.db import constants as cst
from validator.db.sql.submissions_and_scoring import fetch_all_task_data
from validator.db.sql.submissions_and_scoring import fetch_model_params


async def get_all_model_size_distribution(config: Config) -> AllModelSizes:
    """Get model size distribution for all time intervals in a single database query.

    First try to parse model sizes using regex, if that fails, use content.gradients.io.
    """
    all_data = await fetch_all_task_data(config)
    now = datetime.now(timezone.utc)

    all_model_ids = {row[cst.MODEL_ID] for row in all_data}

    # attempt to first parse everything with regex
    model_id_to_size: Dict[str, float | None] = {model_id: parse_model_size_from_id(model_id) for model_id in all_model_ids}

    # fallback to CDN parameter value for any parsing fails returned by `parse_model_size_from_id`
    # as some models don't have size encoded in their name
    missing_model_ids = [model_id for model_id, size in model_id_to_size.items() if size is None]
    if missing_model_ids:
        fetched_sizes = await asyncio.gather(*(fetch_model_params(model_id, config.keypair) for model_id in missing_model_ids))
        model_id_to_size.update({model_id: size for model_id, size in zip(missing_model_ids, fetched_sizes)})

    # finally, format everything nicely
    model_id_to_size_key = {
        model_id: f"{size:.1f}B" if size % 1 != 0 else f"{int(size)}B" for model_id, size in model_id_to_size.items()
    }

    intervals = {
        "daily": now - timedelta(hours=24),
        "three_day": now - timedelta(days=3),
        "weekly": now - timedelta(days=7),
        "monthly": now - timedelta(days=30),
    }

    distributions = {
        "all_time": await count_model_size_distribution(
            all_data,
            model_id_to_size_key,
        ),
    }

    for interval_name, cutoff_time in intervals.items():
        filtered_interval_data = [row for row in all_data if row[cst.CREATED_AT] >= cutoff_time]

        distributions[interval_name] = await count_model_size_distribution(
            filtered_interval_data,
            model_id_to_size_key,
        )

    return AllModelSizes.model_validate(distributions)


async def count_model_size_distribution(
    rows: List[Record],
    model_id_to_size_key: Dict[str, str] = None,
) -> Dict[str, int]:
    distribution: Dict[str, int] = {}
    model_id_to_size_key = model_id_to_size_key if model_id_to_size_key else {}

    for row in rows:
        model_id = row[cst.MODEL_ID]
        size_key = model_id_to_size_key[model_id]
        distribution[size_key] = distribution.get(size_key, 0) + 1

    return dict(
        sorted(
            distribution.items(),
            # sort by size desc
            key=lambda x: float(x[0][:-1]),
            reverse=True,
        )
    )


def parse_model_size_from_id(
    model_id: str,
) -> float | None:
    # attempt to parse it out of the name
    model_id = model_id.lower()
    match = re.search(r".*?(\d+\.?\d*)[mb]", model_id)

    if not match:
        return None

    number = float(match.group(1))
    # convert mega to billions
    if model_id[model_id.find(match.group(1)) + len(match.group(1))] == "m":
        number = number / 1000.0

    return number

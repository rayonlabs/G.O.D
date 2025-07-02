from core.models.utility_models import DpoDatasetType
from miner.logic.job_handler import adapt_columns_for_dpo_dataset


if __name__ == "__main__":
    import pprint

    dataset_path = "/root/dpotest.json"  # ← Replace with your actual file

    dataset_type = DpoDatasetType(
        field_prompt="prompt",       # Replace if your keys are different
        field_system="system",
        field_chosen="chosen",
        field_rejected="rejected",
        prompt_format="{prompt}",
        chosen_format="{chosen}",
        rejected_format="{rejected}"
    )

    # Process and get the transformed dataset
    output_data = adapt_columns_for_dpo_dataset(dataset_path, dataset_type, apply_formatting=False)

    # Show some sample rows
    print("\n📄 Sample Records:")
    sample_size = min(3, len(output_data))
    for i in range(sample_size):
        print(f"\n--- Sample #{i+1} ---")
        pprint.pprint(output_data[i], width=120)

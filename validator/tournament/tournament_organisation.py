import asyncio

from fiber.chain.models import Node

from core.models.tournament_models import Round
from validator.tournament.organiser import organise_tournament_round
from validator.tournament.organiser import summarise_result
from validator.tournament.task_creator import create_image_tournament_round
from validator.tournament.task_creator import create_text_tournament_round


async def _real_task_creation_demo(round_data: Round, is_final: bool = False):
    print("\n--- REAL Task Creation Attempt ---")

    try:
        print("Loading real validator config...")
        from validator.core.config import load_config
        config = load_config()
        print("✅ Config loaded successfully")

        print(f"🔍 TOURNAMENT DEBUG: config.keypair type: {type(config.keypair)}")
        print(f"🔍 TOURNAMENT DEBUG: config.keypair: {config.keypair}")
        if hasattr(config.keypair, '_mock_name'):
            print(f"🚨 TOURNAMENT: Config keypair is a Mock: {config.keypair}")

        try:
            print("Attempting to create text tournament...")
            text_tournament = await create_text_tournament_round(round_data, config, is_final)
            print(f"✅ Text tournament created with {len(text_tournament.tasks)} tasks")
            print(f"   Task IDs: {text_tournament.tasks}")
        except Exception as e:
            print(f"❌ Text tournament creation failed: {type(e).__name__}: {e}")

        try:
            print("\nAttempting to create image tournament...")
            image_tournament = await create_image_tournament_round(round_data, config)
            print(f"✅ Image tournament created with {len(image_tournament.tasks)} tasks")
            print(f"   Task IDs: {image_tournament.tasks}")
        except Exception as e:
            print(f"❌ Image tournament creation failed: {type(e).__name__}: {e}")

    except Exception as e:
        print(f"❌ Config loading failed: {type(e).__name__}: {e}")
        print("Cannot proceed with real task creation without proper config.")

    print("\nNOTE: This attempts real task creation with actual validator config.")


if __name__ == "__main__":
    test_sizes = [8, 2]

    async def run_demo():
        for size in test_sizes:
            mock_nodes = []
            for i in range(1, size + 1):
                mock_node = Node(
                    hotkey=f"player_{i}",
                    coldkey=f"cold_{i}",
                    node_id=i,
                    netuid=1,
                    ip="127.0.0.1",
                    ip_type=4,
                    port=8080,
                    stake=100.0
                )
                mock_nodes.append(mock_node)

            result = organise_tournament_round(mock_nodes)
            summarise_result(result, size)

            is_final = size <= 2
            await _real_task_creation_demo(result, is_final)
            print()

    asyncio.run(run_demo())

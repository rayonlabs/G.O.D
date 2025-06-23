import asyncio

from fiber.chain.models import Node

from core.models.tournament_models import Round
from validator.tournament.organiser import organise_tournament_round
from validator.tournament.organiser import summarise_result
from validator.tournament.task_creator import create_image_tournament_round
from validator.tournament.task_creator import create_text_tournament_round
from validator.tournament.tournament_cycle import TournamentCycle, mock_submission_and_training, mock_evaluation
from validator.utils.logging import get_logger

logger = get_logger(__name__)


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


async def run_complete_tournament_demo():
    """Run a complete tournament demo with the new cycle system."""
    print("\n=== COMPLETE TOURNAMENT DEMO ===")
    
    try:
        from validator.core.config import load_config
        config = load_config()
        print("✅ Config loaded successfully")
        
        # Create tournament cycle
        tournament_cycle = TournamentCycle(config, config.psql_db)
        
        # Create mock participants
        test_sizes = [8, 12, 16, 24]
        
        for size in test_sizes:
            print(f"\n--- Testing with {size} participants ---")
            
            # Create mock nodes
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
            
            # Test round organization
            result = organise_tournament_round(mock_nodes)
            summarise_result(result, size)
            
            # Test task creation
            is_final = size <= 2
            await _real_task_creation_demo(result, is_final)
            
            # Create actual tournament
            try:
                from core.models.tournament_models import TournamentType
                tournament_id = await tournament_cycle.create_tournament(mock_nodes, TournamentType.TEXT)
                print(f"✅ Created tournament: {tournament_id}")
                
                # Mock the tournament cycle
                print("🔄 Mocking tournament cycle...")
                await mock_submission_and_training(tournament_id, config.psql_db)
                await mock_evaluation(tournament_id, config.psql_db)
                
                # Process one cycle
                print("🔄 Processing tournament cycle...")
                await tournament_cycle._process_tournament(
                    await tournament_cycle.psql_db.get_tournament(tournament_id, config.psql_db)
                )
                
            except Exception as e:
                print(f"❌ Tournament creation failed: {type(e).__name__}: {e}")
            
            print()
    
    except Exception as e:
        print(f"❌ Demo failed: {type(e).__name__}: {e}")
        print("Running basic demo without database...")
        await run_basic_demo()


async def run_basic_demo():
    """Run basic demo without database requirements."""
    print("\n=== BASIC TOURNAMENT DEMO ===")
    
    test_sizes = [8, 2]

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


if __name__ == "__main__":
    # Try to run complete demo, fall back to basic demo
    try:
        asyncio.run(run_complete_tournament_demo())
    except Exception as e:
        print(f"Complete demo failed: {e}")
        print("Falling back to basic demo...")
        asyncio.run(run_basic_demo())

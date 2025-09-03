#!/usr/bin/env python3
"""Simplified test file for boss round historical functionality."""

import pytest
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

# Use same import pattern as other test files
from core.models.utility_models import TaskStatus, TaskType
from validator.core.constants import NULL_ACCOUNT_ID
import validator.core.constants as cst


class TestHistoricalTaskSelection:
    """Test historical task selection from database."""
    
    @pytest.fixture
    def mock_psql_db(self):
        mock_db = MagicMock()
        mock_connection = AsyncMock()
        mock_db.connection = AsyncMock(return_value=mock_connection)
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=None)
        return mock_db
    
    @pytest.mark.asyncio
    async def test_get_random_historical_task_by_type_success(self, mock_psql_db):
        """Test successfully getting a random historical task."""
        from validator.db.sql.historical_tasks import get_random_historical_task_by_type
        
        expected_task_id = uuid4()
        mock_connection = await mock_psql_db.connection()
        mock_connection.fetchval = AsyncMock(return_value=expected_task_id)
        
        result = await get_random_historical_task_by_type(
            task_type="InstructTextTask",
            start_date="2025-06-01",
            end_date="2025-08-01",
            min_successful_scores=2,
            psql_db=mock_psql_db
        )
        
        assert result == expected_task_id
        mock_connection.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_random_historical_task_no_results(self, mock_psql_db):
        """Test when no historical tasks are found."""
        from validator.db.sql.historical_tasks import get_random_historical_task_by_type
        
        mock_connection = await mock_psql_db.connection()
        mock_connection.fetchval = AsyncMock(return_value=None)
        
        result = await get_random_historical_task_by_type(
            task_type="GrpoTask",
            start_date="2025-06-01",
            end_date="2025-08-01",
            min_successful_scores=10,
            psql_db=mock_psql_db
        )
        
        assert result is None


class TestBossRoundTaskCopying:
    """Test task copying functions for boss rounds."""
    
    @pytest.fixture
    def mock_psql_db(self):
        mock_db = MagicMock()
        mock_connection = AsyncMock()
        mock_db.connection = AsyncMock(return_value=mock_connection)
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=None)
        return mock_db
    
    @pytest.fixture
    def sample_task_dict(self):
        """Create a sample task dict for testing."""
        return {
            'task_id': uuid4(),
            'task_type': TaskType.INSTRUCTTEXTTASK.value,
            'status': TaskStatus.SUCCESS.value,
            'account_id': uuid4(),
            'is_organic': True,
            'model_repo': "test/model",
            'base_model': "llama",
            'dataset': "test_dataset",
            'trained_model_repository': "trained/model",
            'result_model_name': "result_model",
            'field_system': "system",
            'field_instruction': "instruction",
            'field_input': "input",
            'field_output': "output",
            'format': "{instruction}",
            'no_input_format': "{instruction}",
            'max_input_length': 1024,
            'created_at': datetime.utcnow(),
            'times_delayed': 0,
            'n_eval_attempts': 1,
        }
    
    @pytest.mark.asyncio
    async def test_copy_historical_task_into_boss_round(self, mock_psql_db, sample_task_dict):
        """Test copying a historical task into a boss round tournament."""
        from validator.tournament.boss_round_sync import copy_historical_task_into_boss_round_tournament
        
        # Create a mock task object instead of importing InstructTextTask
        sample_task = MagicMock()
        for key, value in sample_task_dict.items():
            setattr(sample_task, key, value)
        sample_task.model_copy = MagicMock(return_value=MagicMock(**sample_task_dict))
        historical_task_id = str(sample_task.task_id)
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = str(uuid4())
        
        with patch('validator.tournament.boss_round_sync.get_task', return_value=sample_task):
            with patch('validator.tournament.boss_round_sync.add_task') as mock_add_task:
                with patch('validator.tournament.boss_round_sync.add_tournament_tasks') as mock_add_tournament:
                    mock_connection = await mock_psql_db.connection()
                    mock_connection.execute = AsyncMock()
                    
                    result = await copy_historical_task_into_boss_round_tournament(
                        historical_task_id=historical_task_id,
                        tournament_id=tournament_id,
                        round_id=round_id,
                        pair_id=pair_id,
                        psql_db=mock_psql_db
                    )
                    
                    # Verify task was copied with correct attributes
                    assert result is not None
                    print(f"\nOriginal task ID: {sample_task.task_id}")
                    print(f"New tournament task ID: {result.task_id}")
                    assert result.task_id != sample_task.task_id  # New ID
                    assert result.status == TaskStatus.PENDING
                    assert result.is_organic == False
                    assert str(result.account_id) == cst.NULL_ACCOUNT_ID
                    assert result.times_delayed == 0
                    assert result.assigned_miners is None
                    assert result.n_eval_attempts == 0
                    
                    # Verify task was added
                    mock_add_task.assert_called_once()
                    added_task = mock_add_task.call_args[0][0]
                    print(f"Task added with status: {added_task.status}")
                    
                    # Verify tournament task entry was created
                    mock_add_tournament.assert_called_once()
                    tournament_entry = mock_add_tournament.call_args[0][0][0]
                    assert tournament_entry.tournament_id == tournament_id
                    assert tournament_entry.round_id == round_id
                    assert tournament_entry.pair_id == pair_id
                    assert tournament_entry.task_id == result.task_id
                    
                    # Verify sync link was recorded
                    mock_connection.execute.assert_called_once()
                    sql_args = mock_connection.execute.call_args[0]
                    assert "boss_round_synced_tasks" in sql_args[0]
                    assert str(result.task_id) == str(sql_args[1])  # tournament_task_id
                    assert historical_task_id == str(sql_args[2])  # general_task_id (historical)
                    print(f"Sync link: tournament_task {sql_args[1]} -> historical_task {sql_args[2]}")
    
    @pytest.mark.asyncio
    async def test_copy_tournament_task_to_general_pool(self, mock_psql_db, sample_task_dict):
        """Test copying a failed tournament task to general miner pool."""
        from validator.tournament.boss_round_sync import copy_tournament_task_into_general_miner_pool
        
        sample_task_dict['status'] = TaskStatus.FAILURE.value
        # Create a mock task object
        sample_task = MagicMock()
        for key, value in sample_task_dict.items():
            setattr(sample_task, key, value)
        sample_task.model_copy = MagicMock(return_value=MagicMock(**sample_task_dict))
        tournament_task_id = str(sample_task.task_id)
        
        with patch('validator.tournament.boss_round_sync.get_task', return_value=sample_task):
            with patch('validator.tournament.boss_round_sync.add_task') as mock_add_task:
                mock_connection = await mock_psql_db.connection()
                mock_connection.execute = AsyncMock()
                
                result = await copy_tournament_task_into_general_miner_pool(
                    tournament_task_id=tournament_task_id,
                    psql_db=mock_psql_db
                )
                
                # Verify task was copied with correct attributes
                assert result is not None
                print(f"\nFailed tournament task ID: {sample_task.task_id}")
                print(f"New general pool task ID: {result.task_id}")
                assert result.task_id != sample_task.task_id  # New ID
                assert result.status == TaskStatus.LOOKING_FOR_NODES  # Ready for general miners
                assert result.is_organic == False
                assert str(result.account_id) == cst.NULL_ACCOUNT_ID
                
                # Verify task was added
                mock_add_task.assert_called_once()
                
                # Verify sync link was recorded
                mock_connection.execute.assert_called_once()
                sql_args = mock_connection.execute.call_args[0]
                assert "boss_round_synced_tasks" in sql_args[0]
                assert tournament_task_id == str(sql_args[1])  # tournament_task_id (original)
                assert str(result.task_id) == str(sql_args[2])  # general_task_id (new copy)
                print(f"Sync link: tournament_task {sql_args[1]} -> general_task {sql_args[2]}")


class TestSyncLinkManagement:
    """Test boss_round_synced_tasks table operations."""
    
    @pytest.fixture
    def mock_psql_db(self):
        mock_db = MagicMock()
        mock_connection = AsyncMock()
        mock_db.connection = AsyncMock(return_value=mock_connection)
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=None)
        return mock_db
    
    @pytest.mark.asyncio
    async def test_get_synced_task_id(self, mock_psql_db):
        """Test retrieving a single synced task ID."""
        from validator.tournament.boss_round_sync import get_synced_task_id
        
        tournament_task_id = str(uuid4())
        expected_general_id = str(uuid4())
        
        mock_connection = await mock_psql_db.connection()
        mock_connection.fetchval = AsyncMock(return_value=expected_general_id)
        
        result = await get_synced_task_id(tournament_task_id, mock_psql_db)
        
        assert result == expected_general_id
        mock_connection.fetchval.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_synced_task_ids_batch(self, mock_psql_db):
        """Test retrieving multiple synced task IDs."""
        from validator.tournament.boss_round_sync import get_synced_task_ids
        
        tournament_ids = [str(uuid4()) for _ in range(3)]
        general_ids = [str(uuid4()) for _ in range(3)]
        
        mock_connection = await mock_psql_db.connection()
        mock_connection.fetch = AsyncMock(return_value=[(gid,) for gid in general_ids])
        
        result = await get_synced_task_ids(tournament_ids, mock_psql_db)
        
        assert result == general_ids
        mock_connection.fetch.assert_called_once()


class TestBossRoundTaskCreation:
    """Test boss round task creation with historical tasks."""
    
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.psql_db = MagicMock()
        return config
    
    @pytest.mark.asyncio
    async def test_create_historical_text_boss_round_tasks_success(self, mock_config):
        """Test creating text boss round tasks from historical data."""
        from validator.tournament.task_creator import _create_historical_text_boss_round_tasks
        
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        
        print("\n=== TEXT BOSS ROUND TASK CREATION TEST ===")
        
        # Mock existing tasks check (no existing tasks)
        with patch('validator.tournament.task_creator.get_tournament_tasks', return_value=[]):
            # Mock historical task IDs - one for each type
            historical_instruct_id = uuid4()
            historical_dpo_id = uuid4()  
            historical_grpo_id = uuid4()
            historical_ids = [historical_instruct_id, historical_dpo_id, historical_grpo_id]
            
            with patch('validator.tournament.task_creator.get_random_historical_task_by_type', 
                      side_effect=historical_ids) as mock_get_historical:
                # Mock task copying
                copied_tasks = []
                for i, hist_id in enumerate(historical_ids):
                    task = MagicMock()
                    task.task_id = uuid4()
                    copied_tasks.append(task)
                    print(f"Historical task {hist_id} -> Tournament task {task.task_id}")
                
                with patch('validator.tournament.task_creator.copy_historical_task_into_boss_round_tournament',
                          side_effect=copied_tasks):
                    
                    result = await _create_historical_text_boss_round_tasks(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        config=mock_config
                    )
                    
                    # Should create 3 tasks (one of each type)
                    assert len(result) == 3
                    
                    # Verify each task type was requested
                    calls = mock_get_historical.call_args_list
                    assert len(calls) == 3
                    
                    # Check that each task type was requested with correct parameters
                    for i, call in enumerate(calls):
                        task_type = call[1]['task_type']
                        print(f"Requested task type {i+1}: {task_type}")
                        assert call[1]['start_date'] == '2025-06-01'
                        assert call[1]['end_date'] == '2025-08-01'
                        assert call[1]['min_successful_scores'] == 2
                    
                    # Verify the task types requested
                    requested_types = [call[1]['task_type'] for call in calls]
                    assert 'InstructTextTask' in requested_types
                    assert 'DpoTask' in requested_types
                    assert 'GrpoTask' in requested_types
                    
                    print(f"✅ Created {len(result)} boss round text tasks from historical data")
    
    @pytest.mark.asyncio
    async def test_create_historical_image_boss_round_tasks_success(self, mock_config):
        """Test creating image boss round tasks from historical data."""
        from validator.tournament.task_creator import _create_historical_image_boss_round_tasks
        
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        
        with patch('validator.tournament.task_creator.get_tournament_tasks', return_value=[]):
            # Mock 3 historical image task IDs
            historical_ids = [uuid4(), uuid4(), uuid4()]
            with patch('validator.tournament.task_creator.get_random_historical_task_by_type',
                      side_effect=historical_ids):
                copied_tasks = []
                for hist_id in historical_ids:
                    task = MagicMock()
                    task.task_id = uuid4()
                    copied_tasks.append(task)
                
                with patch('validator.tournament.task_creator.copy_historical_task_into_boss_round_tournament',
                          side_effect=copied_tasks):
                    
                    result = await _create_historical_image_boss_round_tasks(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        config=mock_config
                    )
                    
                    # Should create 3 image tasks
                    assert len(result) == 3
    
    @pytest.mark.asyncio
    async def test_create_historical_tasks_none_found(self, mock_config):
        """Test handling when no historical tasks are found."""
        from validator.tournament.task_creator import _create_historical_text_boss_round_tasks
        
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        
        with patch('validator.tournament.task_creator.get_tournament_tasks', return_value=[]):
            with patch('validator.tournament.task_creator.get_random_historical_task_by_type',
                      return_value=None) as mock_get_historical:
                
                # The function logs errors but doesn't raise - it returns empty list
                result = await _create_historical_text_boss_round_tasks(
                    tournament_id=tournament_id,
                    round_id=round_id,
                    config=mock_config
                )
                
                # Should return empty list when no historical tasks found
                assert result == []
                
                # Should have attempted to get each task type
                assert mock_get_historical.call_count == 3
                print(f"\nAttempted to fetch {mock_get_historical.call_count} task types")
                print("No historical tasks found - returned empty list as expected")


class TestIntegrationFlow:
    """Test the complete integration flow with detailed assertions."""
    
    @pytest.fixture
    def mock_psql_db(self):
        mock_db = MagicMock()
        mock_connection = AsyncMock()
        mock_db.connection = AsyncMock(return_value=mock_connection)
        mock_connection.__aenter__ = AsyncMock(return_value=mock_connection)
        mock_connection.__aexit__ = AsyncMock(return_value=None)
        return mock_db
    
    @pytest.mark.asyncio
    async def test_complete_boss_round_historical_flow(self, mock_psql_db):
        """Test the complete flow from historical task selection to tournament execution."""
        from validator.db.sql.historical_tasks import get_random_historical_task_by_type
        from validator.tournament.boss_round_sync import (
            copy_historical_task_into_boss_round_tournament,
            get_synced_task_id
        )
        
        print("\n=== COMPLETE BOSS ROUND HISTORICAL FLOW TEST ===")
        
        # Step 1: Simulate finding a historical task
        historical_task_id = uuid4()
        mock_connection = await mock_psql_db.connection()
        mock_connection.fetchval = AsyncMock(return_value=historical_task_id)
        
        selected_id = await get_random_historical_task_by_type(
            task_type=TaskType.INSTRUCTTEXTTASK.value,
            start_date=cst.BOSS_ROUND_HISTORICAL_START_DATE,
            end_date=cst.BOSS_ROUND_HISTORICAL_END_DATE,
            min_successful_scores=cst.MIN_SUCCESSFUL_SCORES_FOR_HISTORICAL_TASK,
            psql_db=mock_psql_db
        )
        
        assert selected_id == historical_task_id
        print(f"Step 1: Selected historical task: {historical_task_id}")
        
        # Step 2: Copy historical task to tournament
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = str(uuid4())
        
        # Create mock historical task
        historical_task_mock = MagicMock()
        historical_task_mock.task_id = historical_task_id
        historical_task_mock.task_type = TaskType.INSTRUCTTEXTTASK
        historical_task_mock.status = TaskStatus.SUCCESS
        historical_task_mock.is_organic = True
        historical_task_mock.account_id = uuid4()
        historical_task_mock.model_copy = MagicMock(return_value=MagicMock(
            task_id=historical_task_id,
            task_type=TaskType.INSTRUCTTEXTTASK,
            status=TaskStatus.SUCCESS
        ))
        
        with patch('validator.tournament.boss_round_sync.get_task', return_value=historical_task_mock):
            with patch('validator.tournament.boss_round_sync.add_task'):
                with patch('validator.tournament.boss_round_sync.add_tournament_tasks'):
                    mock_connection.execute = AsyncMock()
                    
                    tournament_task = await copy_historical_task_into_boss_round_tournament(
                        historical_task_id=str(historical_task_id),
                        tournament_id=tournament_id,
                        round_id=round_id,
                        pair_id=pair_id,
                        psql_db=mock_psql_db
                    )
                    
                    assert tournament_task is not None
                    assert tournament_task.task_id != historical_task_id
                    assert tournament_task.status == TaskStatus.PENDING
                    print(f"Step 2: Created tournament task: {tournament_task.task_id}")
                    print(f"  - Status: {tournament_task.status}")
                    print(f"  - Is organic: {tournament_task.is_organic}")
        
        # Step 3: Verify sync link retrieval
        mock_connection.fetchval = AsyncMock(return_value=str(historical_task_id))
        
        synced_id = await get_synced_task_id(str(tournament_task.task_id), mock_psql_db)
        assert synced_id == str(historical_task_id)
        print(f"Step 3: Verified sync link: tournament {tournament_task.task_id} -> historical {synced_id}")
        
        # Step 4: Simulate performance comparison flow
        print("\nStep 4: Performance Comparison Flow:")
        print(f"  - Tournament miners will train on task {tournament_task.task_id}")
        print(f"  - Results will be compared against historical task {historical_task_id}")
        print(f"  - Best historical score will be used as 'boss' baseline")
        print(f"  - Tournament winner's score will be compared against this baseline")
        
        print("\n✅ Complete flow verified successfully!")
    
    @pytest.mark.asyncio
    async def test_failed_tournament_task_sync_flow(self, mock_psql_db):
        """Test the flow when a tournament task fails and needs to be synced to general pool."""
        from validator.tournament.boss_round_sync import copy_tournament_task_into_general_miner_pool
        
        print("\n=== FAILED TASK SYNC FLOW TEST ===")
        
        # Create a failed tournament task
        failed_tournament_task_id = uuid4()
        failed_task = MagicMock()
        failed_task.task_id = failed_tournament_task_id
        failed_task.status = TaskStatus.FAILURE
        failed_task.task_type = TaskType.DPOTASK
        failed_task.is_organic = False
        failed_task.model_copy = MagicMock(return_value=MagicMock(
            task_id=failed_tournament_task_id,
            status=TaskStatus.FAILURE
        ))
        
        with patch('validator.tournament.boss_round_sync.get_task', return_value=failed_task):
            with patch('validator.tournament.boss_round_sync.add_task'):
                mock_connection = await mock_psql_db.connection()
                mock_connection.execute = AsyncMock()
                
                general_task = await copy_tournament_task_into_general_miner_pool(
                    tournament_task_id=str(failed_tournament_task_id),
                    psql_db=mock_psql_db
                )
                
                assert general_task is not None
                assert general_task.task_id != failed_tournament_task_id
                assert general_task.status == TaskStatus.LOOKING_FOR_NODES
                
                print(f"Failed tournament task {failed_tournament_task_id} synced to general pool")
                print(f"New general task ID: {general_task.task_id}")
                print(f"Status: {general_task.status} (ready for general miners)")
                
                # Verify sync link
                sql_args = mock_connection.execute.call_args[0]
                assert str(failed_tournament_task_id) == str(sql_args[1])
                assert str(general_task.task_id) == str(sql_args[2])
                print(f"Sync link recorded: {sql_args[1]} -> {sql_args[2]}")
                
        print("\n✅ Failed task sync flow verified successfully!")


class TestHistoricalTasksWithPartialExisting:
    """Test boss round creation when some tasks already exist."""
    
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.psql_db = MagicMock()
        return config
    
    @pytest.mark.asyncio
    async def test_create_text_boss_round_with_partial_existing(self, mock_config):
        """Test creating text boss round when 1 task already exists."""
        from validator.tournament.task_creator import _create_historical_text_boss_round_tasks
        from core.models.tournament_models import TournamentTask
        
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        
        print("\n=== TEXT BOSS ROUND WITH PARTIAL EXISTING TASKS ===")
        
        # Mock that 1 task already exists (e.g., InstructText already created)
        existing_instruct_task = TournamentTask(
            tournament_id=tournament_id,
            round_id=round_id,
            task_id=uuid4(),
            pair_id=f"{round_id}_pair_001",
            group_id=None
        )
        
        with patch('validator.tournament.task_creator.get_tournament_tasks', 
                  return_value=[existing_instruct_task]):
            print(f"Existing task found: {existing_instruct_task.task_id}")
            
            # We should only need to fetch 2 more tasks (DPO and GRPO)
            historical_dpo_id = uuid4()
            historical_grpo_id = uuid4()
            
            with patch('validator.tournament.task_creator.get_random_historical_task_by_type',
                      side_effect=[historical_dpo_id, historical_grpo_id]) as mock_get_historical:
                
                # Mock the existing task retrieval
                existing_raw_task = MagicMock()
                existing_raw_task.task_id = existing_instruct_task.task_id
                
                with patch('validator.db.sql.tasks.get_task', 
                          return_value=existing_raw_task):
                    
                    # Mock copying for the 2 new tasks
                    new_tasks = []
                    for hist_id in [historical_dpo_id, historical_grpo_id]:
                        task = MagicMock()
                        task.task_id = uuid4()
                        new_tasks.append(task)
                        print(f"Creating new task from historical {hist_id} -> {task.task_id}")
                    
                    with patch('validator.tournament.task_creator.copy_historical_task_into_boss_round_tournament',
                              side_effect=new_tasks):
                        
                        result = await _create_historical_text_boss_round_tasks(
                            tournament_id=tournament_id,
                            round_id=round_id,
                            config=mock_config
                        )
                        
                        # Should have 3 tasks total (1 existing + 2 new)
                        assert len(result) == 3
                        
                        # Should only have called get_random_historical 2 times
                        assert mock_get_historical.call_count == 2
                        print(f"✅ Only fetched {mock_get_historical.call_count} new historical tasks")
                        
                        # Check that we didn't try to fetch InstructText since it exists
                        called_types = [call[1]['task_type'] for call in mock_get_historical.call_args_list]
                        assert 'InstructTextTask' not in called_types
                        assert 'DpoTask' in called_types
                        assert 'GrpoTask' in called_types
                        print(f"Task types fetched: {called_types}")
    
    @pytest.mark.asyncio
    async def test_create_image_boss_round_with_partial_existing(self, mock_config):
        """Test creating image boss round when 2 tasks already exist."""
        from validator.tournament.task_creator import _create_historical_image_boss_round_tasks
        from core.models.tournament_models import TournamentTask
        
        tournament_id = str(uuid4())
        round_id = str(uuid4())
        pair_id = f"{round_id}_pair_001"
        
        print("\n=== IMAGE BOSS ROUND WITH PARTIAL EXISTING TASKS ===")
        
        # Mock that 2 image tasks already exist
        existing_tasks = [
            TournamentTask(
                tournament_id=tournament_id,
                round_id=round_id,
                task_id=uuid4(),
                pair_id=pair_id,
                group_id=None
            ),
            TournamentTask(
                tournament_id=tournament_id,
                round_id=round_id,
                task_id=uuid4(),
                pair_id=pair_id,
                group_id=None
            )
        ]
        
        with patch('validator.tournament.task_creator.get_tournament_tasks', 
                  return_value=existing_tasks):
            print(f"Found {len(existing_tasks)} existing tasks")
            
            # We should only need to fetch 1 more task (3 total - 2 existing = 1)
            new_historical_id = uuid4()
            
            with patch('validator.tournament.task_creator.get_random_historical_task_by_type',
                      return_value=new_historical_id) as mock_get_historical:
                
                # Mock the existing tasks retrieval
                existing_raw_tasks = []
                for existing_task in existing_tasks:
                    raw_task = MagicMock()
                    raw_task.task_id = existing_task.task_id
                    existing_raw_tasks.append(raw_task)
                
                with patch('validator.db.sql.tasks.get_task', 
                          side_effect=existing_raw_tasks):
                    
                    # Mock copying for the 1 new task
                    new_task = MagicMock()
                    new_task.task_id = uuid4()
                    print(f"Creating new task from historical {new_historical_id} -> {new_task.task_id}")
                    
                    with patch('validator.tournament.task_creator.copy_historical_task_into_boss_round_tournament',
                              return_value=new_task):
                        
                        result = await _create_historical_image_boss_round_tasks(
                            tournament_id=tournament_id,
                            round_id=round_id,
                            config=mock_config
                        )
                        
                        # Should have 3 tasks total (2 existing + 1 new)
                        assert len(result) == 3
                        
                        # Should only have called get_random_historical 1 time
                        assert mock_get_historical.call_count == 1
                        print(f"✅ Only fetched {mock_get_historical.call_count} new historical task")
                        
                        # Verify parameters
                        call_args = mock_get_historical.call_args[1]
                        assert call_args['task_type'] == 'ImageTask'
                        assert call_args['start_date'] == '2025-06-01'
                        assert call_args['end_date'] == '2025-08-01'
                        assert call_args['min_successful_scores'] == 2
                        print(f"Parameters verified for historical task fetch")
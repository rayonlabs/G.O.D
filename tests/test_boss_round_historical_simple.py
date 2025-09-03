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
        from core.models.task_models import InstructTextTask
        
        sample_task = InstructTextTask(**sample_task_dict)
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
                    assert result.task_id != sample_task.task_id  # New ID
                    assert result.status == TaskStatus.PENDING
                    assert result.is_organic == False
                    assert result.account_id == UUID(cst.NULL_ACCOUNT_ID)
                    
                    # Verify task was added
                    mock_add_task.assert_called_once()
                    
                    # Verify tournament task entry was created
                    mock_add_tournament.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_copy_tournament_task_to_general_pool(self, mock_psql_db, sample_task_dict):
        """Test copying a failed tournament task to general miner pool."""
        from validator.tournament.boss_round_sync import copy_tournament_task_into_general_miner_pool
        from core.models.task_models import InstructTextTask
        
        sample_task_dict['status'] = TaskStatus.FAILED.value
        sample_task = InstructTextTask(**sample_task_dict)
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
                assert result.task_id != sample_task.task_id  # New ID
                assert result.status == TaskStatus.LOOKING_FOR_NODES
                assert result.is_organic == False
                
                # Verify task was added
                mock_add_task.assert_called_once()


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
        
        # Mock existing tasks check (no existing tasks)
        with patch('validator.tournament.task_creator.get_tournament_tasks', return_value=[]):
            # Mock historical task IDs
            historical_ids = [uuid4(), uuid4(), uuid4()]
            with patch('validator.tournament.task_creator.get_random_historical_task_by_type', 
                      side_effect=historical_ids):
                # Mock task copying
                copied_tasks = []
                for hist_id in historical_ids:
                    task = MagicMock()
                    task.task_id = uuid4()
                    copied_tasks.append(task)
                
                with patch('validator.tournament.task_creator.copy_historical_task_into_boss_round_tournament',
                          side_effect=copied_tasks):
                    
                    result = await _create_historical_text_boss_round_tasks(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        config=mock_config
                    )
                    
                    # Should create 3 tasks (one of each type)
                    assert len(result) == 3
    
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
                      return_value=None):
                
                # Should raise an error when no historical tasks found
                with pytest.raises(ValueError, match="No historical .* tasks found"):
                    await _create_historical_text_boss_round_tasks(
                        tournament_id=tournament_id,
                        round_id=round_id,
                        config=mock_config
                    )
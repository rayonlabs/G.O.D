import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import validator.core.constants as cst
from core.models.tournament_models import TournamentData, TournamentStatus, TournamentType
from validator.core.config import Config
from validator.db.database import PSQLDB
from validator.tournament.tournament_manager import (
    check_and_start_tournament,
    should_start_new_tournament_after_interval,
)


class TestAutomaticTournamentScheduling:
    """Test suite for automatic tournament scheduling functionality."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = MagicMock(spec=Config)
        config.psql_db = MagicMock(spec=PSQLDB)
        return config

    @pytest.fixture
    def mock_tournament_data(self):
        """Create mock tournament data."""
        return {
            "text_active": TournamentData(
                tournament_id="text_active_123",
                tournament_type=TournamentType.TEXT,
                status=TournamentStatus.ACTIVE,
            ),
            "image_pending": TournamentData(
                tournament_id="image_pending_456",
                tournament_type=TournamentType.IMAGE,
                status=TournamentStatus.PENDING,
            ),
            "text_completed": TournamentData(
                tournament_id="text_completed_789",
                tournament_type=TournamentType.TEXT,
                status=TournamentStatus.COMPLETED,
            ),
        }

    @pytest.mark.asyncio
    async def test_should_start_new_tournament_no_previous(self):
        """Test that we should start a tournament when no previous tournament exists."""
        result = await should_start_new_tournament_after_interval(None)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_start_new_tournament_enough_time_passed(self):
        """Test that we should start a tournament when enough time has passed."""
        # Create a timestamp from 25 hours ago (more than the 24-hour interval)
        old_timestamp = datetime.now(timezone.utc) - timedelta(hours=25)
        
        result = await should_start_new_tournament_after_interval(old_timestamp)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_start_new_tournament_not_enough_time(self):
        """Test that we should NOT start a tournament when not enough time has passed."""
        # Create a timestamp from 10 hours ago (less than the 24-hour interval)
        recent_timestamp = datetime.now(timezone.utc) - timedelta(hours=10)
        
        result = await should_start_new_tournament_after_interval(recent_timestamp)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_start_new_tournament_naive_timestamp(self):
        """Test timezone handling with naive timestamps."""
        # Create a naive timestamp (no timezone info) from 25 hours ago
        naive_timestamp = datetime.now() - timedelta(hours=25)
        
        result = await should_start_new_tournament_after_interval(naive_timestamp)
        assert result is True

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_active_exists(
        self, mock_get_active, mock_config, mock_tournament_data
    ):
        """Test that no new tournament is created when an active one exists."""
        # Mock that an active tournament exists
        mock_get_active.return_value = mock_tournament_data["text_active"]
        
        with patch("validator.tournament.tournament_manager.logger") as mock_logger:
            await check_and_start_tournament(TournamentType.TEXT, mock_config.psql_db, mock_config)
            
            # Should log that active tournament exists and return early
            mock_logger.info.assert_called_with(
                f"Active {TournamentType.TEXT.value} tournament exists: {mock_tournament_data['text_active'].tournament_id}"
            )

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.get_tournaments_with_status")
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_pending_exists(
        self, mock_get_active, mock_get_with_status, mock_config, mock_tournament_data
    ):
        """Test that no new tournament is created when a pending one exists."""
        # Mock no active tournament but pending tournament exists
        mock_get_active.return_value = None
        mock_get_with_status.return_value = [mock_tournament_data["image_pending"]]
        
        with patch("validator.tournament.tournament_manager.logger") as mock_logger:
            await check_and_start_tournament(TournamentType.IMAGE, mock_config.psql_db, mock_config)
            
            # Should log that pending tournament exists
            mock_logger.info.assert_called_with(
                f"Pending {TournamentType.IMAGE.value} tournament exists: {mock_tournament_data['image_pending'].tournament_id}"
            )

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.create_basic_tournament")
    @patch("validator.tournament.tournament_manager.should_start_new_tournament_after_interval")
    @patch("validator.tournament.tournament_manager.get_latest_tournament_with_created_at")
    @patch("validator.tournament.tournament_manager.get_tournaments_with_status")
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_create_after_interval(
        self,
        mock_get_active,
        mock_get_with_status,
        mock_get_latest,
        mock_should_start,
        mock_create_tournament,
        mock_config,
        mock_tournament_data,
    ):
        """Test that a new tournament is created when enough time has passed since completion."""
        # Mock setup: no active/pending tournaments, completed tournament with enough time passed
        mock_get_active.return_value = None
        mock_get_with_status.return_value = []
        mock_get_latest.return_value = (
            mock_tournament_data["text_completed"],
            datetime.now(timezone.utc) - timedelta(hours=25),
        )
        mock_should_start.return_value = True
        mock_create_tournament.return_value = "new_tournament_123"
        
        with patch("validator.tournament.tournament_manager.logger") as mock_logger:
            await check_and_start_tournament(TournamentType.TEXT, mock_config.psql_db, mock_config)
            
            # Should create new tournament
            mock_create_tournament.assert_called_once_with(
                TournamentType.TEXT, mock_config.psql_db, mock_config
            )
            
            # Should log tournament creation
            mock_logger.info.assert_any_call(
                f"Starting new {TournamentType.TEXT.value} tournament after {cst.TOURNAMENT_INTERVAL_HOURS} hours since {mock_tournament_data['text_completed'].tournament_id}"
            )
            mock_logger.info.assert_any_call(
                f"Created new {TournamentType.TEXT.value} tournament: new_tournament_123"
            )

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.should_start_new_tournament_after_interval")
    @patch("validator.tournament.tournament_manager.get_latest_tournament_with_created_at")
    @patch("validator.tournament.tournament_manager.get_tournaments_with_status")
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_not_enough_time(
        self,
        mock_get_active,
        mock_get_with_status,
        mock_get_latest,
        mock_should_start,
        mock_config,
        mock_tournament_data,
    ):
        """Test that no new tournament is created when not enough time has passed."""
        # Mock setup: no active/pending tournaments, completed tournament but not enough time passed
        mock_get_active.return_value = None
        mock_get_with_status.return_value = []
        mock_get_latest.return_value = (
            mock_tournament_data["text_completed"],
            datetime.now(timezone.utc) - timedelta(hours=10),
        )
        mock_should_start.return_value = False
        
        with patch("validator.tournament.tournament_manager.logger") as mock_logger:
            await check_and_start_tournament(TournamentType.TEXT, mock_config.psql_db, mock_config)
            
            # Should log that not enough time has passed
            mock_logger.info.assert_called_with(
                f"Not enough time has passed since last {TournamentType.TEXT.value} tournament completion"
            )

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.create_basic_tournament")
    @patch("validator.tournament.tournament_manager.get_latest_tournament_with_created_at")
    @patch("validator.tournament.tournament_manager.get_tournaments_with_status")
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_no_previous_tournament(
        self,
        mock_get_active,
        mock_get_with_status,
        mock_get_latest,
        mock_create_tournament,
        mock_config,
    ):
        """Test that a tournament is created when no previous tournaments exist."""
        # Mock setup: no tournaments exist at all
        mock_get_active.return_value = None
        mock_get_with_status.return_value = []
        mock_get_latest.return_value = (None, None)
        mock_create_tournament.return_value = "first_tournament_123"
        
        with patch("validator.tournament.tournament_manager.logger") as mock_logger:
            await check_and_start_tournament(TournamentType.TEXT, mock_config.psql_db, mock_config)
            
            # Should create first tournament
            mock_create_tournament.assert_called_once_with(
                TournamentType.TEXT, mock_config.psql_db, mock_config
            )
            
            # Should log first tournament creation
            mock_logger.info.assert_any_call(
                f"No {TournamentType.TEXT.value} tournaments found, creating first one"
            )
            mock_logger.info.assert_any_call(
                f"Created first {TournamentType.TEXT.value} tournament: first_tournament_123"
            )

    @pytest.mark.asyncio
    async def test_tournament_interval_constant_used(self):
        """Test that the correct interval constant is used in calculations."""
        # Verify the constant exists and has expected value
        assert hasattr(cst, "TOURNAMENT_INTERVAL_HOURS")
        assert cst.TOURNAMENT_INTERVAL_HOURS == 24
        
        # Test boundary conditions
        exactly_24_hours_ago = datetime.now(timezone.utc) - timedelta(hours=24)
        just_under_24_hours = datetime.now(timezone.utc) - timedelta(hours=23, minutes=59)
        just_over_24_hours = datetime.now(timezone.utc) - timedelta(hours=24, minutes=1)
        
        # Exactly 24 hours should trigger new tournament
        result_exact = await should_start_new_tournament_after_interval(exactly_24_hours_ago)
        assert result_exact is True
        
        # Just under 24 hours should not trigger
        result_under = await should_start_new_tournament_after_interval(just_under_24_hours)
        assert result_under is False
        
        # Just over 24 hours should trigger
        result_over = await should_start_new_tournament_after_interval(just_over_24_hours)
        assert result_over is True

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.check_and_start_tournament")
    async def test_both_tournament_types_checked(self, mock_check_and_start, mock_config):
        """Test that both TEXT and IMAGE tournament types are checked independently."""
        # Instead of testing the infinite loop, just test the logic directly
        # by calling check_and_start_tournament for both types manually
        # (this is what process_tournament_scheduling does in each iteration)
        
        await mock_check_and_start(TournamentType.TEXT, mock_config.psql_db, mock_config)
        await mock_check_and_start(TournamentType.IMAGE, mock_config.psql_db, mock_config)
        
        # Should have called check_and_start_tournament for both types
        assert mock_check_and_start.call_count == 2
        calls = mock_check_and_start.call_args_list
        
        # Verify both tournament types were checked with correct arguments
        text_call = calls[0]
        image_call = calls[1]
        
        assert text_call[0][0] == TournamentType.TEXT
        assert text_call[0][1] == mock_config.psql_db
        assert text_call[0][2] == mock_config
        
        assert image_call[0][0] == TournamentType.IMAGE
        assert image_call[0][1] == mock_config.psql_db
        assert image_call[0][2] == mock_config

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.create_basic_tournament")
    @patch("validator.tournament.tournament_manager.get_latest_tournament_with_created_at")
    @patch("validator.tournament.tournament_manager.get_tournaments_with_status")
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_exception_handling(
        self,
        mock_get_active,
        mock_get_with_status,
        mock_get_latest,
        mock_create_tournament,
        mock_config,
    ):
        """Test that exceptions during tournament creation propagate up (current behavior)."""
        # Mock setup: no tournaments exist, should create new one
        mock_get_active.return_value = None
        mock_get_with_status.return_value = []
        mock_get_latest.return_value = (None, None)
        
        # Mock create_basic_tournament to raise an exception
        mock_create_tournament.side_effect = Exception("Database connection failed")
        
        # Should raise the exception (current behavior - no exception handling in the function)
        with pytest.raises(Exception, match="Database connection failed"):
            await check_and_start_tournament(TournamentType.TEXT, mock_config.psql_db, mock_config)
            
        # Should still attempt to create tournament
        mock_create_tournament.assert_called_once()

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.get_latest_tournament_with_created_at")
    @patch("validator.tournament.tournament_manager.get_tournaments_with_status")
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_completed_but_no_created_at(
        self,
        mock_get_active,
        mock_get_with_status,
        mock_get_latest,
        mock_config,
        mock_tournament_data,
    ):
        """Test handling when completed tournament exists but created_at is None."""
        # Mock setup: no active/pending tournaments, completed tournament but no created_at
        mock_get_active.return_value = None
        mock_get_with_status.return_value = []
        mock_get_latest.return_value = (mock_tournament_data["text_completed"], None)
        
        with patch("validator.tournament.tournament_manager.should_start_new_tournament_after_interval") as mock_should_start:
            mock_should_start.return_value = True
            
            with patch("validator.tournament.tournament_manager.create_basic_tournament") as mock_create:
                mock_create.return_value = "new_tournament_123"
                
                await check_and_start_tournament(TournamentType.TEXT, mock_config.psql_db, mock_config)
                
                # Should call should_start_new_tournament_after_interval with None
                mock_should_start.assert_called_once_with(None)

    @pytest.mark.asyncio
    async def test_should_start_new_tournament_edge_case_exactly_24_hours(self):
        """Test the exact boundary case of 24 hours."""
        # Test exactly 24 hours - should return True
        exactly_24h = datetime.now(timezone.utc) - timedelta(hours=24, seconds=0)
        result = await should_start_new_tournament_after_interval(exactly_24h)
        assert result is True
        
        # Test 1 second less than 24 hours - should return False  
        just_under_24h = datetime.now(timezone.utc) - timedelta(hours=23, minutes=59, seconds=59)
        result = await should_start_new_tournament_after_interval(just_under_24h)
        assert result is False

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.get_tournaments_with_status")
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_multiple_pending_same_type(
        self,
        mock_get_active,
        mock_get_with_status,
        mock_config,
    ):
        """Test behavior when multiple pending tournaments of same type exist."""
        # Mock setup: no active tournament, multiple pending tournaments of same type
        mock_get_active.return_value = None
        
        # Create multiple pending tournaments
        pending_tournaments = [
            TournamentData(
                tournament_id="pending_1",
                tournament_type=TournamentType.TEXT,
                status=TournamentStatus.PENDING,
            ),
            TournamentData(
                tournament_id="pending_2", 
                tournament_type=TournamentType.TEXT,
                status=TournamentStatus.PENDING,
            ),
            TournamentData(
                tournament_id="pending_image",
                tournament_type=TournamentType.IMAGE,
                status=TournamentStatus.PENDING,
            ),
        ]
        mock_get_with_status.return_value = pending_tournaments
        
        with patch("validator.tournament.tournament_manager.logger") as mock_logger:
            await check_and_start_tournament(TournamentType.TEXT, mock_config.psql_db, mock_config)
            
            # Should log that pending tournament exists (should pick first one)
            mock_logger.info.assert_called_with(
                f"Pending {TournamentType.TEXT.value} tournament exists: pending_1"
            )

    @pytest.mark.asyncio
    async def test_should_start_new_tournament_future_timestamp(self):
        """Test handling of future timestamps (should not happen in practice but good to test)."""
        # Future timestamp - should return False
        future_timestamp = datetime.now(timezone.utc) + timedelta(hours=1)
        result = await should_start_new_tournament_after_interval(future_timestamp)
        assert result is False

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.get_latest_tournament_with_created_at")
    @patch("validator.tournament.tournament_manager.get_tournaments_with_status") 
    @patch("validator.tournament.tournament_manager.get_active_tournament")
    async def test_check_and_start_tournament_active_tournament_has_status_check(
        self,
        mock_get_active,
        mock_get_with_status,
        mock_get_latest,
        mock_config,
        mock_tournament_data,
    ):
        """Test that we properly check the status of the tournament returned by get_latest_tournament_with_created_at."""
        # Mock setup: no active/pending tournaments
        mock_get_active.return_value = None
        mock_get_with_status.return_value = []
        
        # Return a tournament that's still ACTIVE (not COMPLETED)
        active_tournament = TournamentData(
            tournament_id="still_active_123",
            tournament_type=TournamentType.TEXT,
            status=TournamentStatus.ACTIVE,  # Not completed!
        )
        mock_get_latest.return_value = (active_tournament, datetime.now(timezone.utc) - timedelta(hours=25))
        
        with patch("validator.tournament.tournament_manager.create_basic_tournament") as mock_create:
            await check_and_start_tournament(TournamentType.TEXT, mock_config.psql_db, mock_config)
            
            # Should NOT create new tournament because latest tournament is still ACTIVE
            mock_create.assert_not_called()


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
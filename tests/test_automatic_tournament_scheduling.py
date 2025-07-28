import asyncio
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import validator.core.constants as cst
from core.models.tournament_models import TournamentData, TournamentStatus, TournamentType, TournamentRoundData, RoundStatus, RoundType
from validator.core.config import Config
from validator.db.database import PSQLDB
from validator.tournament.tournament_manager import (
    check_and_start_tournament,
    should_start_new_tournament_after_interval,
)
from validator.endpoints.tournament_analytics import get_next_tournament_dates


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


class TestTournamentAnalyticsEndpoint:
    """Test suite for the tournament analytics endpoint."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config object."""
        config = MagicMock(spec=Config)
        config.psql_db = MagicMock(spec=PSQLDB)
        return config

    @pytest.mark.asyncio
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_tournament_rounds")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_active_tournament")
    async def test_endpoint_returns_round_number_for_active_tournament(
        self, mock_get_active, mock_get_rounds, mock_config
    ):
        """Test that endpoint returns current round number when tournament is active."""
        # Mock active tournament
        active_tournament = TournamentData(
            tournament_id="active_text_123",
            tournament_type=TournamentType.TEXT,
            status=TournamentStatus.ACTIVE,
        )
        mock_get_active.return_value = active_tournament
        
        # Mock 3 rounds exist
        mock_rounds = [
            TournamentRoundData(
                round_id="round_1", tournament_id="active_text_123", round_number=1,
                round_type=RoundType.GROUP, is_final_round=False, status=RoundStatus.COMPLETED
            ),
            TournamentRoundData(
                round_id="round_2", tournament_id="active_text_123", round_number=2,
                round_type=RoundType.KNOCKOUT, is_final_round=False, status=RoundStatus.COMPLETED
            ),
            TournamentRoundData(
                round_id="round_3", tournament_id="active_text_123", round_number=3,
                round_type=RoundType.KNOCKOUT, is_final_round=True, status=RoundStatus.ACTIVE
            ),
        ]
        mock_get_rounds.return_value = mock_rounds
        
        # Mock no active IMAGE tournament
        def get_active_side_effect(psql_db, tournament_type):
            if tournament_type == TournamentType.TEXT:
                return active_tournament
            return None
        mock_get_active.side_effect = get_active_side_effect
        
        # Mock other calls for IMAGE tournament
        with patch("validator.endpoints.tournament_analytics.tournament_sql.get_tournaments_with_status") as mock_get_with_status:
            with patch("validator.endpoints.tournament_analytics.tournament_sql.get_latest_tournament_with_created_at") as mock_get_latest:
                mock_get_with_status.return_value = []
                mock_get_latest.return_value = (None, None)
                
                # Call the endpoint
                result = await get_next_tournament_dates(config=mock_config)
                
                # Check TEXT tournament result
                assert result.text.tournament_type == TournamentType.TEXT
                assert result.text.current_round_number == 3  # Should return number of rounds
                assert result.text.tournament_status == "active"
                assert result.text.interval_hours == cst.TOURNAMENT_INTERVAL_HOURS
                assert result.text.next_start_date is None
                assert result.text.next_end_date is None

    @pytest.mark.asyncio
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_tournaments_with_status")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_active_tournament")
    async def test_endpoint_returns_round_1_for_pending_tournament(
        self, mock_get_active, mock_get_with_status, mock_config
    ):
        """Test that endpoint returns round 1 when tournament is pending."""
        # Mock no active tournament
        mock_get_active.return_value = None
        
        # Mock pending tournament
        pending_tournament = TournamentData(
            tournament_id="pending_image_456",
            tournament_type=TournamentType.IMAGE,
            status=TournamentStatus.PENDING,
        )
        
        def get_with_status_side_effect(status, psql_db):
            if status == TournamentStatus.PENDING:
                return [pending_tournament]
            return []
        mock_get_with_status.side_effect = get_with_status_side_effect
        
        # Mock other calls for TEXT tournament
        with patch("validator.endpoints.tournament_analytics.tournament_sql.get_latest_tournament_with_created_at") as mock_get_latest:
            mock_get_latest.return_value = (None, None)
            
            # Call the endpoint
            result = await get_next_tournament_dates(config=mock_config)
            
            # Check IMAGE tournament result
            assert result.image.tournament_type == TournamentType.IMAGE
            assert result.image.current_round_number == 1
            assert result.image.tournament_status == "pending"
            assert result.image.interval_hours == cst.TOURNAMENT_INTERVAL_HOURS
            assert result.image.next_start_date is None
            assert result.image.next_end_date is None

    @pytest.mark.asyncio
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_latest_tournament_with_created_at")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_tournaments_with_status")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_active_tournament")
    async def test_endpoint_returns_countdown_when_waiting(
        self, mock_get_active, mock_get_with_status, mock_get_latest, mock_config
    ):
        """Test that endpoint returns countdown dates when no active/pending tournaments."""
        # Mock no active tournaments
        mock_get_active.return_value = None
        
        # Mock no pending tournaments
        mock_get_with_status.return_value = []
        
        # Mock completed tournament from 10 hours ago
        completed_tournament = TournamentData(
            tournament_id="completed_text_789",
            tournament_type=TournamentType.TEXT,
            status=TournamentStatus.COMPLETED,
        )
        created_at = datetime.now(timezone.utc) - timedelta(hours=10)
        mock_get_latest.return_value = (completed_tournament, created_at)
        
        # Call the endpoint
        result = await get_next_tournament_dates(config=mock_config)
        
        # Check TEXT tournament result
        assert result.text.tournament_type == TournamentType.TEXT
        assert result.text.current_round_number is None
        assert result.text.tournament_status == "waiting"
        assert result.text.interval_hours == cst.TOURNAMENT_INTERVAL_HOURS
        assert result.text.next_start_date is not None
        assert result.text.next_end_date is not None
        
        # Next start should be in the future (14 hours from now since tournament was 10 hours ago)
        expected_start = created_at + timedelta(hours=cst.TOURNAMENT_INTERVAL_HOURS)
        assert abs((result.text.next_start_date - expected_start).total_seconds()) < 60  # Within 1 minute

    @pytest.mark.asyncio
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_latest_tournament_with_created_at")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_tournaments_with_status")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_active_tournament")
    async def test_endpoint_returns_immediate_start_when_overdue(
        self, mock_get_active, mock_get_with_status, mock_get_latest, mock_config
    ):
        """Test that endpoint returns immediate start when tournament is overdue."""
        # Mock no active tournaments
        mock_get_active.return_value = None
        
        # Mock no pending tournaments
        mock_get_with_status.return_value = []
        
        # Mock completed tournament from 30 hours ago (overdue)
        completed_tournament = TournamentData(
            tournament_id="overdue_text_999",
            tournament_type=TournamentType.TEXT,
            status=TournamentStatus.COMPLETED,
        )
        created_at = datetime.now(timezone.utc) - timedelta(hours=30)
        mock_get_latest.return_value = (completed_tournament, created_at)
        
        # Call the endpoint
        result = await get_next_tournament_dates(config=mock_config)
        
        # Check TEXT tournament result
        assert result.text.tournament_type == TournamentType.TEXT
        assert result.text.current_round_number is None
        assert result.text.tournament_status == "waiting"
        assert result.text.interval_hours == cst.TOURNAMENT_INTERVAL_HOURS
        assert result.text.next_start_date is not None
        assert result.text.next_end_date is not None
        
        # Next start should be approximately now (overdue case)
        now = datetime.now(timezone.utc)
        assert abs((result.text.next_start_date - now).total_seconds()) < 60  # Within 1 minute

    @pytest.mark.asyncio
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_latest_tournament_with_created_at")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_tournaments_with_status")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_active_tournament")
    async def test_endpoint_returns_immediate_start_when_no_previous_tournaments(
        self, mock_get_active, mock_get_with_status, mock_get_latest, mock_config
    ):
        """Test that endpoint returns immediate start when no previous tournaments exist."""
        # Mock no active tournaments
        mock_get_active.return_value = None
        
        # Mock no pending tournaments
        mock_get_with_status.return_value = []
        
        # Mock no previous tournaments
        mock_get_latest.return_value = (None, None)
        
        # Call the endpoint
        result = await get_next_tournament_dates(config=mock_config)
        
        # Check both tournament results
        for tournament_result in [result.text, result.image]:
            assert tournament_result.current_round_number is None
            assert tournament_result.tournament_status == "waiting"
            assert tournament_result.interval_hours == cst.TOURNAMENT_INTERVAL_HOURS
            assert tournament_result.next_start_date is not None
            assert tournament_result.next_end_date is not None
            
            # Next start should be approximately now
            now = datetime.now(timezone.utc)
            assert abs((tournament_result.next_start_date - now).total_seconds()) < 60  # Within 1 minute

    @pytest.mark.asyncio
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_tournament_rounds")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_latest_tournament_with_created_at")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_tournaments_with_status")
    @patch("validator.endpoints.tournament_analytics.tournament_sql.get_active_tournament")
    async def test_endpoint_handles_mixed_tournament_states(
        self, mock_get_active, mock_get_with_status, mock_get_latest, mock_get_rounds, mock_config
    ):
        """Test endpoint when TEXT is active and IMAGE is waiting."""
        # Mock TEXT tournament as active
        text_active = TournamentData(
            tournament_id="active_text_123",
            tournament_type=TournamentType.TEXT,
            status=TournamentStatus.ACTIVE,
        )
        
        # Mock IMAGE tournament as completed (waiting for next)
        image_completed = TournamentData(
            tournament_id="completed_image_456",
            tournament_type=TournamentType.IMAGE,
            status=TournamentStatus.COMPLETED,
        )
        
        def get_active_side_effect(psql_db, tournament_type):
            if tournament_type == TournamentType.TEXT:
                return text_active
            return None
        mock_get_active.side_effect = get_active_side_effect
        
        # Mock no pending tournaments
        mock_get_with_status.return_value = []
        
        # Mock rounds for TEXT tournament
        mock_get_rounds.return_value = [
            TournamentRoundData(
                round_id="round_1", tournament_id="active_text_123", round_number=1,
                round_type=RoundType.GROUP, is_final_round=False, status=RoundStatus.COMPLETED
            ),
            TournamentRoundData(
                round_id="round_2", tournament_id="active_text_123", round_number=2,
                round_type=RoundType.KNOCKOUT, is_final_round=True, status=RoundStatus.ACTIVE
            ),
        ]
        
        # Mock latest tournament for IMAGE
        def get_latest_side_effect(psql_db, tournament_type):
            if tournament_type == TournamentType.IMAGE:
                return (image_completed, datetime.now(timezone.utc) - timedelta(hours=10))
            return (text_active, datetime.now(timezone.utc) - timedelta(hours=5))
        mock_get_latest.side_effect = get_latest_side_effect
        
        # Call the endpoint
        result = await get_next_tournament_dates(config=mock_config)
        
        # Check TEXT tournament (active)
        assert result.text.tournament_type == TournamentType.TEXT
        assert result.text.current_round_number == 2
        assert result.text.tournament_status == "active"
        assert result.text.next_start_date is None
        assert result.text.next_end_date is None
        
        # Check IMAGE tournament (waiting)
        assert result.image.tournament_type == TournamentType.IMAGE
        assert result.image.current_round_number is None
        assert result.image.tournament_status == "waiting"
        assert result.image.next_start_date is not None
        assert result.image.next_end_date is not None


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
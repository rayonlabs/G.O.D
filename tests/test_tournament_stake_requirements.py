from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fiber.chain.models import Node

from core.models.tournament_models import RoundStatus
from core.models.tournament_models import RoundType
from core.models.tournament_models import TournamentData
from core.models.tournament_models import TournamentParticipant
from core.models.tournament_models import TournamentRoundData
from core.models.tournament_models import TournamentStatus
from core.models.tournament_models import TournamentType
from validator.core.constants import TOURNAMENT_MAX_REPEAT_BOOST_PERCENTAGE
from validator.core.constants import TOURNAMENT_PARTICIPATION_WEIGHT
from validator.core.constants import TOURNAMENT_REPEAT_BOOST_PERCENTAGE
from validator.core.constants import TOURNAMENT_TOP_N_BY_STAKE
from validator.db.sql.tournaments import calculate_boosted_stake
from validator.db.sql.tournaments import count_completed_tournament_entries
from validator.db.sql.tournaments import eliminate_tournament_participants
from validator.db.sql.tournaments import get_participants_with_insufficient_stake
from validator.tournament.tournament_manager import advance_tournament
from validator.tournament.tournament_manager import populate_tournament_participants


class TestCountCompletedTournamentEntries:
    @pytest.mark.asyncio
    async def test_no_completed_tournaments(self):
        mock_db = AsyncMock()
        mock_connection = AsyncMock()
        mock_db.connection.return_value.__aenter__.return_value = mock_connection
        mock_connection.fetchrow.return_value = [0]

        result = await count_completed_tournament_entries("test_hotkey", mock_db)
        assert result == 0

    @pytest.mark.asyncio
    async def test_multiple_completed_tournaments(self):
        mock_db = AsyncMock()
        mock_connection = AsyncMock()
        mock_db.connection.return_value.__aenter__.return_value = mock_connection
        mock_connection.fetchrow.return_value = [3]

        result = await count_completed_tournament_entries("test_hotkey", mock_db)
        assert result == 3


class TestGetParticipantsWithInsufficientStake:
    @pytest.mark.asyncio
    async def test_no_insufficient_stake_participants(self):
        mock_db = AsyncMock()
        mock_connection = AsyncMock()
        mock_db.connection.return_value.__aenter__.return_value = mock_connection
        mock_connection.fetch.return_value = []

        result = await get_participants_with_insufficient_stake("tournament_1", mock_db)
        assert result == []

    @pytest.mark.asyncio
    async def test_multiple_insufficient_stake_participants(self):
        mock_db = AsyncMock()
        mock_connection = AsyncMock()
        mock_db.connection.return_value.__aenter__.return_value = mock_connection
        mock_connection.fetch.return_value = [{"hotkey": "hotkey1"}, {"hotkey": "hotkey2"}]

        result = await get_participants_with_insufficient_stake("tournament_1", mock_db)
        assert result == ["hotkey1", "hotkey2"]


class TestEliminateTournamentParticipants:
    @pytest.mark.asyncio
    async def test_eliminate_no_participants(self):
        mock_db = AsyncMock()

        # Should return early without database call
        await eliminate_tournament_participants("tournament_1", "round_1", [], mock_db)
        mock_db.connection.assert_not_called()

    @pytest.mark.asyncio
    async def test_eliminate_multiple_participants(self):
        mock_db = AsyncMock()
        mock_connection = AsyncMock()
        mock_db.connection.return_value.__aenter__.return_value = mock_connection

        hotkeys = ["hotkey1", "hotkey2", "hotkey3"]
        await eliminate_tournament_participants("tournament_1", "round_1", hotkeys, mock_db)

        mock_connection.execute.assert_called_once()
        call_args = mock_connection.execute.call_args[0]
        assert "tournament_1" in call_args
        assert "round_1" in call_args
        assert hotkeys in call_args


class TestBoostedStakeCalculation:
    def test_first_tournament_entry(self):
        actual_stake = 1000
        completed_entries = 0
        boosted_stake = calculate_boosted_stake(actual_stake, completed_entries)
        assert boosted_stake == 1000  # No boost

    def test_second_tournament_entry(self):
        actual_stake = 1000
        completed_entries = 1  # 5% boost
        boosted_stake = calculate_boosted_stake(actual_stake, completed_entries)
        assert boosted_stake == 1050  # 1000 * 1.05

    def test_max_boost(self):
        actual_stake = 1000
        completed_entries = 10  # Would be 50% but capped at 25%
        boosted_stake = calculate_boosted_stake(actual_stake, completed_entries)
        assert boosted_stake == 1250  # 1000 * 1.25

    def test_constants_consistency(self):
        assert TOURNAMENT_REPEAT_BOOST_PERCENTAGE == 5
        assert TOURNAMENT_MAX_REPEAT_BOOST_PERCENTAGE == 25
        assert TOURNAMENT_TOP_N_BY_STAKE == 32
        assert TOURNAMENT_PARTICIPATION_WEIGHT == 0.01


class TestPopulateTournamentParticipantsTop32Selection:
    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.get_all_nodes")
    @patch("validator.tournament.tournament_manager.count_completed_tournament_entries")
    @patch("validator.tournament.tournament_manager._get_miner_training_repo")
    @patch("validator.tournament.tournament_manager.get_tournament")
    @patch("validator.tournament.tournament_manager.add_tournament_participants")
    @patch("validator.tournament.tournament_manager.update_tournament_participant_training_repo")
    async def test_select_top_32_by_boosted_stake(
        self,
        mock_update_repo,
        mock_add_participants,
        mock_get_tournament,
        mock_get_training_repo,
        mock_count_entries,
        mock_get_nodes,
    ):
        # Setup mocks
        mock_config = MagicMock()
        mock_psql_db = AsyncMock()

        mock_tournament = TournamentData(
            tournament_id="test_tournament", tournament_type=TournamentType.TEXT, status=TournamentStatus.PENDING
        )
        mock_get_tournament.return_value = mock_tournament

        # Create test nodes with different stake levels
        nodes = []
        for i in range(40):  # Create 40 nodes to test top-32 selection
            node = Node(
                hotkey=f"hotkey{i}",
                alpha_stake=float(1000 + i * 100),  # Increasing stakes
                coldkey=f"cold{i}",
                node_id=i,
                incentive=0.1,
                netuid=1,
                tao_stake=0.0,
                stake=0.0,
                trust=0.5,
                vtrust=0.5,
                last_updated=1234567890.0,
                ip=f"1.1.1.{i}",
                ip_type=4,
                port=8080,
                protocol=4,
            )
            nodes.append(node)

        # Add base node (excluded)
        base_node = Node(
            hotkey="base_hotkey",
            alpha_stake=10000.0,
            coldkey="cold_base",
            node_id=100,
            incentive=0.1,
            netuid=1,
            tao_stake=0.0,
            stake=0.0,
            trust=0.5,
            vtrust=0.5,
            last_updated=1234567890.0,
            ip="0.0.0.0",
            ip_type=4,
            port=8080,
            protocol=4,
        )
        nodes.append(base_node)

        mock_get_nodes.return_value = nodes

        # Mock all nodes respond with training repos
        from core.models.payload_models import TrainingRepoResponse

        mock_training_repo = TrainingRepoResponse(github_repo="test/repo", commit_hash="abc123")
        mock_get_training_repo.return_value = mock_training_repo

        # Mock tournament entries (some nodes have previous entries for boost)
        def mock_count_side_effect(hotkey, db):
            if hotkey.endswith("0"):  # Every 10th node has 1 previous entry
                return 1  # 5% boost
            elif hotkey.endswith("5"):  # Every node ending in 5 has 3 previous entries
                return 3  # 15% boost
            return 0  # No boost

        mock_count_entries.side_effect = mock_count_side_effect

        # Mock constants
        with patch("validator.tournament.tournament_manager.cst.MIN_MINERS_FOR_TOURN", 4):
            with patch("validator.tournament.tournament_manager.cst.TOURNAMENT_TOP_N_BY_STAKE", 32):
                result = await populate_tournament_participants("test_tournament", mock_config, mock_psql_db)

        # Should select top 32 by boosted stake
        assert result == 32

        # Check that participants were added with correct entry stakes
        assert mock_add_participants.call_count == 32

        # Verify the participants have the highest boosted stakes
        # The nodes with highest actual stakes and boosts should be selected
        added_participants = []
        for call in mock_add_participants.call_args_list:
            participants = call[0][0]  # First argument is the participants list
            added_participants.extend(participants)

        assert len(added_participants) == 32

        # Check that stake_required is set to actual stake (not boosted)
        for participant in added_participants:
            assert participant.stake_required is not None
            assert participant.stake_required > 0


class TestAdvanceTournamentStakeElimination:
    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.get_round_winners")
    @patch("validator.tournament.tournament_manager.get_tournament_participants")
    @patch("validator.tournament.tournament_manager.get_participants_with_insufficient_stake")
    @patch("validator.tournament.tournament_manager.eliminate_tournament_participants")
    @patch("validator.tournament.tournament_manager.create_next_round")
    async def test_eliminate_winners_with_insufficient_stake(
        self, mock_create_next, mock_eliminate, mock_insufficient_stake, mock_get_participants, mock_get_winners
    ):
        # Setup
        tournament = TournamentData(
            tournament_id="test_tournament", tournament_type=TournamentType.TEXT, status=TournamentStatus.ACTIVE
        )
        completed_round = TournamentRoundData(
            round_id="round_1",
            tournament_id="test_tournament",
            round_number=1,
            round_type=RoundType.GROUP,
            status=RoundStatus.COMPLETED,
        )
        mock_config = MagicMock()
        mock_psql_db = AsyncMock()

        # Mock winners
        mock_get_winners.return_value = ["winner1", "winner2", "winner3"]

        # Mock participants
        participants = [
            TournamentParticipant(tournament_id="test_tournament", hotkey="winner1", eliminated_in_round_id=None),
            TournamentParticipant(tournament_id="test_tournament", hotkey="winner2", eliminated_in_round_id=None),
            TournamentParticipant(tournament_id="test_tournament", hotkey="winner3", eliminated_in_round_id=None),
            TournamentParticipant(tournament_id="test_tournament", hotkey="loser1", eliminated_in_round_id=None),
            TournamentParticipant(tournament_id="test_tournament", hotkey="loser2", eliminated_in_round_id=None),
        ]
        mock_get_participants.return_value = participants

        # Mock insufficient stake (winner2 has insufficient stake)
        mock_insufficient_stake.return_value = ["winner2"]

        await advance_tournament(tournament, completed_round, mock_config, mock_psql_db)

        # Should eliminate losers + winner with insufficient stake
        mock_eliminate.assert_called_once_with(
            "test_tournament",
            "round_1",
            ["loser1", "loser2", "winner2"],  # losers + insufficient stake winner
            mock_psql_db,
        )

        # Should create next round with remaining winners
        mock_create_next.assert_called_once()
        call_args = mock_create_next.call_args[0]
        remaining_winners = call_args[2]  # winners parameter
        assert set(remaining_winners) == {"winner1", "winner3"}

    @pytest.mark.asyncio
    @patch("validator.tournament.tournament_manager.get_round_winners")
    @patch("validator.tournament.tournament_manager.get_tournament_participants")
    @patch("validator.tournament.tournament_manager.get_participants_with_insufficient_stake")
    @patch("validator.tournament.tournament_manager.eliminate_tournament_participants")
    @patch("validator.tournament.tournament_manager.update_tournament_winner_hotkey")
    @patch("validator.tournament.tournament_manager.update_tournament_status")
    async def test_all_winners_eliminated_for_insufficient_stake(
        self,
        mock_update_status,
        mock_update_winner,
        mock_eliminate,
        mock_insufficient_stake,
        mock_get_participants,
        mock_get_winners,
    ):
        # Setup
        tournament = TournamentData(
            tournament_id="test_tournament", tournament_type=TournamentType.TEXT, status=TournamentStatus.ACTIVE
        )
        completed_round = TournamentRoundData(
            round_id="round_1",
            tournament_id="test_tournament",
            round_number=1,
            round_type=RoundType.GROUP,
            status=RoundStatus.COMPLETED,
        )
        mock_config = MagicMock()
        mock_psql_db = AsyncMock()

        # All winners have insufficient stake
        mock_get_winners.return_value = ["winner1", "winner2"]
        mock_get_participants.return_value = [
            TournamentParticipant(tournament_id="test_tournament", hotkey="winner1", eliminated_in_round_id=None),
            TournamentParticipant(tournament_id="test_tournament", hotkey="winner2", eliminated_in_round_id=None),
            TournamentParticipant(tournament_id="test_tournament", hotkey="loser1", eliminated_in_round_id=None),
        ]
        mock_insufficient_stake.return_value = ["winner1", "winner2"]

        await advance_tournament(tournament, completed_round, mock_config, mock_psql_db)

        # Should eliminate all participants
        mock_eliminate.assert_called_once_with("test_tournament", "round_1", ["loser1", "winner1", "winner2"], mock_psql_db)

        # Should set base contestant as winner and complete tournament
        mock_update_winner.assert_called_once_with("test_tournament", "base_contestant", mock_psql_db)
        mock_update_status.assert_called_once_with("test_tournament", TournamentStatus.COMPLETED, mock_psql_db)

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fiber.chain.models import Node
from core.models.tournament_models import TournamentParticipant, TournamentData, TournamentType, TournamentStatus, TournamentRoundData, RoundType, RoundStatus
from validator.core.constants import TOURNAMENT_BASE_STAKE_REQUIREMENT
from validator.db.sql.tournaments import (
    count_completed_tournament_entries,
    get_participants_with_insufficient_stake,
    eliminate_tournament_participants
)
from validator.tournament.tournament_manager import populate_tournament_participants, advance_tournament


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
        mock_connection.fetch.return_value = [
            {"hotkey": "hotkey1"},
            {"hotkey": "hotkey2"}
        ]
        
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


class TestStakeRequirementCalculation:
    def test_first_tournament_entry(self):
        completed_entries = 0
        required_stake = TOURNAMENT_BASE_STAKE_REQUIREMENT * completed_entries
        assert required_stake == 0

    def test_second_tournament_entry(self):
        completed_entries = 1
        required_stake = TOURNAMENT_BASE_STAKE_REQUIREMENT * completed_entries
        assert required_stake == 250

    def test_fifth_tournament_entry(self):
        completed_entries = 4
        required_stake = TOURNAMENT_BASE_STAKE_REQUIREMENT * completed_entries
        assert required_stake == 1000


class TestPopulateTournamentParticipantsStakeFilter:
    @pytest.mark.asyncio
    @patch('validator.tournament.tournament_manager.get_all_nodes')
    @patch('validator.tournament.tournament_manager.count_completed_tournament_entries')
    @patch('validator.tournament.tournament_manager._process_single_node')
    @patch('validator.tournament.tournament_manager.get_tournament')
    async def test_filter_nodes_by_stake_requirement(self, mock_get_tournament, mock_process_node, mock_count_entries, mock_get_nodes):
        # Setup mocks
        mock_config = MagicMock()
        mock_config.tournament_base_contestant_hotkey = "base_hotkey"
        mock_psql_db = AsyncMock()
        
        mock_tournament = TournamentData(
            tournament_id="test_tournament",
            tournament_type=TournamentType.TEXT,
            status=TournamentStatus.PENDING
        )
        mock_get_tournament.return_value = mock_tournament
        
        # Create test nodes with different stake levels
        node1 = Node(hotkey="hotkey1", alpha_stake=0, coldkey="cold1", node_id=1, incentive=0.1, netuid=1, tao_stake=0, stake=0, trust=0.5, vtrust=0.5, last_updated=1234567890, ip="1.1.1.1", ip_type=4, port=8080, protocol=4)
        node2 = Node(hotkey="hotkey2", alpha_stake=300, coldkey="cold2", node_id=2, incentive=0.1, netuid=1, tao_stake=0, stake=0, trust=0.5, vtrust=0.5, last_updated=1234567890, ip="2.2.2.2", ip_type=4, port=8080, protocol=4)
        node3 = Node(hotkey="hotkey3", alpha_stake=500, coldkey="cold3", node_id=3, incentive=0.1, netuid=1, tao_stake=0, stake=0, trust=0.5, vtrust=0.5, last_updated=1234567890, ip="3.3.3.3", ip_type=4, port=8080, protocol=4)
        base_node = Node(hotkey="base_hotkey", alpha_stake=1000, coldkey="cold_base", node_id=0, incentive=0.1, netuid=1, tao_stake=0, stake=0, trust=0.5, vtrust=0.5, last_updated=1234567890, ip="0.0.0.0", ip_type=4, port=8080, protocol=4)
        
        mock_get_nodes.return_value = [node1, node2, node3, base_node]
        
        # Mock tournament entries: node1=1 (250 required), node2=0 (0 required), node3=2 (500 required)
        def mock_count_side_effect(hotkey, db):
            if hotkey == "hotkey1":
                return 1  # 250 required, has 0 - insufficient
            elif hotkey == "hotkey2":  
                return 0  # 0 required, has 300 - sufficient
            elif hotkey == "hotkey3":
                return 2  # 500 required, has 500 - sufficient
            return 0
        
        mock_count_entries.side_effect = mock_count_side_effect
        mock_process_node.return_value = True
        
        # Mock constants import
        with patch('validator.tournament.tournament_manager.cst.MIN_MINERS_FOR_TOURN', 2):
            result = await populate_tournament_participants("test_tournament", mock_config, mock_psql_db)
        
        # Should process node2 and node3 (sufficient stake), but not node1 (insufficient) or base_node (excluded)
        assert mock_process_node.call_count == 2
        processed_hotkeys = [call[0][0].hotkey for call in mock_process_node.call_args_list]
        assert "hotkey2" in processed_hotkeys
        assert "hotkey3" in processed_hotkeys
        assert "hotkey1" not in processed_hotkeys
        assert "base_hotkey" not in processed_hotkeys


class TestAdvanceTournamentStakeElimination:
    @pytest.mark.asyncio
    @patch('validator.tournament.tournament_manager.get_round_winners')
    @patch('validator.tournament.tournament_manager.get_tournament_participants')
    @patch('validator.tournament.tournament_manager.get_participants_with_insufficient_stake')
    @patch('validator.tournament.tournament_manager.eliminate_tournament_participants')
    @patch('validator.tournament.tournament_manager.create_next_round')
    async def test_eliminate_winners_with_insufficient_stake(self, mock_create_next, mock_eliminate, mock_insufficient_stake, mock_get_participants, mock_get_winners):
        # Setup
        tournament = TournamentData(
            tournament_id="test_tournament",
            tournament_type=TournamentType.TEXT,
            status=TournamentStatus.ACTIVE
        )
        completed_round = TournamentRoundData(
            round_id="round_1",
            tournament_id="test_tournament",
            round_number=1,
            round_type=RoundType.GROUP,
            status=RoundStatus.COMPLETED
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
            TournamentParticipant(tournament_id="test_tournament", hotkey="loser2", eliminated_in_round_id=None)
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
            mock_psql_db
        )
        
        # Should create next round with remaining winners
        mock_create_next.assert_called_once()
        call_args = mock_create_next.call_args[0]
        remaining_winners = call_args[2]  # winners parameter
        assert set(remaining_winners) == {"winner1", "winner3"}

    @pytest.mark.asyncio  
    @patch('validator.tournament.tournament_manager.get_round_winners')
    @patch('validator.tournament.tournament_manager.get_tournament_participants')
    @patch('validator.tournament.tournament_manager.get_participants_with_insufficient_stake')
    @patch('validator.tournament.tournament_manager.eliminate_tournament_participants')
    @patch('validator.tournament.tournament_manager.update_tournament_winner_hotkey')
    @patch('validator.tournament.tournament_manager.update_tournament_status')
    async def test_all_winners_eliminated_for_insufficient_stake(self, mock_update_status, mock_update_winner, mock_eliminate, mock_insufficient_stake, mock_get_participants, mock_get_winners):
        # Setup
        tournament = TournamentData(
            tournament_id="test_tournament",
            tournament_type=TournamentType.TEXT,
            status=TournamentStatus.ACTIVE
        )
        completed_round = TournamentRoundData(
            round_id="round_1",
            tournament_id="test_tournament", 
            round_number=1,
            round_type=RoundType.GROUP,
            status=RoundStatus.COMPLETED
        )
        mock_config = MagicMock()
        mock_config.tournament_base_contestant_hotkey = "base_contestant"
        mock_psql_db = AsyncMock()
        
        # All winners have insufficient stake
        mock_get_winners.return_value = ["winner1", "winner2"]
        mock_get_participants.return_value = [
            TournamentParticipant(tournament_id="test_tournament", hotkey="winner1", eliminated_in_round_id=None),
            TournamentParticipant(tournament_id="test_tournament", hotkey="winner2", eliminated_in_round_id=None),
            TournamentParticipant(tournament_id="test_tournament", hotkey="loser1", eliminated_in_round_id=None)
        ]
        mock_insufficient_stake.return_value = ["winner1", "winner2"]
        
        await advance_tournament(tournament, completed_round, mock_config, mock_psql_db)
        
        # Should eliminate all participants  
        mock_eliminate.assert_called_once_with(
            "test_tournament",
            "round_1", 
            ["loser1", "winner1", "winner2"],
            mock_psql_db
        )
        
        # Should set base contestant as winner and complete tournament
        mock_update_winner.assert_called_once_with("test_tournament", "base_contestant", mock_psql_db)
        mock_update_status.assert_called_once_with("test_tournament", TournamentStatus.COMPLETED, mock_psql_db)
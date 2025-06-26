-- migrate:up

-- Core tournament
CREATE TABLE IF NOT EXISTS tournaments (
    tournament_id TEXT PRIMARY KEY, -- "tourn_abc123_20250101" (generated in application)
    tournament_type VARCHAR(50) NOT NULL, -- 'text', 'image'
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'active', 'completed', 'cancelled'
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Tournament rounds
CREATE TABLE IF NOT EXISTS tournament_rounds (
    round_id TEXT PRIMARY KEY, -- "tourn_abc123_round_001" (generated in application)
    tournament_id TEXT REFERENCES tournaments(tournament_id),
    round_number INTEGER NOT NULL,
    round_type VARCHAR(20) NOT NULL, -- 'group', 'knockout'
    is_final_round BOOLEAN DEFAULT FALSE,
    status VARCHAR(20) DEFAULT 'pending', -- 'pending', 'active', 'completed'
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- All participants (simple)
CREATE TABLE IF NOT EXISTS tournament_participants (
    tournament_id TEXT REFERENCES tournaments(tournament_id),
    hotkey TEXT NOT NULL,
    eliminated_in_round_id TEXT REFERENCES tournament_rounds(round_id),
    final_position INTEGER,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tournament_id, hotkey)
);

-- Groups
CREATE TABLE IF NOT EXISTS tournament_groups (
    group_id TEXT PRIMARY KEY, -- "tourn_abc123_round_001_group_001" (generated in application)
    round_id TEXT REFERENCES tournament_rounds(round_id),
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Group membership
CREATE TABLE IF NOT EXISTS tournament_group_members (
    group_id TEXT REFERENCES tournament_groups(group_id),
    hotkey TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (group_id, hotkey)
);

-- Pairs
CREATE TABLE IF NOT EXISTS tournament_pairs (
    pair_id TEXT PRIMARY KEY, -- "tourn_abc123_round_002_pair_001" (generated in application)
    round_id TEXT REFERENCES tournament_rounds(round_id),
    hotkey1 TEXT NOT NULL,
    hotkey2 TEXT NOT NULL,
    winner_hotkey TEXT,
    completed_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Tasks
CREATE TABLE IF NOT EXISTS tournament_tasks (
    tournament_id TEXT REFERENCES tournaments(tournament_id),
    round_id TEXT REFERENCES tournament_rounds(round_id),
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    group_id TEXT REFERENCES tournament_groups(group_id), -- NULL for knockout
    pair_id TEXT REFERENCES tournament_pairs(pair_id),     -- NULL for group
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (tournament_id, task_id),
    CHECK ((group_id IS NOT NULL AND pair_id IS NULL) OR (group_id IS NULL AND pair_id IS NOT NULL))
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tournaments_status ON tournaments(status);
CREATE INDEX IF NOT EXISTS idx_tournament_rounds_tournament_id ON tournament_rounds(tournament_id);
CREATE INDEX IF NOT EXISTS idx_tournament_rounds_status ON tournament_rounds(status);
CREATE INDEX IF NOT EXISTS idx_tournament_participants_hotkey ON tournament_participants(hotkey);
CREATE INDEX IF NOT EXISTS idx_tournament_group_members_hotkey ON tournament_group_members(hotkey);
CREATE INDEX IF NOT EXISTS idx_tournament_tasks_task_id ON tournament_tasks(task_id);

-- migrate:down
DROP TABLE IF EXISTS tournament_tasks;
DROP TABLE IF EXISTS tournament_pairs;
DROP TABLE IF EXISTS tournament_group_members;
DROP TABLE IF EXISTS tournament_groups;
DROP TABLE IF EXISTS tournament_participants;
DROP TABLE IF EXISTS tournament_rounds;
DROP TABLE IF EXISTS tournaments;

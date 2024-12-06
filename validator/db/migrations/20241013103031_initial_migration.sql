-- migrate:up

-- Tasks table (unchanged as it doesn't have coldkey/netuid)
CREATE TABLE IF NOT EXISTS tasks (
    task_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id TEXT NOT NULL,
    ds_id TEXT NOT NULL,
    system TEXT,
    instruction TEXT NOT NULL,
    input TEXT,
    output TEXT,
    status TEXT NOT NULL,
    test_data TEXT,
    synthetic_data TEXT,
    hf_training_repo TEXT,
    format TEXT,
    no_input_format TEXT,
    miner_scores FLOAT[],
    hours_to_complete INTEGER,
    end_timestamp TIMESTAMPTZ,
    user_id TEXT,
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    delay_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    started_timestamp TIMESTAMPTZ,
    completed_timestamp TIMESTAMPTZ
);

-- Nodes table with composite primary key
CREATE TABLE IF NOT EXISTS nodes (
    hotkey TEXT NOT NULL,
    netuid INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    coldkey TEXT NOT NULL,
    ip TEXT NOT NULL,
    ip_type INTEGER NOT NULL,
    port INTEGER NOT NULL,
    symmetric_key TEXT,
    symmetric_key_uuid TEXT,
    trust FLOAT,
    vtrust FLOAT,
    stake FLOAT NOT NULL,
    incentive FLOAT DEFAULT 0.0,
    last_updated FLOAT,
    protocol INTEGER DEFAULT 4,
    our_validator BOOLEAN DEFAULT FALSE,
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hotkey, netuid)
);

-- Create indexes for nodes table
CREATE INDEX IF NOT EXISTS idx_nodes_hotkey ON nodes(hotkey);
CREATE INDEX IF NOT EXISTS idx_nodes_netuid ON nodes(netuid);

-- Task nodes junction table
CREATE TABLE IF NOT EXISTS task_nodes (
    task_id UUID NOT NULL,
    hotkey TEXT NOT NULL,
    netuid INTEGER NOT NULL,
    quality_score FLOAT,
    PRIMARY KEY (task_id, hotkey, netuid)
);

-- Submissions table
CREATE TABLE IF NOT EXISTS submissions (
    submission_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id UUID NOT NULL,
    hotkey TEXT NOT NULL,
    netuid INTEGER NOT NULL,
    repo TEXT NOT NULL,
    created_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_on TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Nodes history table
CREATE TABLE IF NOT EXISTS nodes_history (
    id SERIAL NOT NULL,
    hotkey TEXT NOT NULL,
    netuid INTEGER NOT NULL,
    coldkey TEXT NOT NULL,
    node_id INTEGER NOT NULL,
    incentive FLOAT NOT NULL,
    stake FLOAT NOT NULL,
    trust FLOAT NOT NULL,
    vtrust FLOAT NOT NULL,
    last_updated FLOAT,
    ip TEXT NOT NULL,
    ip_type INTEGER NOT NULL,
    port INTEGER NOT NULL,
    protocol INTEGER NOT NULL DEFAULT 4,
    our_validator BOOLEAN NOT NULL DEFAULT FALSE,
    expired_at TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'UTC'),
    PRIMARY KEY (id, hotkey, netuid)
);

-- migrate:down
DROP TABLE IF EXISTS submissions;
DROP TABLE IF EXISTS task_nodes;
DROP TABLE IF EXISTS nodes_history;
DROP TABLE IF EXISTS nodes;
DROP TABLE IF EXISTS tasks;

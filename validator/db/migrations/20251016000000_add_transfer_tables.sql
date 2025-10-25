-- migrate:up

-- Transfers table to store TaoStats transfer data
CREATE TABLE IF NOT EXISTS transfers (
    id TEXT PRIMARY KEY, -- Transfer ID from TaoStats API
    to_ss58 TEXT NOT NULL, -- Recipient SS58 address
    to_hex TEXT NOT NULL, -- Recipient hex address
    from_ss58 TEXT NOT NULL, -- Sender SS58 address
    from_hex TEXT NOT NULL, -- Sender hex address
    network TEXT NOT NULL, -- Network name (e.g., 'finney')
    block_number INTEGER NOT NULL, -- Block number
    timestamp TIMESTAMPTZ NOT NULL, -- Transfer timestamp
    amount_rao NUMERIC(30, 0) NOT NULL, -- Transfer amount in RAO (with full precision)
    fee_rao NUMERIC(30, 0) NOT NULL, -- Transaction fee in RAO (with full precision)
    transaction_hash TEXT NOT NULL, -- Transaction hash
    extrinsic_id TEXT NOT NULL, -- Extrinsic ID
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, -- When we first saw this transfer
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP -- When we last updated this record
);

-- Coldkey balances table to track RAO balances and transfers to our target address
CREATE TABLE IF NOT EXISTS coldkey_balances (
    coldkey TEXT PRIMARY KEY, -- Coldkey SS58 address
    balance_rao NUMERIC(30, 0) NOT NULL DEFAULT 0, -- Current RAO balance (with full precision)
    total_sent_rao NUMERIC(30, 0) NOT NULL DEFAULT 0, -- Total RAO sent to our target address
    transfer_count INTEGER NOT NULL DEFAULT 0, -- Number of transfers sent to our target address
    last_transfer_at TIMESTAMPTZ, -- Timestamp of last transfer to our target address
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP, -- When we first saw this coldkey
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP -- When we last updated this record
);

-- Transfer processing state table to track the last processed timestamp
CREATE TABLE IF NOT EXISTS transfer_processing_state (
    id INTEGER PRIMARY KEY DEFAULT 1, -- Single row table
    last_processed_timestamp TIMESTAMPTZ NOT NULL, -- Last timestamp we processed
    last_processed_block INTEGER NOT NULL, -- Last block number we processed
    processing_interval_hours INTEGER NOT NULL DEFAULT 24, -- How often to check for new transfers
    target_address TEXT NOT NULL, -- The address we're monitoring for transfers
    network TEXT NOT NULL DEFAULT 'finney', -- Network we're monitoring
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT single_row CHECK (id = 1)
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_transfers_to_ss58 ON transfers(to_ss58);
CREATE INDEX IF NOT EXISTS idx_transfers_from_ss58 ON transfers(from_ss58);
CREATE INDEX IF NOT EXISTS idx_transfers_timestamp ON transfers(timestamp);
CREATE INDEX IF NOT EXISTS idx_transfers_block_number ON transfers(block_number);
CREATE INDEX IF NOT EXISTS idx_transfers_network ON transfers(network);
CREATE INDEX IF NOT EXISTS idx_transfers_transaction_hash ON transfers(transaction_hash);

CREATE INDEX IF NOT EXISTS idx_coldkey_balances_balance ON coldkey_balances(balance_rao);
CREATE INDEX IF NOT EXISTS idx_coldkey_balances_total_sent ON coldkey_balances(total_sent_rao);
CREATE INDEX IF NOT EXISTS idx_coldkey_balances_last_transfer ON coldkey_balances(last_transfer_at);

-- Insert default processing state (will be updated by the application)
INSERT INTO transfer_processing_state (
    last_processed_timestamp,
    last_processed_block,
    processing_interval_hours,
    target_address,
    network
) VALUES (
    NOW() - INTERVAL '1 day', -- Default to 1 day ago
    0, -- Default to block 0
    24, -- Check every 24 hours by default
    '', -- Will be set by environment variable
    'finney'
) ON CONFLICT (id) DO NOTHING;

-- migrate:down
DROP TABLE IF EXISTS transfer_processing_state;
DROP TABLE IF EXISTS coldkey_balances;
DROP TABLE IF EXISTS transfers;

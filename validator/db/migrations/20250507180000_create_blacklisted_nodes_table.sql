-- migrate:up
-- Create blacklisted_nodes table
CREATE TABLE IF NOT EXISTS blacklisted_nodes (
    hotkey TEXT NOT NULL,
    netuid INTEGER NOT NULL,
    created_timestamp TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (hotkey, netuid)
);

-- Migrate existing blacklisted nodes to the new table
INSERT INTO blacklisted_nodes (hotkey, netuid)
SELECT hotkey, netuid FROM nodes
WHERE is_blacklisted = TRUE;

-- Remove is_blacklisted column from nodes table
ALTER TABLE nodes DROP COLUMN is_blacklisted;

-- migrate:down
-- Add is_blacklisted column back to nodes table
ALTER TABLE nodes ADD COLUMN is_blacklisted BOOLEAN DEFAULT FALSE;

-- Restore blacklisted status from blacklisted_nodes table
UPDATE nodes
SET is_blacklisted = TRUE
FROM blacklisted_nodes
WHERE nodes.hotkey = blacklisted_nodes.hotkey
AND nodes.netuid = blacklisted_nodes.netuid;

-- Drop blacklisted_nodes table
DROP TABLE IF EXISTS blacklisted_nodes;
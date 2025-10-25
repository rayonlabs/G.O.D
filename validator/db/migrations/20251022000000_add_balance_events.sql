-- migrate:up

-- Add balance_events table to track all balance changes tied to tournaments
CREATE TABLE balance_events (
    id SERIAL PRIMARY KEY,
    tournament_id VARCHAR(255) NOT NULL,
    coldkey VARCHAR(255) NOT NULL,
    event_type VARCHAR(50) NOT NULL, -- 'participation_fee', 'refund', 'transfer_in'
    amount_rao BIGINT NOT NULL, -- positive for credits, negative for debits
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Add indexes for efficient queries
CREATE INDEX idx_balance_events_tournament_id ON balance_events(tournament_id);
CREATE INDEX idx_balance_events_coldkey ON balance_events(coldkey);
CREATE INDEX idx_balance_events_event_type ON balance_events(event_type);
CREATE INDEX idx_balance_events_created_at ON balance_events(created_at);

-- Add foreign key constraint to tournaments table (if it exists)
-- Note: This assumes tournaments table exists with tournament_id as primary key
-- ALTER TABLE balance_events ADD CONSTRAINT fk_balance_events_tournament_id
--     FOREIGN KEY (tournament_id) REFERENCES tournaments(tournament_id);

-- Add comment for documentation
COMMENT ON TABLE balance_events IS 'Tracks all balance changes tied to tournaments for audit and refund purposes';
COMMENT ON COLUMN balance_events.tournament_id IS 'Tournament ID this event is associated with';
COMMENT ON COLUMN balance_events.coldkey IS 'Coldkey address affected by this event';
COMMENT ON COLUMN balance_events.event_type IS 'Type of balance event: participation_fee, refund, transfer_in';
COMMENT ON COLUMN balance_events.amount_rao IS 'Amount in RAO (positive for credits, negative for debits)';
COMMENT ON COLUMN balance_events.description IS 'Human-readable description of the event';

-- migrate:down
DROP TABLE IF EXISTS balance_events;

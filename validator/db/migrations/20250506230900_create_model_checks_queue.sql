-- migrate:up
CREATE TABLE model_checks_queue (
    id UUID PRIMARY KEY,
    model_id TEXT NOT NULL,
    status TEXT NOT NULL, -- Stores enum values like 'PENDING', 'PROCESSING', 'SUCCESS', 'FAILURE'
    requested_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    parameter_count BIGINT,
    error_message TEXT
);

CREATE INDEX idx_model_checks_queue_status_requested_at ON model_checks_queue (status, requested_at ASC);

-- migrate:down
DROP INDEX IF EXISTS idx_model_checks_queue_model_id_status;
DROP INDEX IF EXISTS idx_model_checks_queue_status_requested_at;
DROP TABLE IF EXISTS model_checks_queue; 
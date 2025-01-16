-- migrate:up

CREATE TABLE IF NOT EXISTS offer_responses (
    id SERIAL PRIMARY KEY,
    task_id UUID NOT NULL,
    hotkey VARCHAR(66) NOT NULL,
    offer_response JSONB NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- migrate:down

DROP TABLE IF EXISTS offer_responses;

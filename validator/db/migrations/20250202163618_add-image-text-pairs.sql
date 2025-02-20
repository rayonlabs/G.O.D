-- migrate:up
CREATE TABLE IF NOT EXISTS image_text_pairs (
    id SERIAL PRIMARY KEY,
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    text_url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_image_text_pairs_task_id ON image_text_pairs(task_id);

-- migrate:down
DROP TABLE IF EXISTS image_text_pairs;

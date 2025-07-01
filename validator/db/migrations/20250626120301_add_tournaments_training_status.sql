-- migrate:up

-- Tournament task hotkey trainings
CREATE TABLE IF NOT EXISTS tournament_task_hotkey_trainings (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL,
    is_trained BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tournament_task_hotkey_trainings_is_trained ON tournament_task_hotkey_trainings(is_trained);

-- migrate:down
DROP TABLE IF EXISTS tournament_task_hotkey_trainings;

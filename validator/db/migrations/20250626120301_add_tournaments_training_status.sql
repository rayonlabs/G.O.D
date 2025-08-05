-- migrate:up

-- Tournament task hotkey trainings
CREATE TABLE IF NOT EXISTS tournament_task_hotkey_trainings (
    task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    hotkey TEXT NOT NULL,
    training_status TEXT DEFAULT 'pending',
    n_training_attempts INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (task_id, hotkey)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_tournament_task_hotkey_trainings_training_status ON tournament_task_hotkey_trainings(training_status);

-- migrate:down
DROP TABLE IF EXISTS tournament_task_hotkey_trainings;

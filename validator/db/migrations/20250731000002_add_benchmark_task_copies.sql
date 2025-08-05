-- migrate:up

CREATE TABLE IF NOT EXISTS benchmark_task_copies (
    copy_task_id UUID PRIMARY KEY REFERENCES tasks(task_id) ON DELETE CASCADE,
    root_task_id UUID NOT NULL REFERENCES benchmark_root_tasks(task_id) ON DELETE CASCADE,
    participant_hotkey TEXT NOT NULL,
    tournament_id TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(root_task_id, participant_hotkey)
);

CREATE INDEX IF NOT EXISTS idx_benchmark_task_copies_root_task_id ON benchmark_task_copies(root_task_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_task_copies_participant_hotkey ON benchmark_task_copies(participant_hotkey);
CREATE INDEX IF NOT EXISTS idx_benchmark_task_copies_tournament_id ON benchmark_task_copies(tournament_id);

-- migrate:down
DROP TABLE IF EXISTS benchmark_task_copies; 
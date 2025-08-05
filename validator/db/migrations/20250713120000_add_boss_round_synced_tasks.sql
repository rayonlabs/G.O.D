-- migrate:up

-- Boss round synced tasks linking table
CREATE TABLE IF NOT EXISTS boss_round_synced_tasks (
    tournament_task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    general_task_id UUID NOT NULL REFERENCES tasks(task_id) ON DELETE CASCADE,
    PRIMARY KEY (tournament_task_id, general_task_id)
);

-- migrate:down
DROP TABLE IF EXISTS boss_round_synced_tasks;
-- migrate:up

CREATE INDEX IF NOT EXISTS idx_task_nodes_hotkey_netuid ON task_nodes(hotkey, netuid);
CREATE INDEX IF NOT EXISTS idx_task_nodes_quality_score ON task_nodes(quality_score);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON tasks(created_at);

-- migrate:down

DROP INDEX IF EXISTS idx_task_nodes_hotkey_netuid;
DROP INDEX IF EXISTS idx_task_nodes_quality_score;
DROP INDEX IF EXISTS idx_tasks_created_at;

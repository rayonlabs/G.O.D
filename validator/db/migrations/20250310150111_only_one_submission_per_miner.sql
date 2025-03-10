-- migrate:up

-- delete dangling submissions
DELETE FROM submissions
WHERE task_id NOT IN (SELECT task_id FROM tasks);

-- clean up duplicates
WITH ranked_submissions AS (
    SELECT 
        *,
        ROW_NUMBER() OVER (
            PARTITION BY task_id, hotkey, netuid 
            ORDER BY created_on DESC
        ) as rn
    FROM submissions
)
DELETE FROM submissions 
WHERE submission_id IN (
    SELECT submission_id 
    FROM ranked_submissions 
    WHERE rn > 1
);

-- add foreign key constraint
ALTER TABLE submissions
ADD CONSTRAINT fk_submissions_task
FOREIGN KEY (task_id) 
REFERENCES tasks(task_id)
ON DELETE CASCADE;

-- add unique constraint
ALTER TABLE submissions 
ADD CONSTRAINT unique_submission_per_miner_task 
UNIQUE (task_id, hotkey, netuid);

-- migrate:down
ALTER TABLE submissions 
DROP CONSTRAINT unique_submission_per_miner_task;

ALTER TABLE submissions
DROP CONSTRAINT fk_submissions_task;


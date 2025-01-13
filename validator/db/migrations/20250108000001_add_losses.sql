-- migrate:up
ALTER TABLE task_nodes
ADD COLUMN test_loss FLOAT,
ADD COLUMN synth_loss FLOAT;

-- migrate:down
ALTER TABLE task_nodes
DROP COLUMN test_loss,
DROP COLUMN synth_loss;

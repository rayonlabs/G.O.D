-- migrate:up

ALTER TABLE tournaments 
ADD COLUMN winning_performance_difference DOUBLE PRECISION;

-- migrate:down

ALTER TABLE tournaments 
DROP COLUMN IF EXISTS winning_performance_difference;


-- migrate:up

CREATE TABLE IF NOT EXISTS trainers_gpus (
    trainer_ip TEXT NOT NULL,
    gpu_id INTEGER NOT NULL,
    gpu_type VARCHAR(100) NOT NULL,
    vram_gb INTEGER NOT NULL,
    used_until TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (trainer_ip, gpu_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trainers_gpus_gpu_type ON trainers_gpus(gpu_type);
CREATE INDEX IF NOT EXISTS idx_trainers_gpus_used_until ON trainers_gpus(used_until);

-- migrate:down
DROP TABLE IF EXISTS trainers_gpus;

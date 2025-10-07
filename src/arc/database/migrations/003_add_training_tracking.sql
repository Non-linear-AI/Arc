-- Migration: Add training tracking tables
-- Description: Tables for tracking training runs, metrics, and checkpoints

-- Training runs table
CREATE TABLE IF NOT EXISTS training_runs (
    run_id VARCHAR PRIMARY KEY,
    job_id VARCHAR,
    model_id VARCHAR,
    trainer_id VARCHAR,

    -- Run metadata
    run_name VARCHAR,
    description TEXT,

    -- Tracking configuration
    tensorboard_enabled BOOLEAN DEFAULT TRUE,
    tensorboard_log_dir VARCHAR,
    metric_log_frequency INTEGER DEFAULT 100,
    checkpoint_frequency INTEGER DEFAULT 5,

    -- Status tracking
    status VARCHAR DEFAULT 'pending',  -- pending, running, paused, stopped, completed, failed
    started_at TIMESTAMP,
    paused_at TIMESTAMP,
    resumed_at TIMESTAMP,
    completed_at TIMESTAMP,

    -- Configuration snapshots
    original_config JSON,
    current_config JSON,
    config_history JSON,

    -- Metrics
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Training metrics table
CREATE TABLE IF NOT EXISTS training_metrics (
    metric_id VARCHAR PRIMARY KEY,
    run_id VARCHAR,

    -- Metric identification
    metric_name VARCHAR,
    metric_type VARCHAR,  -- train, validation, test

    -- Value tracking
    step INTEGER,
    epoch INTEGER,
    value DOUBLE,

    -- Timestamp
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
);

-- Create index for efficient metric queries
CREATE INDEX IF NOT EXISTS idx_training_metrics_run_metric
ON training_metrics(run_id, metric_name, step);

-- Training checkpoints table
CREATE TABLE IF NOT EXISTS training_checkpoints (
    checkpoint_id VARCHAR PRIMARY KEY,
    run_id VARCHAR,

    -- Checkpoint metadata
    epoch INTEGER,
    step INTEGER,
    checkpoint_path VARCHAR,

    -- Performance snapshot
    metrics JSON,
    is_best BOOLEAN DEFAULT FALSE,

    -- File info
    file_size_bytes BIGINT,
    status VARCHAR DEFAULT 'saved',  -- saved, deleted, corrupted

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (run_id) REFERENCES training_runs(run_id) ON DELETE CASCADE
);

-- Create index for checkpoint queries
CREATE INDEX IF NOT EXISTS idx_training_checkpoints_run
ON training_checkpoints(run_id, epoch);

IF OBJECT_ID(N'dbo.training_jobs', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.training_jobs (
        job_id UNIQUEIDENTIFIER NOT NULL,
        equipment_code NVARCHAR(128) NOT NULL,
        meas_code NVARCHAR(128) NOT NULL,
        model_info_id NVARCHAR(128) NOT NULL,
        model_type NVARCHAR(32) NOT NULL,
        plan_json NVARCHAR(MAX) NOT NULL,
        plan_hash CHAR(64) NOT NULL,
        status VARCHAR(16) NOT NULL,
        error_code VARCHAR(64) NULL,
        best_val FLOAT NULL,
        test_loss FLOAT NULL,
        created_at DATETIME2(3) NOT NULL
            CONSTRAINT DF_training_jobs_created_at DEFAULT SYSUTCDATETIME(),
        started_at DATETIME2(3) NULL,
        finished_at DATETIME2(3) NULL,
        CONSTRAINT PK_training_jobs PRIMARY KEY (job_id),
        CONSTRAINT UQ_training_jobs_model_info
            UNIQUE (equipment_code, meas_code, model_info_id),
        CONSTRAINT CK_training_jobs_status
            CHECK (status IN ('QUEUED', 'RUNNING', 'SUCCEEDED', 'FAILED'))
    );

    CREATE INDEX IX_training_jobs_status_created_at
        ON dbo.training_jobs (status, created_at);
END;

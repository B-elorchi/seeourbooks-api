module.exports = {
  apps: [
    {
      name: "seeourbook-api",
      script: "./venv/bin/uvicorn", 
      args: "api.main:app --host 0.0.0.0 --port 8000",
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_memory_restart: "1G",
      env: {
        NODE_ENV: "production",
      }
    },
    {
      name: "seeourbook-worker",
      script: "./venv/bin/celery",
      args: "-A api.worker worker --loglevel=info --concurrency=4",
      interpreter: "none",
      autorestart: true,
      watch: false,
      max_memory_restart: "2G",
      env: {
        NODE_ENV: "production",
      }
    }
  ]
};
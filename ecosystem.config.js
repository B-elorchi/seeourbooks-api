module.exports = {
	apps: [
		{
			name: "seeourbooks-api",
			script: "/usr/bin/uvicorn",
			args: "api.main:app --host 0.0.0.0 --port 8000",
			cwd: "/home/seeourbooks/seeourbooks-api",
			interpreter: "none",
			autorestart: true,
			watch: false,
			max_memory_restart: "1G",
			env: {
				PYTHONPATH: ".",
				NODE_ENV: "production",
			},
		},
		{
			name: "seeourbooks-worker",
			script: "/usr/local/bin/celery",
			args: "-A api.celery_app worker --loglevel=info --concurrency=4",
			cwd: "/home/seeourbooks/seeourbooks-api",
			interpreter: "none",
			autorestart: true,
			watch: false,
			max_memory_restart: "2G",
			env: {
				PYTHONPATH: ".",
				NODE_ENV: "production",
			},
		},
	],
};

# Simple doctor checks for the local AI agent environment
Write-Host "Running environment checks..."
py -c "import sys,shutil,psutil; print('Python', sys.version.split()[0]); print('yt-dlp', shutil.which('yt-dlp')); print('ffmpeg', shutil.which('ffmpeg')); print('nvidia-smi', shutil.which('nvidia-smi')); print('vulkaninfo', shutil.which('vulkaninfo')); import src.runtime as r; print('Runtime report:'); import json; print(json.dumps(r.recommendation_report(), indent=2))"

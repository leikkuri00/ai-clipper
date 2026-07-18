# Verify installed components and quick smoke tests
py -c "import sys,shutil; print('Python', sys.version); print('psutil', __import__('psutil').__version__); print('rich', __import__('rich').__version__); print('yt-dlp', shutil.which('yt-dlp'))"

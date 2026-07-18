# Simple benchmark script to test CPU throughput for model inference (proxy)
Write-Host "Running a quick CPU benchmark (token/sec proxy)..."
py -c "import time,random; t0=time.time(); total=0
for _ in range(100000):
    total += random.random()
print('Elapsed', time.time()-t0)
"

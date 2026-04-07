# knod - self-learning knowledge network

set windows-shell := ["pwsh", "-NoProfile", "-Command"]

bin_dir := "bin"

# Build optimized executable into bin/
[windows]
build:
	@if (-not (Test-Path {{bin_dir}})) { New-Item -ItemType Directory {{bin_dir}} | Out-Null }
	odin build src -o:speed -out:{{bin_dir}}/knod.exe

[linux]
build:
	mkdir -p {{bin_dir}}
	odin build src -o:speed -out:{{bin_dir}}/knod

# Build with debug info
[windows]
debug:
	@if (-not (Test-Path {{bin_dir}})) { New-Item -ItemType Directory {{bin_dir}} | Out-Null }
	odin build src -debug -out:{{bin_dir}}/knod.exe

[linux]
debug:
	mkdir -p {{bin_dir}}
	odin build src -debug -out:{{bin_dir}}/knod

# Run tests (finds all packages containing @test.odin)
# ingest tests use -define:ODIN_TEST_THREADS=1 due to shared global config state.
[windows]
test:
	@Get-ChildItem -Path src -Filter '@test.odin' -Recurse | ForEach-Object { $threads = ''; if ($_.DirectoryName -match 'ingest$') { $threads = '-define:ODIN_TEST_THREADS=1' }; $cmd = "odin test $($_.DirectoryName) $threads"; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } }

[linux]
test:
	@for dir in $(find src -name '@test.odin' -exec dirname {} \;); do threads=''; if echo "$$dir" | grep -q 'ingest$$'; then threads='-define:ODIN_TEST_THREADS=1'; fi; odin test "$$dir" $$threads || exit 1; done

# Run the built executable
[windows]
run: build
	./{{bin_dir}}/knod.exe

[linux]
run: build
	./{{bin_dir}}/knod

# Start ingest node in background (logger writes to bin/logs/)
[windows]
ingest: build
	@Start-Process -FilePath "./{{bin_dir}}/knod.exe" -WorkingDirectory "{{bin_dir}}" -WindowStyle Hidden; Write-Output "knod started (port {{INGEST_PORT}}, logs: {{bin_dir}}/logs/)"

[linux]
ingest: build
	nohup ./{{bin_dir}}/knod > /dev/null 2>&1 & echo "knod started (pid $!, port {{INGEST_PORT}}, logs: {{bin_dir}}/logs/)"

# View the knod logger
[windows]
log:
	@Get-Content {{bin_dir}}/logs/knod.log -ErrorAction SilentlyContinue

[linux]
log:
	@cat {{bin_dir}}/logs/knod.log 2>/dev/null || echo "no log file"

INGEST_PORT := "7999"
APP_PORT    := "3000"

# Send text to a running ingest node
[windows]
send text:
	@$c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', {{INGEST_PORT}}); $s = $c.GetStream(); $b = [System.Text.Encoding]::UTF8.GetBytes('{{text}}'); $s.Write($b, 0, $b.Length); $s.Close(); $c.Close(); Write-Output "sent $($b.Length) bytes"

[linux]
send text:
	echo "{{text}}" | nc 127.0.0.1 {{INGEST_PORT}} && echo "sent"

# Ask a question to a running knod node (returns generated text)
[windows]
ask question:
	@$c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', {{INGEST_PORT}}); $s = $c.GetStream(); $b = [System.Text.Encoding]::UTF8.GetBytes('ASK:{{question}}'); $s.Write($b, 0, $b.Length); $c.Client.Shutdown([System.Net.Sockets.SocketShutdown]::Send); $buf = New-Object byte[] 4096; $resp = ''; while (($n = $s.Read($buf, 0, $buf.Length)) -gt 0) { $resp += [System.Text.Encoding]::UTF8.GetString($buf, 0, $n) }; $s.Close(); $c.Close(); Write-Output $resp

[linux]
ask question:
	echo -n "ASK:{{question}}" | nc 127.0.0.1 {{INGEST_PORT}}

# Send a file to a running ingest node
[windows]
send-file path:
	@$c = New-Object System.Net.Sockets.TcpClient('127.0.0.1', {{INGEST_PORT}}); $s = $c.GetStream(); $b = [System.IO.File]::ReadAllBytes('{{path}}'); $s.Write($b, 0, $b.Length); $s.Close(); $c.Close(); Write-Output "sent $($b.Length) bytes from {{path}}"

[linux]
send-file path:
	nc 127.0.0.1 {{INGEST_PORT}} < "{{path}}" && echo "sent {{path}}"

# Stop the ingest node
[windows]
stop:
	@Get-Process knod -ErrorAction SilentlyContinue | Stop-Process -Force; Write-Output "knod stopped"

[linux]
stop:
	pkill -f "knod ingest" && echo "knod stopped" || echo "no knod process found"

# Show status of running knod
[windows]
status:
	@$p = Get-Process knod -ErrorAction SilentlyContinue; if ($p) { Write-Output "knod running (pid $($p.Id))"; netstat -an | Select-String "{{INGEST_PORT}}" } else { Write-Output "knod not running" }

[linux]
status:
	@pgrep -f "knod ingest" > /dev/null && echo "knod running (pid $(pgrep -f 'knod ingest'))" || echo "knod not running"

# Serve the app/ directory on APP_PORT (default 3000)
[windows]
serve:
	@Start-Process -FilePath "python" -ArgumentList "-m http.server {{APP_PORT}} --directory app" -WindowStyle Hidden; Write-Output "app server started on http://localhost:{{APP_PORT}}"

[linux]
serve:
	nohup python3 -m http.server {{APP_PORT}} --directory app > /dev/null 2>&1 & echo "app server started on http://localhost:{{APP_PORT}}"

# Stop the app static file server
[windows]
stop-serve:
	@Get-NetTCPConnection -LocalPort {{APP_PORT}} -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }; Write-Output "app server stopped"

[linux]
stop-serve:
	pkill -f "http.server {{APP_PORT}}" && echo "app server stopped" || echo "app server not running"

# Start both knod and the app file server (knod on :8080, app on :{{APP_PORT}})
[windows]
dev: ingest serve
	@Write-Output "knod: http://localhost:8080  |  app: http://localhost:{{APP_PORT}}/explore.html"

[linux]
dev: ingest serve
	@echo "knod: http://localhost:8080  |  app: http://localhost:{{APP_PORT}}/explore.html"

# Run Python unit tests (pytest)
[windows]
pytest:
	python -m pytest tests/test_expand.py -v

[linux]
pytest:
	python3 -m pytest tests/test_expand.py -v

# Run full integration test (requires OpenAI API key)
[windows]
integration:
	python tests/integration.py

[linux]
integration:
	python3 tests/integration.py

# Format all Python files with ruff (tabs, width 2)
fmt:
	ruff format knod tests

# Check formatting without changing files
fmt-check:
	ruff format --check knod tests

# Clean build artifacts
[windows]
clean:
	@if (Test-Path {{bin_dir}}) { Remove-Item -Recurse -Force {{bin_dir}} }

[linux]
clean:
	rm -rf {{bin_dir}}

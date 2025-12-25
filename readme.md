# REVART HOCKEY SCOREBOARD Integrity Benchmark

## Run examples 

Live serial long-time capture (stop with Ctrl+C):

    .\uart_stats.exe --port COM15 --baud 38400 --series series.csv --hist-wall wall.png --hist-game game.png --flush-every 50 --heartbeat-s 5

Live serial short-time capture (stop with Ctrl+C):

    .\uart_stats.exe --port COM15 --baud 38400 --series series.csv --hist-wall wall.png --hist-game game.png --limit 1000

Capture from file and generate histograms:

    .\uart_stats.exe --file frames.txt --series run.csv --hist-wall --hist-game

In case of runnig as python script, replace:

`.\uart_stats.exe` with `python bm.py`

## Input

`--port PORT`\
Serial port to read from (e.g. `COM5`, `/dev/ttyUSB0`)

`--baud BAUD`\
Serial baud rate (default: `115200`)

`--file FILE`\
Read frames from a text file (one message per line)

If no input option is given, data is read from `STDIN`.

## Output

`--series FILE`\
Write streaming CSV with wall/game/error timing series and final stats
header\
(saved under the `data/` directory)

`--hist-wall [FILE]`\
Save wall-clock interval histogram\
(default: `wall_hist.png`)

`--hist-game [FILE]`\
Save game-clock interval histogram\
(default: `game_hist.png`)

## Timing and validation

`--tolerance-ms MS`\
Error tolerance in milliseconds (default: `30.0`)

`--include-bad-crc`\
Include frames with bad CRC in analysis

## Runtime behavior

`--heartbeat-s SECONDS`\
Print heartbeat every N seconds\
(`0` disables heartbeat, default: `60`)

`--limit N`\
Stop after N frames (`0` = unlimited)

`--flush-every N`\
Flush streaming CSV every N samples (default: `1`)

## Notes

All output files are written to the `data/` directory (created
automatically).\
When built as an executable, `data/` is created next to the executable.\
Terminate long runs with Ctrl+C to finalize statistics and write output
files.

## Build
    pyinstaller --onefile --name uart_stats bm.py

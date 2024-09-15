$NUM_CLIENTS = 5

for ($i = 0; $i -lt $NUM_CLIENTS; $i++) {
    Start-Process python -ArgumentList "client.py", "--partition-id", $i -NoNewWindow
}

Wait-Process -Name python
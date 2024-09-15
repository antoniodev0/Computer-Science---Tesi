$NUM_CLIENTS = 5

for ($i = 0; $i -lt $NUM_CLIENTS; $i++) {
    # Avvia ogni client con il parametro --partition-id corrispondente
    Start-Process python -ArgumentList "C:\Users\anton\Desktop\DP\client.py", "--partition-id", $i -NoNewWindow
}

# Attendi che tutti i processi Python terminino
Wait-Process -Name python

Write-Host "Tutti i client hanno terminato l'esecuzione."

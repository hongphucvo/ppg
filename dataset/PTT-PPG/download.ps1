curl https://physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/ -o index.html

# Read the HTML file and extract links
$links = Select-String -Path "index.html" -Pattern 'href="([^"]+\.(dat|hea|atr|csv))"' | ForEach-Object {
    ($_ -match 'href="([^"]+\.(dat|hea|atr|csv))"') | Out-Null
    $matches[1]
}

# Save the links to a file (optional)
$links | Set-Content -Path "data_links.txt"

# Base URL
$baseUrl = "https://physionet.org/files/pulse-transit-time-ppg/1.1.0/csv/"

# Download each file
foreach ($link in $links) {
    $url = "$baseUrl$link"
    curl -O $url
}
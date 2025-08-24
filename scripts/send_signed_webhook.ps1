Param(
    [string]$Url = "http://127.0.0.1:8080/webhook/n8n/chat/",
    [string]$Secret = "supersecret",
    [string]$UserId = "user_123",
    [string]$Message = "hello",
    [string]$Timestamp = (Get-Date -Format o)
)

# Build payload and compact JSON body
$payload = @{ user_id = $UserId; user_message = $Message; timestamp = $Timestamp }
$body = ($payload | ConvertTo-Json -Compress)

# Compute HMAC-SHA256 hex digest (lowercase)
$hmacKey = [System.Text.Encoding]::UTF8.GetBytes($Secret)
$hmac = New-Object System.Security.Cryptography.HMACSHA256 -ArgumentList (, $hmacKey)
$sigBytes = $hmac.ComputeHash([System.Text.Encoding]::UTF8.GetBytes($body))
$sig = ([System.BitConverter]::ToString($sigBytes)).Replace('-', '').ToLower()

Write-Host "POST URL: $Url"
Write-Host "BODY: $body"
Write-Host "SIG: $sig"

try {
    $resp = Invoke-RestMethod -Uri $Url -Method Post -Body $body -ContentType 'application/json' -Headers @{ 'X-N8N-SIGNATURE' = $sig }
    Write-Host "Response:`n" ($resp | ConvertTo-Json -Compress)
}
catch {
    Write-Host "Error: $($_.Exception.Message)"
    if ($_.Exception.Response) {
        $stream = $_.Exception.Response.GetResponseStream()
        $reader = New-Object System.IO.StreamReader($stream)
        $bodyText = $reader.ReadToEnd()
        Write-Host "Response body:`n$bodyText"
    }
}

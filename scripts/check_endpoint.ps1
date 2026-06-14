param(
    [string]$Endpoint = "http://localhost:8082/serve/text-sentiment",
    [string]$Text = "The delivery arrived early and support was helpful."
)

$Body = @{ text = $Text } | ConvertTo-Json -Compress
Invoke-RestMethod -Uri $Endpoint -Method Post -ContentType "application/json" -Body $Body
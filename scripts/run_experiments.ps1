param(
    [string]$DatasetId = "",
    [string]$Queue = "students"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $ProjectRoot

if (-not $DatasetId) {
    $DatasetIdPath = Join-Path $ProjectRoot "dataset\dataset_id.txt"
    if (-not (Test-Path -LiteralPath $DatasetIdPath)) {
        throw "DatasetId is empty and dataset\dataset_id.txt does not exist. Run dataset\create_dataset.py first."
    }
    $DatasetId = (Get-Content -LiteralPath $DatasetIdPath -Raw).Trim()
}

python .\train\train.py --dataset-id $DatasetId --queue $Queue --remote --max-features 800 --ngram-max 1 --c 0.7 --random-state 2026
python .\train\train.py --dataset-id $DatasetId --queue $Queue --remote --max-features 1600 --ngram-max 2 --c 1.8 --random-state 2027
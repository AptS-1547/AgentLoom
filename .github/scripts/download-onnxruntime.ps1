param(
    [Parameter(Mandatory = $true)]
    [string]$Url,

    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,

    [Parameter(Mandatory = $true)]
    [string]$ExtractRoot,

    [Parameter(Mandatory = $true)]
    [string]$ExpectedRoot
)

$ErrorActionPreference = "Stop"

if (Test-Path -LiteralPath $ExpectedRoot) {
    Write-Host "ONNX Runtime already present at $ExpectedRoot"
    exit 0
}

New-Item -ItemType Directory -Force -Path $ExtractRoot | Out-Null

if (-not (Test-Path -LiteralPath $ArchivePath)) {
    Write-Host "Downloading $Url"
    Invoke-WebRequest -Uri $Url -OutFile $ArchivePath
} else {
    Write-Host "Using cached archive $ArchivePath"
}

Write-Host "Extracting $ArchivePath to $ExtractRoot"
Expand-Archive -Path $ArchivePath -DestinationPath $ExtractRoot -Force

if (-not (Test-Path -LiteralPath $ExpectedRoot)) {
    throw "Expected extracted directory not found: $ExpectedRoot"
}

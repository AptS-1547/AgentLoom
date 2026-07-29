[CmdletBinding()]
param(
    [string]$OutputPath = (Join-Path $PSScriptRoot "..\config\certs\mozilla-ca-bundle.pem"),
    [string]$BundleUrl = "https://curl.se/ca/cacert.pem",
    [string]$ChecksumUrl = "https://curl.se/ca/cacert.pem.sha256"
)

$ErrorActionPreference = "Stop"

$bundleTemp = [System.IO.Path]::GetTempFileName()
$checksumTemp = [System.IO.Path]::GetTempFileName()

try {
    Invoke-WebRequest -UseBasicParsing -Uri $BundleUrl -OutFile $bundleTemp
    Invoke-WebRequest -UseBasicParsing -Uri $ChecksumUrl -OutFile $checksumTemp

    $checksumText = Get-Content -LiteralPath $checksumTemp -Raw -Encoding UTF8
    $checksumMatch = [regex]::Match($checksumText, "(?i)\b[0-9a-f]{64}\b")
    if (-not $checksumMatch.Success) {
        throw "The official checksum response did not contain a SHA-256 value."
    }

    $expectedHash = $checksumMatch.Value.ToUpperInvariant()
    $actualHash = (Get-FileHash -LiteralPath $bundleTemp -Algorithm SHA256).Hash.ToUpperInvariant()
    if ($actualHash -ne $expectedHash) {
        throw "Mozilla CA bundle SHA-256 mismatch: expected $expectedHash, got $actualHash."
    }

    $bundleText = Get-Content -LiteralPath $bundleTemp -Raw -Encoding UTF8
    $certificateCount = ([regex]::Matches(
        $bundleText,
        "-----BEGIN CERTIFICATE-----"
    )).Count
    if ($certificateCount -lt 1) {
        throw "The downloaded file does not contain PEM certificates."
    }

    $resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
    $outputDirectory = [System.IO.Path]::GetDirectoryName($resolvedOutput)
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
    Copy-Item -LiteralPath $bundleTemp -Destination $resolvedOutput -Force

    Write-Host "Mozilla CA bundle updated: $resolvedOutput"
    Write-Host "SHA-256: $actualHash"
    Write-Host "Certificates: $certificateCount"
}
finally {
    Remove-Item -LiteralPath $bundleTemp -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $checksumTemp -Force -ErrorAction SilentlyContinue
}

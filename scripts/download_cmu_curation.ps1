param(
    [Parameter(Mandatory = $true)][string]$Config,
    [string]$Destination = 'data/curation/sources/cmu'
)
# Use the Windows certificate store; do not disable TLS verification.
$ErrorActionPreference = 'Stop'
$configuration = Get-Content -LiteralPath $Config -Raw -Encoding UTF8 | ConvertFrom-Json
if ($configuration.source_base -ne 'https://mocap.cs.cmu.edu/subjects') {
    throw 'Only the CMU official capture host is supported.'
}
$destinationRoot = [IO.Path]::GetFullPath($Destination)
New-Item -ItemType Directory -Force -Path $destinationRoot | Out-Null
$files = [ordered]@{}
foreach ($group in $configuration.groups) {
    $subject = $group.subject
    if ($subject -notmatch '^\d{2,3}$') { throw 'Invalid subject identifier.' }
    $files["$subject.asf"] = "$($configuration.source_base)/$subject/$subject.asf"
    foreach ($clip in $group.clips.PSObject.Properties.Name) {
        if ($clip -notmatch '^\d{2}$') { throw 'Invalid clip identifier.' }
        $name = "${subject}_${clip}.amc"
        $files[$name] = "$($configuration.source_base)/$subject/$name"
    }
}
$index = 0
foreach ($name in $files.Keys) {
    $index++
    $capturePath = [IO.Path]::GetFullPath((Join-Path $destinationRoot $name))
    if (-not $capturePath.StartsWith($destinationRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Capture destination escaped the data folder.'
    }
    $recordPath = $capturePath + '.source.json'
    if (Test-Path -LiteralPath $capturePath) {
        if (-not (Test-Path -LiteralPath $recordPath)) {
            throw "Cached capture lacks provenance: $name. Use a fresh destination."
        }
        $record = Get-Content -LiteralPath $recordPath -Raw -Encoding UTF8 | ConvertFrom-Json
        $currentHash = (Get-FileHash -LiteralPath $capturePath -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($record.url -ne $files[$name] -or $record.sha256 -ne $currentHash) {
            throw "Cached capture provenance changed: $name"
        }
        Write-Output "$index/$($files.Count) cached $name"
        continue
    }
    $temporaryPath = $capturePath + '.download'
    Invoke-WebRequest -Uri $files[$name] -OutFile $temporaryPath -TimeoutSec 60
    $firstLine = Get-Content -LiteralPath $temporaryPath -TotalCount 1
    if ($firstLine -match '^\s*<' -or (Get-Item -LiteralPath $temporaryPath).Length -eq 0) {
        throw "Unexpected capture response: $name"
    }
    # Both absolute paths were derived from the checked destination above.
    Move-Item -LiteralPath $temporaryPath -Destination $capturePath
    $captureHash = (Get-FileHash -LiteralPath $capturePath -Algorithm SHA256).Hash.ToLowerInvariant()
    [ordered]@{
        url = $files[$name]
        sha256 = $captureHash
        downloaded_at = [DateTimeOffset]::UtcNow.ToString('o')
    } | ConvertTo-Json | Set-Content -LiteralPath $recordPath -Encoding UTF8
    Write-Output "$index/$($files.Count) downloaded $name"
}

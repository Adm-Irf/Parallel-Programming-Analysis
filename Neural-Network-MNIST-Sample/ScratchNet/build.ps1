param([ValidateSet('all','sequential','openmp','mpi')][string]$Mode = 'all')
$ErrorActionPreference = 'Stop'
$vswhere = Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Microsoft Visual Studio\Installer\vswhere.exe'
if (-not (Get-Command cl -ErrorAction SilentlyContinue)) {
    if (-not (Test-Path $vswhere)) { throw 'Install Visual Studio Build Tools with Desktop development with C++.' }
    $installation = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installation) { throw 'C++ compiler component not found.' }
    & "$installation\Common7\Tools\Launch-VsDevShell.ps1" -Arch amd64 -HostArch amd64 -SkipAutomaticLocation
}
Push-Location $PSScriptRoot
try {
    $targets = if ($Mode -eq 'all') { @('sequential','openmp','mpi') } else { @($Mode) }
    foreach ($target in $targets) {
        New-Item -ItemType Directory -Force "build\$target" | Out-Null
        $flags = @('/nologo','/EHsc','/std:c++17','/O2','/MT','/Iinclude','src\*.cpp',"/Fo:build\$target\")
        $name = 'neural_network.exe'
        if ($target -eq 'openmp') { $flags += '/openmp'; $name = 'neural_network_omp.exe' }
        if ($target -eq 'mpi') {
            if (-not $env:MSMPI_INC -or -not $env:MSMPI_LIB64) { throw 'Install the Microsoft MPI SDK and reopen PowerShell.' }
            $env:INCLUDE = "$env:MSMPI_INC;$env:INCLUDE"
            $env:LIB = "$env:MSMPI_LIB64;$env:LIB"
            $flags += '/DUSE_MPI'
            $name = 'neural_network_mpi.exe'
        }
        $flags += "/Fe:build\$name"
        if ($target -eq 'mpi') { $flags += @('/link','msmpi.lib') }
        & cl @flags
        if ($LASTEXITCODE -ne 0) { throw "Compilation failed: $target" }
    }
} finally { Pop-Location }

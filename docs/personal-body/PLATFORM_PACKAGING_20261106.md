# Runtime packaging follow-up — 2026-11-06

The builder now checks the native Blender and Python executable headers before
creating an output directory. A darwin-arm64 binary cannot be labeled darwin-x64;
Windows requires AMD64 PE32+ headers. Supported layouts remain darwin-arm64,
darwin-x64 and windows-x64. Universal runtime packages are not qualified.

The candidate startup entry checks the host OS and architecture, all required
hash-pinned entries, the exact file set and every file hash before importing the
host. Unlisted files/directories, symlinks, special files, malformed paths,
missing entries, modified bytes and incorrectly labeled executables are rejected.
Distributions with symlinks need a future explicit packaging contract; silently
leaving linked executables outside the file manifest is not accepted.

These are builder/entry source changes for a **new immutable package**. They do
not modify or retroactively harden the existing personal-converter/10 runtime
with manifest `eaac3841c8b9e693ff88c959eabaceaf1ecc298e547eede748e83ad4a0de1d70`.
The converter owner must issue and record the next version/manifest/fingerprint,
then rerun native packaging and model checks. No new runtime is adopted here.
No geometry, retarget, deformation selector or safety threshold changed.

Validation on macOS 26.6.2 arm64: five new synthetic artifact tests; local
regression suite 27 collected / 23 passed / four Blender-only skips; 54 smoke
tests passed using the existing project Python environment; compileall and diff
checks passed. The new validator also accepted all 6,591 files of immutable v10.
Synthetic Mach-O/PE headers and mocked host identity test rejection behavior,
not target OS execution. The baseline runtime has separate actual FBX/CS3C
adapter evidence in the client platform handoff.

Only darwin-arm64 has an actual approved package/host in this follow-up. Existing
PR159 CI used Windows Server 2025 x64 and macOS 26 arm64 for Rust compile/unit
checks; both ignored five packaged-runtime opt-in tests. No Intel Mac package
or actual Windows converter execution is certified. No private model, decoder
payload or new runtime was uploaded to CI.

Signing/notarization, Windows/Intel runtime execution, decoder redistribution,
Blender corresponding-source/notices delivery, CSP acceptance, live BFF and
geometry quality remain distinct gates. Draft/default-off/releaseReady:false
remain. No deployment, merge, release, key or production-setting change occurred.

The Windows batch launcher is written as exact UTF-8 bytes. Its authored CRLF
must not be translated again by Windows text mode (which would produce
CRCRLF). A synthetic Windows-layout build with emulated text translation checks
the exact launcher line endings and quoted bundled-interpreter path. This is a
source/byte regression, not Windows execution or signing acceptance.

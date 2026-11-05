# Staging-only integration · 2026-11-05

This temporary deployment branch preserves the currently deployed staging props, CSP body and thumbnail-outline changes from d6e502637bbfdca88ae9554b5d014bc20eb4f79d while integrating develop efa8d68ff6ad7a27b1c613069ea77af53c00b109.

It does not merge those pending changes to develop/main or enable any new feature. The existing staging catalog/assets/secrets/feature settings remain. The ordinary production workflow guards and latest image tag are unchanged. The explicit staging branch is accepted only for the manual staging deployment path; the dedicated scene worker uses the already provisioned staging target and existing deploy role. No release or production action is authorized by this branch.

Companion BFF deploy uses develop b4a943019de97d661c574216a505e2407a613c33. Desktop staging release is staging-dev.448 at 5934f0d2799d64074cb5836b9d7bab48d7246f1e.

## Legacy recipe compatibility

The develop hand refactor only extracts the existing no-I/O solve. Existing approved scene recipes bind the complete legacy hands.py bytes. To retain those reviewed bindings without changing any receipt/token, this integration preserves the legacy file byte-for-byte and exposes the same solve in interactive_hands.py for the new interactive adapter. The solve statement AST was compared to the legacy bake body after excluding reset/import/export I/O and was identical. No catalog or reviewed asset hash was rewritten.

268 focused deployment, interactive and existing prop/outline/hand regression tests passed before deployment. GitHub deployment workflows retain their complete test and image startup gates.

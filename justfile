# Default lists available commands
default:
    @just --list

# Format code using clang-format
format *ARGS:
    @./format.sh {{ARGS}}

# Static analysis using clang-tidy
tidy *ARGS:
    @./tidy.sh {{ARGS}}

# Format and then run tidy checks
all: format tidy

# Show diffs of what would be formatted without applying changes
format-check:
    @./format.sh --dry-run

# Run tidy with fixes
tidy-fix:
    @./tidy.sh --fix

# Format and fix everything
fix: format tidy-fix


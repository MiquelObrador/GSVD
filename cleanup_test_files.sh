#!/bin/bash

# Clean up test files generated during validation
echo "🧹 Cleaning up test files..."

# Remove temporary analysis directories
if [ -d "test_analysis" ]; then
    rm -rf test_analysis
    echo "✅ Removed test_analysis directory"
fi

if [ -d "test_analysis_fixed" ]; then
    rm -rf test_analysis_fixed
    echo "✅ Removed test_analysis_fixed directory"
fi

# Remove any temporary result files
find . -name "tmp*.jsonl" -delete 2>/dev/null || true
find /tmp -name "tmp*.jsonl" -delete 2>/dev/null || true

echo "🎉 Cleanup complete!"
#!/bin/bash
# Download 10 matrices from SuiteSparse for benchmarking
# Based on benchmarking_plan.md selection criteria

set -e  # Exit on error

DOWNLOAD_DIR="suitesparse/raw"
mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

echo "============================================"
echo "Downloading 10 Benchmark Matrices"
echo "============================================"
echo ""

# Base URL for SuiteSparse Matrix Collection
BASE_URL="https://suitesparse-collection-website.herokuapp.com/MM"

# Matrix list with properties
# Format: "group/name|nnz/M|description"
declare -a MATRICES=(
    # CSR-targeted matrices (will use CSR in DSL)
    "HB/bcsstk01|3.8|Structural FEM (48x48)"
    "Pothen/bodyy4|3.9|Structural problem (17546x17546)"
    "Nasa/nasa2146|4.2|Structural NASA matrix (2146x2146)"

    # High density matrices
    "vanHeukelum/cage6|4.7|DNA electrophoresis (3360x3360)"

    # CSC-targeted matrices (will use CSC in DSL)
    "Belcastro/mouse_gene|2.3|Gene expression (45101x45101)"
    "Williams/consph|4.2|Computational fluid dynamics (83334x83334)"
    "Williams/cant|4.0|Cantilever FEM (62451x62451)"
    "Williams/pdb1HYS|4.8|Protein data bank (36417x36417)"

    # Additional variety
    "Simon/raefsky1|2.4|Fluid structure (3242x3242)"
    "Simon/venkat01|2.5|CFD (62424x62424)"
)

# Download function
download_matrix() {
    local matrix_path=$1
    local nnz_per_m=$2
    local description=$3

    local group=$(dirname "$matrix_path")
    local name=$(basename "$matrix_path")

    echo "----------------------------------------"
    echo "Downloading: $name"
    echo "  Group: $group"
    echo "  nnz/M: ~$nnz_per_m"
    echo "  Description: $description"
    echo "----------------------------------------"

    # Download tar.gz file
    local url="${BASE_URL}/${group}/${name}.tar.gz"
    echo "URL: $url"

    if curl -L -s -o "${name}.tar.gz" "$url"; then
        echo "✓ Downloaded ${name}.tar.gz"

        # Extract .mtx file
        tar -xzf "${name}.tar.gz" "${name}/${name}.mtx" 2>/dev/null || \
        tar -xzf "${name}.tar.gz" "*/mat/${name}.mtx" 2>/dev/null || \
        tar -xzf "${name}.tar.gz" "*/${name}.mtx" 2>/dev/null || true

        # Find and move .mtx file
        find "${name}" -name "${name}.mtx" -exec mv {} . \; 2>/dev/null || true

        if [ -f "${name}.mtx" ]; then
            echo "✓ Extracted ${name}.mtx"
            # Get matrix info
            local dims=$(grep -v "^%" "${name}.mtx" | head -1)
            echo "  Dimensions: $dims"
            rm -rf "${name}" "${name}.tar.gz"
        else
            echo "✗ Failed to extract .mtx file"
            rm -rf "${name}" "${name}.tar.gz"
            return 1
        fi
    else
        echo "✗ Download failed"
        return 1
    fi

    echo ""
}

# Download all matrices
success_count=0
fail_count=0

for matrix_spec in "${MATRICES[@]}"; do
    IFS='|' read -r path nnz_per_m desc <<< "$matrix_spec"

    if download_matrix "$path" "$nnz_per_m" "$desc"; then
        ((success_count++))
    else
        ((fail_count++))
    fi
done

echo "============================================"
echo "Download Summary"
echo "============================================"
echo "Success: $success_count"
echo "Failed: $fail_count"
echo ""
echo "Downloaded matrices:"
ls -lh *.mtx 2>/dev/null || echo "No .mtx files found"

exit 0

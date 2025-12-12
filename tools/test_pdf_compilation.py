"""Test that .tex files compile to match uploaded PDFs (page count + file size)."""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

# Try to import PyMuPDF for PDF analysis
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    print("Warning: PyMuPDF not installed. Install with: pip install pymupdf")

ROOT = Path(__file__).parent.parent
MAPPING_FILE = Path(__file__).parent / "pdf_source_mapping.json"
DOWNLOADS = ROOT / "static" / "downloads"
SIZE_TOLERANCE = 0.15  # 15% tolerance for file size


def get_pdf_info(pdf_path: Path) -> dict:
    """Get page count and file size for a PDF."""
    if not pdf_path.exists():
        return {"pages": 0, "size": 0, "exists": False}

    size = pdf_path.stat().st_size

    if HAS_FITZ:
        try:
            doc = fitz.open(pdf_path)
            pages = len(doc)
            doc.close()
            return {"pages": pages, "size": size, "exists": True}
        except Exception as e:
            return {"pages": 0, "size": size, "exists": True, "error": str(e)}

    # Fallback: use pdfinfo if available
    try:
        result = subprocess.run(
            ["pdfinfo", str(pdf_path)],
            capture_output=True,
            text=True,
            timeout=10
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Pages:"):
                pages = int(line.split(":")[1].strip())
                return {"pages": pages, "size": size, "exists": True}
    except Exception:
        pass

    return {"pages": -1, "size": size, "exists": True, "error": "Could not read pages"}


def compile_tex(tex_path: Path, output_dir: Path) -> Path:
    """Compile a .tex file and return path to output PDF."""
    if not tex_path.exists():
        return None

    # Copy .tex and any included files to temp dir
    tex_dir = tex_path.parent

    # Run pdflatex twice (for references)
    env = os.environ.copy()
    for _ in range(2):
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(output_dir), str(tex_path)],
            capture_output=True,
            text=True,
            cwd=str(tex_dir),
            timeout=120,
            env=env
        )

    # Find output PDF
    pdf_name = tex_path.stem + ".pdf"
    output_pdf = output_dir / pdf_name

    if output_pdf.exists():
        return output_pdf
    return None


def test_single_pdf(mapping: dict, verbose: bool = True) -> dict:
    """Test a single PDF against its source .tex file."""
    pdf_name = mapping["pdf"]
    result = {
        "pdf": pdf_name,
        "status": "SKIP",
        "reason": None,
        "pages_expected": 0,
        "pages_actual": 0,
        "size_ratio": 0
    }

    # Skip generated files
    if mapping["type"] == "generated":
        result["reason"] = "Generated file (no source)"
        return result

    # Get expected PDF info
    expected_pdf = DOWNLOADS / pdf_name
    expected_info = get_pdf_info(expected_pdf)

    if not expected_info["exists"]:
        result["status"] = "FAIL"
        result["reason"] = "Expected PDF not found"
        return result

    result["pages_expected"] = expected_info["pages"]
    result["size_expected"] = expected_info["size"]

    # Get source .tex path
    tex_path = Path(mapping["full_path"]) if mapping["full_path"] else None
    if not tex_path or not tex_path.exists():
        result["status"] = "FAIL"
        result["reason"] = "Source .tex not found"
        return result

    # Compile in temp directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        try:
            compiled_pdf = compile_tex(tex_path, tmpdir)

            if not compiled_pdf:
                result["status"] = "FAIL"
                result["reason"] = "Compilation failed"
                return result

            # Compare
            actual_info = get_pdf_info(compiled_pdf)
            result["pages_actual"] = actual_info["pages"]
            result["size_actual"] = actual_info["size"]

            # Check page count
            if expected_info["pages"] != actual_info["pages"]:
                result["status"] = "FAIL"
                result["reason"] = f"Page mismatch: {actual_info['pages']}/{expected_info['pages']}"
                return result

            # Check file size (within tolerance)
            if expected_info["size"] > 0:
                size_ratio = actual_info["size"] / expected_info["size"]
                result["size_ratio"] = size_ratio

                if abs(1 - size_ratio) > SIZE_TOLERANCE:
                    result["status"] = "WARN"
                    result["reason"] = f"Size differs: {size_ratio:.0%}"
                    return result

            result["status"] = "PASS"
            result["size_ratio"] = actual_info["size"] / expected_info["size"] if expected_info["size"] > 0 else 1.0

        except subprocess.TimeoutExpired:
            result["status"] = "FAIL"
            result["reason"] = "Compilation timeout"
        except Exception as e:
            result["status"] = "FAIL"
            result["reason"] = str(e)

    return result


def main():
    """Run compilation tests on all PDFs."""
    print("PDF Compilation Test")
    print("=" * 60)

    if not HAS_FITZ:
        print("ERROR: PyMuPDF required. Install with: pip install pymupdf")
        return

    # Load mappings
    if not MAPPING_FILE.exists():
        print(f"ERROR: Mapping file not found: {MAPPING_FILE}")
        print("Run: python tools/map_pdf_sources.py")
        return

    with open(MAPPING_FILE) as f:
        data = json.load(f)

    mappings = data["mappings"]
    print(f"Testing {len(mappings)} PDF mappings...\n")

    results = []
    passed = 0
    failed = 0
    skipped = 0
    warned = 0

    for mapping in mappings:
        result = test_single_pdf(mapping)
        results.append(result)

        # Print result
        pdf = result["pdf"]
        status = result["status"]

        if status == "PASS":
            passed += 1
            ratio = result.get("size_ratio", 1)
            print(f"  PASS  {pdf:40} pages: {result['pages_expected']}, size: {ratio:.0%}")
        elif status == "WARN":
            warned += 1
            print(f"  WARN  {pdf:40} {result['reason']}")
        elif status == "SKIP":
            skipped += 1
            print(f"  SKIP  {pdf:40} {result['reason']}")
        else:
            failed += 1
            print(f"  FAIL  {pdf:40} {result['reason']}")

    # Summary
    print("\n" + "=" * 60)
    print(f"Summary: {passed} passed, {warned} warnings, {failed} failed, {skipped} skipped")
    print(f"Total: {len(mappings)} PDFs")

    # Return exit code
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main() or 0)

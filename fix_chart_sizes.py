"""
LaTeX Chart Size Fixer
Analyzes and fixes oversized charts in Beamer presentations
Can be run on individual weeks or all weeks
"""

import re
import os
import sys
from pathlib import Path
from datetime import datetime
import shutil

class ChartSizeFixer:
    def __init__(self, week_dir):
        self.week_dir = Path(week_dir)
        self.issues = []
        self.fixes_applied = []

        # Optimal sizes for different contexts
        self.optimal_sizes = {
            'full_width': '0.85\\textwidth',
            'two_column': '0.9\\textwidth',  # In 0.48 column
            'three_column': '0.9\\textwidth',  # In 0.32 column
            'in_tcolorbox': '0.85\\textwidth',
            'default': '0.8\\textwidth'
        }

    def analyze_tex_file(self, tex_file):
        """Analyze a single .tex file for chart sizing issues"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {tex_file.name}")
        print(f"{'='*60}")

        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')

        # Find all includegraphics commands
        pattern = r'\\includegraphics\[([^\]]*)\]\{([^}]*)\}'
        matches = list(re.finditer(pattern, content))

        file_issues = []

        for match in matches:
            params = match.group(1)
            chart_path = match.group(2)

            # Get line number
            line_num = content[:match.start()].count('\n') + 1

            # Extract width parameter
            width_match = re.search(r'width=([^,\]]+)', params)

            if width_match:
                width_value = width_match.group(1).strip()
                issue = self._analyze_width(width_value, chart_path, line_num, lines[line_num-1])
                if issue:
                    file_issues.append(issue)
            else:
                # No width specified - potentially problematic
                file_issues.append({
                    'file': tex_file.name,
                    'line': line_num,
                    'chart': chart_path,
                    'issue': 'No width specified',
                    'current': 'none',
                    'suggested': self.optimal_sizes['default'],
                    'context': lines[line_num-1].strip()[:80]
                })

        return file_issues

    def _analyze_width(self, width_value, chart_path, line_num, context_line):
        """Analyze if a width value is problematic"""
        issues = []

        # Check for full textwidth (usually too large)
        if width_value == '\\textwidth':
            return {
                'issue': 'Full textwidth (too large)',
                'current': width_value,
                'suggested': self.optimal_sizes['full_width'],
                'chart': chart_path,
                'line': line_num,
                'severity': 'high',
                'context': context_line.strip()[:80]
            }

        # Check for >0.9\textwidth
        match = re.match(r'(0\.\d+)\\textwidth', width_value)
        if match:
            ratio = float(match.group(1))
            if ratio > 0.90:
                return {
                    'issue': f'Too large ({ratio:.0%} of textwidth)',
                    'current': width_value,
                    'suggested': self.optimal_sizes['full_width'],
                    'chart': chart_path,
                    'line': line_num,
                    'severity': 'medium',
                    'context': context_line.strip()[:80]
                }
            elif ratio > 0.95:
                return {
                    'issue': f'Way too large ({ratio:.0%} of textwidth)',
                    'current': width_value,
                    'suggested': self.optimal_sizes['full_width'],
                    'chart': chart_path,
                    'line': line_num,
                    'severity': 'high',
                    'context': context_line.strip()[:80]
                }

        return None

    def fix_file(self, tex_file, issues, dry_run=True):
        """Apply fixes to a .tex file"""
        if not issues:
            return

        print(f"\n{'='*60}")
        print(f"Fixing: {tex_file.name}")
        print(f"{'='*60}")

        # Backup original file
        if not dry_run:
            backup_dir = self.week_dir / 'archive' / 'previous'
            backup_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            backup_path = backup_dir / f"{tex_file.stem}_pre_chartfix_{timestamp}.tex"
            shutil.copy2(tex_file, backup_path)
            print(f"[OK] Backed up to: {backup_path.name}")

        with open(tex_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply fixes in reverse order (to preserve line numbers)
        sorted_issues = sorted(issues, key=lambda x: x['line'], reverse=True)

        fixed_content = content
        for issue in sorted_issues:
            # Find and replace the specific width parameter
            old_pattern = f"width={issue['current']}"
            new_pattern = f"width={issue['suggested']}"

            # Use more precise replacement to avoid false matches
            lines = fixed_content.split('\n')
            if issue['line'] - 1 < len(lines):
                line = lines[issue['line'] - 1]
                if old_pattern in line:
                    lines[issue['line'] - 1] = line.replace(old_pattern, new_pattern, 1)
                    fixed_content = '\n'.join(lines)

                    print(f"  Line {issue['line']}: {issue['chart']}")
                    print(f"    {old_pattern} -> {new_pattern}")

                    if not dry_run:
                        self.fixes_applied.append({
                            'file': tex_file.name,
                            'line': issue['line'],
                            'chart': issue['chart'],
                            'change': f"{old_pattern} -> {new_pattern}"
                        })

        if not dry_run:
            with open(tex_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            print(f"[OK] Fixed {len(issues)} chart size issues")
        else:
            print(f"[DRY RUN] Would fix {len(issues)} issues")

    def analyze_week(self):
        """Analyze all .tex files in the week directory"""
        print(f"\n{'#'*60}")
        print(f"# Analyzing Week: {self.week_dir.name}")
        print(f"{'#'*60}")

        # Find all part*.tex files
        tex_files = sorted(self.week_dir.glob('part*.tex'))

        if not tex_files:
            print("No part*.tex files found!")
            return

        all_issues = {}
        total_issues = 0

        for tex_file in tex_files:
            issues = self.analyze_tex_file(tex_file)
            if issues:
                all_issues[tex_file] = issues
                total_issues += len(issues)

                print(f"\n  Found {len(issues)} issues:")
                for issue in issues:
                    severity_mark = '[HIGH]' if issue.get('severity') == 'high' else '[MED]'
                    print(f"    {severity_mark} Line {issue['line']}: {issue['issue']}")
                    print(f"       Chart: {issue['chart']}")
                    print(f"       Current: {issue['current']} -> Suggested: {issue['suggested']}")
            else:
                print(f"\n  [OK] No issues found")

        print(f"\n{'='*60}")
        print(f"SUMMARY: Found {total_issues} chart sizing issues across {len(all_issues)} files")
        print(f"{'='*60}")

        return all_issues

    def fix_week(self, dry_run=True):
        """Fix all issues in the week"""
        all_issues = self.analyze_week()

        if not all_issues:
            print("\n[OK] No issues to fix!")
            return

        print(f"\n{'='*60}")
        if dry_run:
            print("DRY RUN MODE - No files will be modified")
        else:
            print("APPLYING FIXES...")
        print(f"{'='*60}")

        for tex_file, issues in all_issues.items():
            self.fix_file(tex_file, issues, dry_run)

        if not dry_run:
            print(f"\n{'='*60}")
            print(f"[OK] FIXED {len(self.fixes_applied)} issues across {len(all_issues)} files")
            print(f"{'='*60}")
            print("\nRecommendation: Recompile LaTeX to verify fixes")
        else:
            print(f"\n{'='*60}")
            print(f"To apply fixes, run with --fix flag")
            print(f"{'='*60}")


def fix_all_weeks(base_dir, dry_run=True):
    """Fix chart sizes in all weeks"""
    base_path = Path(base_dir)
    week_dirs = sorted([d for d in base_path.glob('Week_*') if d.is_dir()])

    print(f"\n{'#'*60}")
    print(f"# Analyzing ALL WEEKS")
    print(f"{'#'*60}")
    print(f"Found {len(week_dirs)} weeks\n")

    total_issues = 0
    weeks_with_issues = []

    for week_dir in week_dirs:
        fixer = ChartSizeFixer(week_dir)
        all_issues = fixer.analyze_week()

        if all_issues:
            weeks_with_issues.append((week_dir, fixer, all_issues))
            total_issues += sum(len(issues) for issues in all_issues.values())

    print(f"\n{'#'*60}")
    print(f"# GLOBAL SUMMARY")
    print(f"{'#'*60}")
    print(f"Weeks with issues: {len(weeks_with_issues)}/{len(week_dirs)}")
    print(f"Total issues found: {total_issues}")

    if weeks_with_issues and not dry_run:
        print(f"\n{'='*60}")
        print("FIXING ALL WEEKS...")
        print(f"{'='*60}")

        for week_dir, fixer, all_issues in weeks_with_issues:
            print(f"\nFixing {week_dir.name}...")
            for tex_file, issues in all_issues.items():
                fixer.fix_file(tex_file, issues, dry_run=False)

        print(f"\n{'='*60}")
        print(f"[OK] COMPLETED: Fixed {total_issues} issues across {len(weeks_with_issues)} weeks")
        print(f"{'='*60}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Fix oversized charts in LaTeX Beamer presentations')
    parser.add_argument('week', nargs='?', help='Week directory (e.g., Week_04) or "all"')
    parser.add_argument('--fix', action='store_true', help='Apply fixes (default is dry-run)')
    parser.add_argument('--all', action='store_true', help='Process all weeks')

    args = parser.parse_args()

    if args.all or (args.week and args.week.lower() == 'all'):
        # Fix all weeks
        base_dir = Path(__file__).parent
        fix_all_weeks(base_dir, dry_run=not args.fix)
    elif args.week:
        # Fix specific week
        week_dir = Path(args.week)
        if not week_dir.exists():
            week_dir = Path(__file__).parent / args.week

        if not week_dir.exists():
            print(f"Error: Directory not found: {args.week}")
            sys.exit(1)

        fixer = ChartSizeFixer(week_dir)
        fixer.fix_week(dry_run=not args.fix)
    else:
        # Interactive mode
        print("LaTeX Chart Size Fixer")
        print("=" * 60)
        print("\nUsage:")
        print("  python fix_chart_sizes.py Week_04          # Analyze Week 4")
        print("  python fix_chart_sizes.py Week_04 --fix    # Fix Week 4")
        print("  python fix_chart_sizes.py --all            # Analyze all weeks")
        print("  python fix_chart_sizes.py --all --fix      # Fix all weeks")


if __name__ == '__main__':
    main()
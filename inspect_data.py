"""
Inspection script for raw image files.

Usage:
    python inspect_data.py --file data/raw/real/metal01_slice0000_H512_W512.raw
    python inspect_data.py --file data/raw/real/metal01_slice0000_H512_W512.raw --png output.png
    python inspect_data.py --dir data/raw/real
"""

import argparse
from pathlib import Path
from src.utils.raw_visualizer import (
    inspect_raw_file, raw_to_png, visualize_raw
)


def main():
    parser = argparse.ArgumentParser(
        description="Inspect and visualize raw CT image files"
    )
    
    parser.add_argument(
        '--file',
        type=str,
        help='Path to single raw file'
    )
    parser.add_argument(
        '--dir',
        type=str,
        help='Directory containing raw files'
    )
    parser.add_argument(
        '--png',
        type=str,
        help='Output PNG filename (if not specified, saves in same folder as input with _normalized.png suffix)'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display matplotlib visualization'
    )
    parser.add_argument(
        '--no-png',
        action='store_true',
        help='Skip PNG conversion'
    )
    parser.add_argument(
        '--shape',
        type=int,
        nargs=2,
        default=[512, 512],
        help='Image shape (height width)'
    )
    
    args = parser.parse_args()
    
    if args.file:
        filepath = Path(args.file)
        if not filepath.exists():
            print(f"Error: File not found: {args.file}")
            return
        
        print(f"\n{'='*60}")
        print(f"Inspecting: {args.file}")
        print(f"{'='*60}\n")
        
        # Inspect
        inspect_raw_file(str(filepath), shape=tuple(args.shape))
        
        # Convert to PNG (default: in same folder with _normalized.png suffix)
        if not args.no_png:
            if args.png:
                # Use explicit output path
                output_path = Path(args.png)
            else:
                # Auto-generate output path in same folder
                output_path = filepath.parent / f"{filepath.stem}_normalized.png"
            
            output_path.parent.mkdir(parents=True, exist_ok=True)
            raw_to_png(str(filepath), str(output_path), shape=tuple(args.shape))
            print(f"PNG saved to: {output_path}\n")
        
        # Show visualization if requested
        if args.show:
            visualize_raw(str(filepath), shape=tuple(args.shape))
    
    elif args.dir:
        dirpath = Path(args.dir)
        if not dirpath.exists():
            print(f"Error: Directory not found: {args.dir}")
            return
        
        raw_files = list(dirpath.glob('*.raw'))
        
        if not raw_files:
            print(f"No .raw files found in {args.dir}")
            return
        
        print(f"\n{'='*60}")
        print(f"Found {len(raw_files)} raw files in {args.dir}")
        print(f"{'='*60}\n")
        
        for filepath in sorted(raw_files):
            inspect_raw_file(str(filepath), shape=tuple(args.shape))
            
            # Convert to PNG (unless --no-png is specified)
            if not args.no_png:
                output_path = filepath.parent / f"{filepath.stem}_normalized.png"
                raw_to_png(str(filepath), str(output_path), shape=tuple(args.shape))
                print(f"  PNG saved to: {output_path.name}")
            
            print()
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

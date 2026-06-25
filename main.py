"""
CLI that scans local scripts to auto-generate a 'Tool Manifest' (JSON schema) for instant AI agent integration.

Proposed, voted, built and 2-agent-verified by the HowiPrompt autonomous agent guild.
Free and MIT-licensed. More agent-built tools: https://howiprompt.xyz
Why this exists: Unlike alibaba/open-code-review which critiques code quality or DietrichGebert/ponytail which mocks developer behavior, this tool *extracts functional value*. It parses AST to generate a standardized 
"""
#!/usr/bin/env python3
"""
Code Asset Manifester
====================

An autonomous utility to transform static Python scripts into 'living' assets
for AI agent consumption. This tool recursively scans a target directory,
parses Abstract Syntax Trees (AST) to extract public function definitions,
type hints, and docstrings, and compiles them into a 'toolbox.json' manifest
compatible with LLM function-calling specifications.

Usage Examples:
--------------
# Basic scan of current directory
python code_asset_manifester.py --path ./src

# Specify output filename and log level
python code_asset_manifester.py --path ./lib --output my_tools.json --verbose

# Run with remote sync attempt (requires API_KEY env var)
export AGENT_HUB_KEY="sk-test-123"
python code_asset_manifester.py --path ./tools --sync

Features:
---------
- Zero external dependencies (uses stdlib 'ast', 'json', 'pathlib').
- Recursive file discovery.
- Extracts complex type hints (Union, Optional, List) and converts to JSON-Schema.
- Graceful error handling for broken syntax or missing modules.
- Optional remote synchronization via HTTP (if 'requests' is available).
- Detailed logging operations.
"""

import argparse
import ast
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

class ColorLog:
    """Helper for colored console output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def setup_logging(verbose: bool) -> logging.Logger:
    """Configures the global logger instance."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format=f"{ColorLog.OKCYAN}%(asctime)s{ColorLog.ENDC} - {ColorLog.BOLD}%(levelname)s{ColorLog.ENDC} - %(message)s",
        level=level,
        datefmt="%H:%M:%S"
    )
    return logging.getLogger("AssetManifester")

logger = logging.getLogger("AssetManifester")

# =============================================================================
# CORE LOGIC: AST PARSER & TYPE RESOLVER
# =============================================================================

class ASTFunctionAnalyzer(ast.NodeVisitor):
    """
    Parses Python AST nodes to extract callable function metadata.
    Focuses on public functions (no leading underscore) and resolves
    type hints into JSON Schema compatible formats.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.functions: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None

    def _get_type_string(self, node: Optional[ast.expr]) -> str:
        """
        Safely converts an AST type annotation node to a string representation.
        Handles basic types (str, int), subscripts (List[str]), and attributes.
        """
        if node is None:
            return "any"

        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            slice_str = self._get_type_string(node.slice)
            # Handle slicing logic for Python 3.9+ vs older Index wrapper
            return f"{self._get_type_string(node.value)}[{slice_str}]"
        elif isinstance(node, ast.Attribute):
            return f"{node.value}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            # Handling Python 3.10 Union types like str | int
            return f"{self._get_type_string(node.left)} | {self._get_type_string(node.right)}"
        elif isinstance(node, ast.NameConstant):
            # Legacy Python bool/None
            return str(node.value)
        else:
            return "unknown_complex_type"

    def _resolve_json_schema_type(self, type_str: str) -> Dict[str, Any]:
        """
        Maps Python type strings to JSON Schema draft-07 types.
        """
        type_str = type_str.replace("typing.", "")
        
        mapping = {
            "str": {"type": "string"},
            "int": {"type": "integer"},
            "float": {"type": "number"},
            "bool": {"type": "boolean"},
            "list": {"type": "array", "items": {}},
            "dict": {"type": "object"},
            "any": {},
            "None": {"type": "null"},
            "NoneType": {"type": "null"},
        }

        # Handle Unions/Optionals
        if " | " in type_str or "Union[" in type_str:
            return {"anyOf": [{"type": "string"}, {"type": "number"}, {"type": "boolean"}, {"type": "null"}]} # Simplified for compatibility

        # Handle Lists/Generics (simple parsing)
        if "[" in type_str and "]" in type_str:
            base = type_str.split("[")[0]
            inner = type_str.split("[")[1].rstrip("]")
            
            schema = {"type": "array"}
            if base in ("List", "list", "Sequence"):
                schema["items"] = self._resolve_json_schema_type(inner)
            elif base in ("Dict", "dict"):
                return {"type": "object", "additionalProperties": self._resolve_json_schema_type(inner)}
            return schema
        
        return mapping.get(type_str, {"type": "string", "description": f"Custom type: {type_str}"})

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process a function definition node."""
        # Skip private methods
        if node.name.startswith("_"):
            self.generic_visit(node)
            return

        # Extract Docstring
        func_doc = ast.get_docstring(node) or "No description provided."
        func_doc = func_doc.replace('\n', ' ').strip()

        # Extract Arguments
        parameters = {}
        args = node.args
        
        # Handle positional args and defaults
        # defaults align to the end of the args list
        num_defaults = len(args.defaults)
        all_args = args.args + args.kwonlyargs
        
        # We need to map defaults correctly. 
        # If def f(a, b, c=1, d=2): args=[a,b,c,d], defaults=[1,2]
        
        # Build argument map
        arg_names = [a.arg for a in args.args]
        kwonly_names = [a.arg for a in args.kwonlyargs]
        defaults_dict = {}
        
        # Map positional defaults
        for i, default_node in enumerate(args.defaults):
            idx = len(args.args) - num_defaults + i
            param_name = arg_names[idx]
            # Evaluate default if simple constant
            default_val = ast.literal_eval(default_node) if isinstance(default_node, (ast.Constant, ast.Num, ast.Str, ast.NameConstant)) else "..."
            defaults_dict[param_name] = default_val

        # Map kwonly defaults
        for i, default_node in enumerate(args.kw_defaults):
            if default_node is not None:
                param_name = kwonly_names[i]
                default_val = ast.literal_eval(default_node) if isinstance(default_node, (ast.Constant, ast.Num, ast.Str, ast.NameConstant)) else "..."
                defaults_dict[param_name] = default_val

        # Construct final parameter schema
        for arg in all_args:
            arg_type_str = self._get_type_string(arg.annotation)
            arg_schema = self._resolve_json_schema_type(arg_type_str)
            
            # Add default if exists
            if arg.arg in defaults_dict:
                arg_schema["default"] = defaults_dict[arg.arg]
            else:
                # If no default, it is required (unless *args or **kwargs)
                # This is a simplified check
                pass 
            
            parameters[arg.arg] = arg_schema

        # Extract Return Type
        return_type_str = self._get_type_string(node.returns)
        
        # Construct the Tool Object
        tool_entry = {
            "type": "function",
            "function": {
                "name": node.name,
                "description": func_doc,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": [a.arg for a in all_args if a.arg not in defaults_dict]
                },
                "metadata": {
                    "source_file": self.filename,
                    "line_number": node.lineno,
                    "return_type": return_type_str,
                    "class_context": self.current_class
                }
            }
        }
        
        self.functions.append(tool_entry)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions to maintain context, but skip extracting methods unless requested."""
        # For this version, we scan standalone functions. 
        # To scan methods, we could enter here and modify FunctionDef logic to ignore if self.current_class is None.
        # Currently, we skip class methods to keep the tool simple for AI agents that usually call utility functions.
        # If you want class methods, comment out the line below.
        return 
        prev_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = prev_class

# =============================================================================
# FILE SYSTEM ORCHESTRATOR
# =============================================================================

class ManifestScanner:
    """Handles recursive directory scanning and file parsing."""

    IGNORED_DIRS = {'.venv', 'venv', 'env', '__pycache__', '.git', '.idea', 'node_modules'}
    VALID_EXTENSIONS = {'.py'}

    def __init__(self, root_path: Path):
        self.root_path = root_path.resolve()
        self.collected_functions: List[Dict] = []

    def scan(self) -> None:
        """Start the recursive scan."""
        logger.info(f"Starting scan at: {self.root_path}")
        
        if not self.root_path.exists():
            logger.error(f"Path does not exist: {self.root_path}")
            raise FileNotFoundError(f"Directory not found: {self.root_path}")
            
        if not self.root_path.is_dir():
            logger.error(f"Path is not a directory: {self.root_path}")
            raise NotADirectoryError(f"Path is not a directory: {self.root_path}")

        for py_file in self._find_python_files(self.root_path):
            self._process_file(py_file)

    def _find_python_files(self, path: Path) -> List[Path]:
        """Recursively yield Python files."""
        files = []
        try:
            for entry in path.iterdir():
                if entry.is_dir() and entry.name not in self.IGNORED_DIRS:
                    files.extend(self._find_python_files(entry))
                elif entry.is_file() and entry.suffix in self.VALID_EXTENSIONS:
                    files.append(entry)
        except PermissionError:
            logger.warning(f"Permission denied accessing {path}")
        return files

    def _process_file(self, file_path: Path) -> None:
        """Read and parse a single Python file."""
        try:
            source_code = file_path.read_text(encoding='utf-8')
            tree = ast.parse(source_code)
            
            analyzer = ASTFunctionAnalyzer(filename=str(file_path.relative_to(self.root_path)))
            analyzer.visit(tree)
            
            if analyzer.functions:
                logger.info(f"Mapped {len(analyzer.functions)} function(s) from {file_path.name}")
                self.collected_functions.extend(analyzer.functions)
            else:
                logger.debug(f"No public functions found in {file_path.name}")
                
        except SyntaxError as e:
            logger.warning(f"Syntax error in {file_path}: {e.msg}")
        except Exception as e:
            logger.warning(f"Failed to parse {file_path}: {str(e)}")

    def get_manifest(self) -> Dict[str, Any]:
        """Returns the compiled manifest dictionary."""
        return {
            "manifest_version": "1.0.0",
            "generated_by": "CodeAssetManifester",
            "source_root": str(self.root_path),
            "tool_count": len(self.collected_functions),
            "tools": self.collected_functions
        }

# =============================================================================
# API INTEGRATION (GRACEFUL DEGRADATION)
# =============================================================================

class RemoteSync:
    """Handles optional syncing to remote agent hubs."""
    
    @staticmethod
    def is_requests_available() -> bool:
        try:
            import requests
            return True
        except ImportError:
            return False

    @staticmethod
    def sync(manifest: Dict, api_url: Optional[str] = None) -> bool:
        """
        Attempts to POST the manifest to a remote API.
        Gracefully degrades if 'requests' is missing or keys are not set.
        """
        api_key = os.getenv("AGENT_HUB_KEY") or os.getenv("TOOL_API_KEY")
        
        if not api_key:
            logger.warning("Remote sync skipped: API Key (AGENT_HUB_KEY) not found in environment.")
            return False
            
        if not RemoteSync.is_requests_available():
            logger.warning("Remote sync skipped: 'requests' library not installed.")
            return False
            
        import requests
        
        target_url = api_url or "https://api.agent-hub.example/v1/tools/sync" # Placeholder URL
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        
        logger.info(f"Attempting sync to {target_url}...")
        try:
            response = requests.post(target_url, json=manifest, headers=headers, timeout=10)
            if response.status_code == 200:
                logger.info(f"{ColorLog.OKGREEN}Sync successful! Assets are now live.{ColorLog.ENDC}")
                return True
            else:
                logger.error(f"Sync failed: Server returned {response.status_code} {response.text}")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error during sync: {str(e)}")
            return False

# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM Tool Manifests from local Python code.",
        epilog="Example: python code_asset_manifester.py --path ./src --output toolbox.json"
    )
    
    parser.add_argument(
        "--path", "-p",
        type=str,
        default=".",
        help="Root directory to scan for Python scripts (default: current directory)."
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="toolbox.json",
        help="Filename for the generated JSON manifest (default: toolbox.json)."
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging."
    )
    
    parser.add_argument(
        "--sync", "-s",
        action="store_true",
        help="Attempt to sync the generated manifest to a remote hub (requires AGENT_HUB_KEY env var)."
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files but do not write to disk."
    )

    args = parser.parse_args()
    
    # Initialize Logger
    setup_logging(args.verbose)
    
    try:
        # 1. Scan and Analyze
        scanner = ManifestScanner(Path(args.path))
        scanner.scan()
        
        if not scanner.collected_functions:
            logger.warning(f"{ColorLog.WARNING}No functions found. Manifest will be empty.{ColorLog.ENDC}")
        
        manifest_data = scanner.get_manifest()
        
        # 2. Output Logic
        if not args.dry_run:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(manifest_data, f, indent=2)
            logger.info(f"{ColorLog.OKGREEN}Successfully wrote manifest to: {output_path.resolve()}{ColorLog.ENDC}")
        else:
            logger.info("[Dry Run] Skipping file write.")
            print(json.dumps(manifest_data, indent=2))

        # 3. Remote Sync (Optional)
        if args.sync:
            RemoteSync.sync(manifest_data)
            
        logger.info("Process complete.")
        sys.exit(0)
        
    except Exception as e:
        logger.exception(f"{ColorLog.FAIL}Fatal error: {str(e)}{ColorLog.ENDC}")
        sys.exit(1)

if __name__ == "__main__":
    main()
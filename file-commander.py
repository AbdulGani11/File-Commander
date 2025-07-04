#!/usr/bin/env python3
"""
File Commander -- Modern Natural Language File Management

A command-line application that uses natural language to perform file operations.
Built with Typer and Rich for a beautiful and intuitive interface.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

import typer
import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
from rich.table import Table
from rich.text import Text

# Initialize components
app = typer.Typer(
    help="File Commander - Natural language file management",
    add_completion=False,
    no_args_is_help=False,
)
console = Console()
load_dotenv()


class Config:
    """Centralized configuration for File Commander."""

    # API Configuration
    DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"
    API_TIMEOUT_SECONDS = 30

    # Performance Limits
    MAX_SEARCH_RESULTS = 20
    MAX_SEARCH_DEPTH = 5
    CACHE_SIZE_LIMIT = 100

    # File Extensions
    VIDEO_EXTENSIONS = {
        ".mp4",
        ".mkv",
        ".avi",
        ".mov",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        ".3gp",
        ".3g2",
        ".m2ts",
    }

    # System Directories to Skip
    SKIP_DIRECTORIES = {
        "__pycache__",
        "node_modules",
        ".git",
        "venv",
        ".venv",
        ".idea",
        ".vscode",
        "target",
        "build",
        "dist",
    }

    # Windows Compatibility
    WINDOWS_RESERVED_NAMES = {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    }
    WINDOWS_DANGEROUS_CHARS = '<>:"|?*'


# Common directory paths
HOME_DIR = Path.home()
COMMON_LOCATIONS = {
    "home": HOME_DIR,
    "desktop": HOME_DIR / "Desktop",
    "downloads": HOME_DIR / "Downloads",
    "documents": HOME_DIR / "Documents",
    "pictures": HOME_DIR / "Pictures",
    "music": HOME_DIR / "Music",
    "videos": HOME_DIR / "Videos",
    "movies": Path("D:/Movies"),
    # Aliases
    "docs": HOME_DIR / "Documents",
    "pics": HOME_DIR / "Pictures",
    "photos": HOME_DIR / "Pictures",
}

# Add Windows drive letters dynamically
for drive_letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
    drive_path = Path(f"{drive_letter}:/")
    if drive_path.exists():
        COMMON_LOCATIONS.update(
            {
                f"drive_{drive_letter.lower()}": drive_path,
                f"{drive_letter.lower()}_drive": drive_path,
                f"{drive_letter.lower()}": drive_path,
            }
        )


class OperationType(str, Enum):
    """Supported operation types."""

    CREATE_FOLDER = "create_folder"
    CREATE_FILE = "create_file"
    RENAME_ITEM = "rename_item"
    MOVE_ITEM = "move_item"
    MOVE_ALL_FILES = "move_all_files"
    MOVE_MATCHING_FILES = "move_matching_files"
    SEARCH_AND_MOVE = "search_and_move"
    OPEN_FILE_EXPLORER = "open_file_explorer"
    SEARCH_FILES = "search_files"
    PLAY_MOVIE = "play_movie"
    UNKNOWN = "unknown"


class FileOperations:
    """Core file operation functionality."""

    def __init__(self):
        self.current_path = COMMON_LOCATIONS["desktop"]
        self._path_cache: Dict[str, Path] = {}

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path string to an absolute Path object with caching."""
        if not path:
            return self.current_path

        path = path.strip()

        # Check cache first
        if path in self._path_cache:
            return self._path_cache[path]

        resolved = self._resolve_path_strategies(path)

        # Cache the result
        if len(self._path_cache) < Config.CACHE_SIZE_LIMIT:
            self._path_cache[path] = resolved

        return resolved

    def _resolve_path_strategies(self, path: str) -> Path:
        """Try different path resolution strategies."""
        # Strategy 1: Absolute path
        path_obj = Path(path)
        if path_obj.is_absolute():
            return path_obj.resolve()

        # Strategy 2: Common locations
        if path.lower() in COMMON_LOCATIONS:
            return COMMON_LOCATIONS[path.lower()]

        # Strategy 3: Windows drive patterns
        drive_match = re.match(r"^(?:drive\s+)?([a-zA-Z])[:\s]?$", path, re.IGNORECASE)
        if drive_match:
            drive_letter = drive_match.group(1).upper()
            return Path(f"{drive_letter}:/")

        # Strategy 4: Smart folder search across drives
        if "/" not in path and "\\" not in path:
            for drive_letter in "DEFGHIJKLMNOPQRSTUVWXYZ":
                drive_path = Path(f"{drive_letter}:/")
                if drive_path.exists():
                    potential_path = drive_path / path
                    if potential_path.exists() and potential_path.is_dir():
                        return potential_path

        # Strategy 5: Relative to current path
        return (self.current_path / path).resolve()

    def _sanitize_filename(self, name: str) -> str:
        """Sanitize file/folder names for Windows compatibility."""
        name = name.replace("\0", "")
        name = name.replace("/", "_").replace("\\", "_")

        for char in Config.WINDOWS_DANGEROUS_CHARS:
            name = name.replace(char, "_")

        name_upper = name.upper()
        if (
            name_upper in Config.WINDOWS_RESERVED_NAMES
            or name_upper.split(".")[0] in Config.WINDOWS_RESERVED_NAMES
        ):
            name = f"_{name}"

        return name.strip()

    def create_folder(self, folder_name: str, location: str = "") -> str:
        """Create a new folder."""
        if not folder_name or not folder_name.strip():
            return "❌ Error: Folder name cannot be empty"

        folder_name = self._sanitize_filename(folder_name.strip())

        try:
            base_path = self._resolve_path(location)
            folder_path = base_path / folder_name

            if folder_path.exists():
                return f"📁 Folder already exists: {folder_path}"

            folder_path.mkdir(parents=True)
            return f"✅ Created folder: {folder_path}"

        except PermissionError:
            return f"❌ Permission denied: Cannot create folder in {base_path}"
        except OSError as e:
            return f"❌ System error creating folder: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def create_file(self, file_name: str, location: str = "", content: str = "") -> str:
        """Create a new file."""
        if not file_name or not file_name.strip():
            return "❌ Error: File name cannot be empty"

        file_name = self._sanitize_filename(file_name.strip())

        try:
            base_path = self._resolve_path(location)
            file_path = base_path / file_name

            if file_path.exists():
                return f"📄 File already exists: {file_path}"

            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content, encoding="utf-8")
            return f"✅ Created file: {file_path}"

        except PermissionError:
            return f"❌ Permission denied: Cannot create file in {base_path}"
        except OSError as e:
            return f"❌ System error creating file: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def rename_item(self, old_name: str, new_name: str, location: str = "") -> str:
        """Rename a file or folder."""
        if not old_name or not new_name:
            return "❌ Error: Both old and new names must be provided"

        try:
            base_path = self._resolve_path(location)
            old_path = base_path / old_name
            new_path = base_path / self._sanitize_filename(new_name)

            if not old_path.exists():
                return f"❌ Source not found: {old_path}"

            if new_path.exists():
                return f"❌ Destination already exists: {new_path}"

            old_path.rename(new_path)
            return f"✅ Renamed '{old_name}' to '{new_path.name}'"

        except PermissionError:
            return f"❌ Permission denied: Cannot rename in {base_path}"
        except OSError as e:
            return f"❌ System error renaming item: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def move_item(self, source: str, destination: str) -> str:
        """Move a file or folder to a new location."""
        if not source or not destination:
            return "❌ Error: Both source and destination must be provided"

        try:
            source_path = self._resolve_path(source)
            dest_path = self._resolve_path(destination)

            if not source_path.exists():
                return f"❌ Source not found: {source_path}"

            # If destination is a directory, move inside it
            if dest_path.is_dir():
                dest_path = dest_path / source_path.name

            if dest_path.exists():
                return f"❌ Destination already exists: {dest_path}"

            shutil.move(str(source_path), str(dest_path))
            return f"✅ Moved '{source_path.name}' to {dest_path.parent}"

        except PermissionError:
            return "❌ Permission denied: Cannot move to destination"
        except OSError as e:
            return f"❌ System error moving item: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def move_all_files(self, source_dir: str, destination_dir: str) -> str:
        """Move all files from source to destination directory."""
        try:
            source_path = self._resolve_path(source_dir)
            dest_path = self._resolve_path(destination_dir)

            # Validation
            if not source_path.exists() or not source_path.is_dir():
                return f"❌ Source directory not found: {source_path}"

            if not dest_path.exists() or not dest_path.is_dir():
                return f"❌ Destination directory not found: {dest_path}"

            if source_path == dest_path:
                return "❌ Source and destination cannot be the same"

            files_to_move = [f for f in source_path.iterdir() if f.is_file()]

            if not files_to_move:
                return f"📂 No files found in {source_path}"

            return self._execute_bulk_operation(files_to_move, dest_path, source_path)

        except PermissionError:
            return "❌ Permission denied: Cannot access directories"
        except OSError as e:
            return f"❌ System error: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def _execute_bulk_operation(
        self,
        files: List[Path],
        dest_path: Path,
        source_path: Path,
        search_term: str = "",
    ) -> str:
        """Execute bulk file operations with progress tracking."""
        moved_count = skipped_count = failed_count = 0

        if search_term:
            console.print(
                f"[cyan]Moving {len(files)} files containing '{search_term}'...[/cyan]"
            )

        for file_path in track(files, description="Moving files"):
            dest_file = dest_path / file_path.name

            try:
                if dest_file.exists():
                    skipped_count += 1
                    if search_term:  # Only show details for selective moves
                        console.print(
                            f"[yellow]Skipped existing: {file_path.name}[/yellow]"
                        )
                    continue

                shutil.move(str(file_path), str(dest_file))
                moved_count += 1
                if search_term:  # Only show details for selective moves
                    console.print(f"[green]Moved: {file_path.name}[/green]")

            except Exception as e:
                failed_count += 1
                console.print(f"[red]Failed to move {file_path.name}: {e}[/red]")

        # Format result message
        source_description = (
            source_path if not search_term else f"files containing '{search_term}'"
        )
        result = (
            f"✅ Moved {moved_count} files from {source_description} to {dest_path}"
        )

        if skipped_count > 0:
            result += f"\n⚠️  Skipped {skipped_count} existing files"
        if failed_count > 0:
            result += f"\n❌ Failed to move {failed_count} files"

        return result

    def open_file_explorer(self, location: str = "") -> str:
        """Open Windows Explorer at the specified location."""
        try:
            # Handle desktop/folder patterns
            if "/" in location and "desktop" in location.lower():
                parts = location.split("/")
                if parts[0].lower() == "desktop":
                    base_desktop = self._resolve_path("desktop")
                    path_to_open = base_desktop / "/".join(parts[1:])
                else:
                    path_to_open = self._resolve_path(location)
            else:
                path_to_open = self._resolve_path(location)

            if not path_to_open.exists():
                return f"❌ Location not found: {path_to_open}"

            subprocess.Popen(f'explorer "{path_to_open}"', shell=True)
            return f"📂 Opened Explorer: {path_to_open}"

        except FileNotFoundError:
            return "❌ Windows Explorer not found"
        except OSError as e:
            return f"❌ System error opening Explorer: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def search_files(self, search_term: str, search_path: str = "") -> str:
        """Search for files with improved performance."""
        if not search_term:
            return "❌ Error: Search term cannot be empty"

        try:
            base_path = self._resolve_path(search_path)

            if not base_path.exists():
                return f"❌ Search location not found: {base_path}"

            found_files = self._perform_file_search(search_term, base_path)

            if not found_files:
                return f"🔍 No files found containing '{search_term}' in {base_path}"

            self._display_search_results(found_files, search_term)
            return f"✅ Found {len(found_files)} files containing '{search_term}'"

        except PermissionError:
            return f"❌ Permission denied: Cannot search in {base_path}"
        except OSError as e:
            return f"❌ System error during search: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def _perform_file_search(self, search_term: str, base_path: Path) -> List[Path]:
        """Perform the actual file search with depth and performance limits."""
        found_files = []
        search_term_lower = search_term.lower()

        for root, dirs, files in os.walk(base_path):
            # Skip hidden and system directories
            dirs[:] = [
                d
                for d in dirs
                if not d.startswith(".") and d not in Config.SKIP_DIRECTORIES
            ]

            # Limit search depth for performance
            depth = len(Path(root).relative_to(base_path).parts)
            if depth > Config.MAX_SEARCH_DEPTH:
                continue

            # Search files
            for file in files:
                if search_term_lower in file.lower():
                    found_files.append(Path(root) / file)

                    if len(found_files) >= Config.MAX_SEARCH_RESULTS:
                        return found_files

        return found_files

    def _display_search_results(self, files: List[Path], search_term: str) -> None:
        """Display search results in a formatted table."""
        table = Table(title=f"Search Results for '{search_term}'", show_header=True)
        table.add_column("#", style="dim", width=3)
        table.add_column("File Name", style="green")
        table.add_column("Path", style="blue")

        display_count = min(10, len(files))
        for i, file_path in enumerate(files[:display_count], 1):
            table.add_row(str(i), file_path.name, str(file_path.parent))

        console.print(table)

        if len(files) > display_count:
            console.print(
                f"\n[dim]Showing {display_count} of {len(files)} results[/dim]"
            )

    def play_movie(self, movie_name: str) -> str:
        """Search for and play a movie using Windows default player."""
        if not movie_name:
            return "❌ Error: Movie name cannot be empty"

        try:
            movies_dir = COMMON_LOCATIONS["movies"]
            if not movies_dir.exists():
                return f"❌ Movies directory not found: {movies_dir}"

            best_match = self._find_best_movie_match(movie_name, movies_dir)

            if not best_match:
                return f"🎬 No movie found matching '{movie_name}'"

            os.startfile(str(best_match))
            return f"▶️  Playing: {best_match.name}"

        except FileNotFoundError:
            return "❌ No default media player found"
        except OSError as e:
            return f"❌ System error playing movie: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def _find_best_movie_match(
        self, movie_name: str, movies_dir: Path
    ) -> Optional[Path]:
        """Find the best matching movie file."""
        movie_name_lower = movie_name.lower()
        scored_movies = []

        for file_path in movies_dir.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in Config.VIDEO_EXTENSIONS
            ):
                score = self._calculate_match_score(
                    movie_name_lower, file_path.stem.lower()
                )
                if score > 0:
                    scored_movies.append((file_path, score))

        if not scored_movies:
            return None

        # Return the best match
        scored_movies.sort(key=lambda x: x[1], reverse=True)
        return scored_movies[0][0]

    def _calculate_match_score(self, search_term: str, filename: str) -> int:
        """Calculate match score for movie search."""
        score = 0

        # Exact substring match gets high score
        if search_term in filename:
            score += 50

        # Individual word matches
        search_words = search_term.split()
        for word in search_words:
            if word in filename:
                score += 10

        return score

    def move_matching_files(
        self, search_term: str, source_dir: str, destination_dir: str
    ) -> str:
        """Move files that match a search term from source to destination."""
        if not search_term:
            return "❌ Error: Search term cannot be empty"

        try:
            source_path = self._resolve_path(source_dir)

            # Handle desktop/folder patterns
            if "/" in destination_dir and "desktop" in destination_dir.lower():
                parts = destination_dir.split("/")
                if parts[0].lower() == "desktop":
                    base_desktop = self._resolve_path("desktop")
                    dest_path = base_desktop / "/".join(parts[1:])
                else:
                    dest_path = self._resolve_path(destination_dir)
            else:
                dest_path = self._resolve_path(destination_dir)

            # Validation
            if not source_path.exists() or not source_path.is_dir():
                return f"❌ Source directory not found: {source_path}"

            # Create destination if it doesn't exist (for newly created folders)
            if not dest_path.exists():
                try:
                    dest_path.mkdir(parents=True, exist_ok=True)
                    console.print(
                        f"[dim]Created destination directory: {dest_path}[/dim]"
                    )
                except Exception as e:
                    return f"❌ Cannot create destination directory {dest_path}: {e}"

            # Find matching files
            matching_files = self._find_matching_files(search_term, source_path)

            if not matching_files:
                return f"🔍 No files found containing '{search_term}' in {source_path}"

            return self._execute_bulk_operation(
                matching_files, dest_path, source_path, search_term
            )

        except PermissionError:
            return "❌ Permission denied: Cannot access directories"
        except OSError as e:
            return f"❌ System error: {e}"
        except Exception as e:
            return f"❌ Unexpected error: {e}"

    def search_and_move(
        self, search_term: str, source_dir: str, destination_dir: str
    ) -> str:
        """Search for files and move them in one operation."""
        return self.move_matching_files(search_term, source_dir, destination_dir)

    def _find_matching_files(self, search_term: str, source_path: Path) -> List[Path]:
        """Find files that match the search term in the given directory."""
        matching_files = []
        search_term_lower = search_term.lower()

        # Search only in the specified directory (not recursive for this operation)
        for file_path in source_path.iterdir():
            if file_path.is_file() and search_term_lower in file_path.name.lower():
                matching_files.append(file_path)

        return matching_files


class CommandProcessor:
    """Process natural language commands using OpenAI client via OpenRouter."""

    def __init__(self, model_name: str = Config.DEFAULT_MODEL):
        """Initialize the command processor."""
        self.api_key = os.getenv("OPENROUTER_API_KEY")

        if not self.api_key:
            console.print("[bold red]Error: OPENROUTER_API_KEY not found.[/]")
            console.print("Please add your OpenRouter API key to the .env file.")
            raise ValueError("API key not configured")

        try:
            self.client = openai.OpenAI(
                api_key=self.api_key, base_url="https://openrouter.ai/api/v1"
            )
            self.model_name = model_name
            self._response_cache: Dict[str, Dict] = {}
        except Exception as e:
            console.print(f"[bold red]Error initializing OpenAI client: {e}[/]")
            raise

    def parse_command(self, command: str) -> Dict:
        """Parse a natural language command with caching."""
        cache_key = command.lower().strip()

        if cache_key in self._response_cache:
            return self._response_cache[cache_key]

        try:
            result = self._call_api(command)

            # Cache successful results
            if len(self._response_cache) < Config.CACHE_SIZE_LIMIT:
                self._response_cache[cache_key] = result

            return result

        except json.JSONDecodeError:
            console.print("[bold yellow]Warning: Could not parse API response.[/]")
            return {"operation": "unknown", "parameters": {}}
        except Exception as e:
            console.print(f"[bold red]Error processing command: {e}[/]")
            return {"operation": "unknown", "parameters": {}}

    def _call_api(self, command: str, debug: bool = False) -> Dict:
        """Make the actual API call."""
        system_prompt = self._build_system_prompt()

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Command: {command}"},
        ]

        try:
            if debug:
                console.print(
                    f"[dim]Calling API with {len(command)} character command...[/dim]"
                )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0,
                max_tokens=800,
                timeout=Config.API_TIMEOUT_SECONDS,
            )

            content = response.choices[0].message.content

            if debug:
                console.print(
                    f"[dim]Raw API Response: {content if content else 'EMPTY RESPONSE!'}[/dim]"
                )

            if not content or content.strip() == "":
                console.print("[bold red]⚠️ API returned empty response![/bold red]")
                return {
                    "operation": "unknown",
                    "parameters": {},
                    "error": "empty_response",
                }

            # Extract JSON from markdown if present
            if "```json" in content:
                json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)
            elif "```" in content:
                json_match = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
                if json_match:
                    content = json_match.group(1)

            return json.loads(content)

        except json.JSONDecodeError as e:
            console.print(f"[bold red]JSON Parse Error: {e}[/bold red]")
            if debug:
                console.print(f"[dim]Content that failed: {content}[/dim]")
            return {"operation": "unknown", "parameters": {}, "error": "json_parse"}
        except Exception as e:
            console.print(f"[bold red]API Error: {e}[/bold red]")
            return {"operation": "unknown", "parameters": {}, "error": str(e)}

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the API."""
        return """You are a file system command interpreter. Parse commands into JSON.

Operations:
- create_folder: folder_name, location
- create_file: file_name, location, content  
- rename_item: old_name, new_name, location
- move_item: source, destination
- move_all_files: source_dir, destination_dir
- move_matching_files: search_term, source_dir, destination_dir
- search_and_move: search_term, source_dir, destination_dir
- open_file_explorer: location
- search_files: search_term, search_path
- play_movie: movie_name

For "find X files and move them" use search_and_move.

Single: {"operation": "type", "parameters": {...}}
Multiple: {"has_multiple_operations": true, "operations": [{"operation": "type", "parameters": {...}}, ...]}

Return only valid JSON."""


# Initialize global instances
file_ops = FileOperations()


def execute_operation(operation: str, parameters: Dict[str, str]) -> str:
    """Execute a single operation based on type and parameters."""
    operation_map = {
        OperationType.CREATE_FOLDER: lambda: file_ops.create_folder(
            parameters.get("folder_name", ""), parameters.get("location", "")
        ),
        OperationType.CREATE_FILE: lambda: file_ops.create_file(
            parameters.get("file_name", ""),
            parameters.get("location", ""),
            parameters.get("content", ""),
        ),
        OperationType.RENAME_ITEM: lambda: file_ops.rename_item(
            parameters.get("old_name", ""),
            parameters.get("new_name", ""),
            parameters.get("location", ""),
        ),
        OperationType.MOVE_ITEM: lambda: file_ops.move_item(
            parameters.get("source", ""), parameters.get("destination", "")
        ),
        OperationType.MOVE_ALL_FILES: lambda: file_ops.move_all_files(
            parameters.get("source_dir", ""), parameters.get("destination_dir", "")
        ),
        OperationType.MOVE_MATCHING_FILES: lambda: file_ops.move_matching_files(
            parameters.get("search_term", ""),
            parameters.get("source_dir", ""),
            parameters.get("destination_dir", ""),
        ),
        OperationType.SEARCH_AND_MOVE: lambda: file_ops.search_and_move(
            parameters.get("search_term", ""),
            parameters.get("source_dir", ""),
            parameters.get("destination_dir", ""),
        ),
        OperationType.OPEN_FILE_EXPLORER: lambda: file_ops.open_file_explorer(
            parameters.get("location", "")
        ),
        OperationType.SEARCH_FILES: lambda: file_ops.search_files(
            parameters.get("search_term", ""), parameters.get("search_path", "")
        ),
        OperationType.PLAY_MOVIE: lambda: file_ops.play_movie(
            parameters.get("movie_name", "")
        ),
    }

    if operation in operation_map:
        return operation_map[operation]()
    else:
        return "❌ Sorry, I couldn't understand that command. Please try again."


@app.command()
def command(
    cmd: str = typer.Argument(..., help="Natural language command"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Show debug information"),
):
    """Process a natural language command."""
    # Display header
    header = Text("File Commander", style="bold blue")
    console.print(Panel(header, subtitle="Natural Language File Management"))

    console.print(f"[bold cyan]Command:[/] {cmd}")

    try:
        processor = CommandProcessor()

        with console.status("[bold green]Processing command...[/]") as status:
            result = processor.parse_command(cmd)

            if debug:
                console.print(
                    f"\n[bold yellow]Debug - Parsed Result:[/] {json.dumps(result, indent=2)}"
                )

            if result.get("has_multiple_operations"):
                operations = result.get("operations", [])

                if not operations:
                    console.print("[bold yellow]No valid operations found.[/]")
                    return

                # Process multiple operations
                for i, op_data in enumerate(operations, 1):
                    operation = op_data.get("operation", "unknown")
                    parameters = op_data.get("parameters", {})

                    console.print(f"\n[bold]Step {i}:[/] {operation}")
                    if debug:
                        console.print(f"[dim]Parameters: {parameters}[/dim]")

                    output = execute_operation(operation, parameters)
                    console.print(f"[green]{output}[/green]")
            else:
                # Single operation
                operation = result.get("operation", "unknown")
                parameters = result.get("parameters", {})

                if debug:
                    console.print(
                        f"[dim]Operation: {operation}, Parameters: {parameters}[/dim]"
                    )

                output = execute_operation(operation, parameters)
                console.print(f"\n[bold green]Result:[/] {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if debug:
            import traceback

            console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command()
def debug_command(
    cmd: str = typer.Argument(..., help="Natural language command to debug")
):
    """Process a command with debug information displayed."""
    # Display header
    header = Text("File Commander - Debug Mode", style="bold yellow")
    console.print(
        Panel(header, subtitle="Natural Language File Management with Debug Info")
    )

    console.print(f"[bold cyan]Command:[/] {cmd}")

    try:
        processor = CommandProcessor()

        with console.status("[bold green]Processing command...[/]") as status:
            result = processor.parse_command(cmd)

            console.print(f"\n[bold yellow]🔍 Debug - Final Parsed Result:[/]")
            console.print(f"[dim]{json.dumps(result, indent=2)}[/dim]")

            # Show if fallback was used
            if result.get("fallback_used"):
                console.print(
                    f"\n[bold magenta]🔄 Fallback parsing was used (API failed)[/]"
                )
            elif result.get("fallback_failed"):
                console.print(f"\n[bold red]❌ Both API and fallback parsing failed[/]")

            if result.get("has_multiple_operations"):
                operations = result.get("operations", [])

                if not operations:
                    console.print("[bold yellow]No valid operations found.[/]")
                    return

                console.print(
                    f"\n[bold magenta]Found {len(operations)} operations to execute:[/]"
                )

                # Process multiple operations
                for i, op_data in enumerate(operations, 1):
                    operation = op_data.get("operation", "unknown")
                    parameters = op_data.get("parameters", {})

                    console.print(f"\n[bold]Step {i}:[/] [cyan]{operation}[/cyan]")
                    console.print(f"[dim]Parameters: {parameters}[/dim]")

                    output = execute_operation(operation, parameters)
                    console.print(f"[green]{output}[/green]")
            else:
                # Single operation
                operation = result.get("operation", "unknown")
                parameters = result.get("parameters", {})

                console.print(f"\n[bold magenta]Single Operation Detected:[/]")
                console.print(f"[cyan]Operation: {operation}[/cyan]")
                console.print(f"[dim]Parameters: {parameters}[/dim]")

                output = execute_operation(operation, parameters)
                console.print(f"\n[bold green]Result:[/] {output}")

    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        import traceback

        console.print(f"[bold red]Full traceback:[/]")
        console.print(f"[dim]{traceback.format_exc()}[/dim]")


@app.command()
def test_complex():
    """Test complex command parsing without executing."""
    console.print("[bold cyan]Testing complex command parsing...[/]")

    test_command = "Find files in the Documents folder with 'budget' in the filename. Create a folder called 'Financial_Reports' on the Desktop. Move those files into 'Financial_Reports'. Open File Explorer in that folder."

    try:
        processor = CommandProcessor()
        console.print(f"[dim]Test command: {test_command[:100]}...[/dim]")

        result = processor.parse_command(test_command)

        if result.get("operation") == "unknown":
            console.print("[bold red]❌ Command parsing failed[/]")
        else:
            console.print("[bold green]✅ Complex command parsing successful[/]")

        console.print(f"[green]Result: {json.dumps(result, indent=2)}[/green]")

    except Exception as e:
        console.print(f"[bold red]❌ Test Failed: {e}[/]")


@app.command()
def test_api():
    """Test the API connection with a simple command."""
    console.print("[bold cyan]Testing API connection...[/]")

    try:
        processor = CommandProcessor()

        # Test with a very simple command
        test_command = "Create folder test"
        console.print(f"[dim]Test command: {test_command}[/dim]")

        result = processor.parse_command(test_command)
        console.print(f"[green]✅ API Response: {json.dumps(result, indent=2)}[/green]")

        if result.get("operation") != "unknown":
            console.print("[bold green]✅ API is working correctly![/]")
        else:
            console.print("[bold red]❌ API returned unknown operation[/]")

    except Exception as e:
        console.print(f"[bold red]❌ API Test Failed: {e}[/]")


if __name__ == "__main__":
    # Handle no command provided
    if len(sys.argv) == 1:
        console.print("[bold red]❌ No command specified.[/]")
        console.print("\n[bold]Available commands:[/]")
        console.print("• [cyan]command[/cyan] - Execute a natural language command")
        console.print("• [cyan]debug-command[/cyan] - Execute with debug information")
        console.print("• [cyan]test-api[/cyan] - Test the API connection")
        console.print("• [cyan]test-complex[/cyan] - Test complex command parsing")
        sys.exit(1)

    # Custom handling for unknown commands
    if len(sys.argv) > 1 and sys.argv[1] not in [
        "command",
        "debug-command",
        "test-api",
        "test-complex",
    ]:
        console.print(f"[bold red]❌ Unknown command: '{sys.argv[1]}'[/]")
        console.print("\n[bold]Available commands:[/]")
        console.print("• [cyan]command[/cyan] - Execute a natural language command")
        console.print("• [cyan]debug-command[/cyan] - Execute with debug information")
        console.print("• [cyan]test-api[/cyan] - Test the API connection")
        console.print("• [cyan]test-complex[/cyan] - Test complex command parsing")
        sys.exit(1)

    try:
        app()
    except typer.Exit:
        pass

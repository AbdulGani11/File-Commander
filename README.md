# Readme

**Python 3.10+** | **License: MIT**

A modern command-line interface that interprets natural language commands to perform file operations. File Commander uses OpenRouter's API via the OpenAI client to understand complex commands and execute file management tasks with minimal user effort.

## ✨ Features

- **Natural Language Command Processing**: Control your file system using everyday language
- **Multi-Step Operations**: Execute complex workflows with a single command
- **Beautiful Terminal UI**: Professional dark-themed interface with rich visual feedback
- **OpenRouter Integration**: Uses DeepSeek V3 via OpenAI-compatible client for fast, accurate command understanding
- **Comprehensive File Operations**: Create, move, rename, search, and more
- **Smart File Matching**: Find and move specific files based on filename patterns
- **Media Playback**: Easily find and play media files with intelligent matching
- **Deep Path Understanding**: Intelligently resolves paths, common locations, and drives
- **Enhanced Error Handling**: Clear, helpful error messages with emojis for better UX
- **Debug Mode**: Detailed debugging tools to understand command processing
- **Desktop-First**: Defaults to Desktop location for intuitive file management

## 📋 Prerequisites

- **Python 3.10+** (supports modern syntax and type hints)
- **pip** (Python package manager)
- **OpenRouter API key** (free tier available)

## 🚀 Installation

1. **Clone this repository:**

```bash
git clone https://github.com/yourusername/file-commander.git
cd file-commander
```

2. **Create a virtual environment (recommended):**

```bash
# On Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install required dependencies:**

```bash
pip install typer rich python-dotenv openai
```

4. **Create a .env file in the project directory:**

```env
OPENROUTER_API_KEY=your-openrouter-api-key-here
```

## 🖥 Usage

File Commander uses a command-line interface for interaction. It provides several commands for different use cases:

```bash
# Process a natural language command
# This is the main command that takes your English instructions,
# converts them to operations using AI, and executes the actual file operations.
python file-commander.py command "Create folder reports on Desktop"
```

## 🔧 Testing & Debugging

### Test Commands

```bash
# Test basic API functionality
# This command only tests if the DeepSeek API is working and can parse commands.
# It sends a simple test command to the AI but does NOT execute any file operations.
python file-commander.py test-api

# Test complex command parsing
# This tests the AI's ability to parse multi-step commands into multiple operations.
# Like test-api, this only tests parsing - no actual file operations are performed.
python file-commander.py test-complex
```

### Debug Mode

```bash
# Debug any command to see detailed processing
# This runs your actual command but shows all the internal processing steps,
# including raw API responses, parsed operations, and execution details.
python file-commander.py debug-command "your complex command here"
```

Debug mode shows:

- Raw API response from DeepSeek V3
- Parsed operations and parameters
- Step-by-step execution details
- Detailed error information if something fails

## 🛠 Troubleshooting

- **Command not understood**: Try rephrasing using simpler language or use debug mode
- **Path errors**: Ensure paths exist and are correctly specified
- **Initial response**: First command may take a moment for API connection
- **API errors**: Check your OpenRouter API key and internet connection
- **JSON parsing errors**: If you see a warning about JSON parsing, retry the command
- **Multiple interpretations**: For ambiguous commands, be more specific

## 🎯 What Makes This Special

### Performance Optimized

- **DeepSeek V3 Integration** - Fast, accurate command parsing with low hallucination rates
- **Smart Caching** - Improved performance with intelligent response caching
- **Standard API Client** - Uses OpenAI-compatible client for reliable API communication

### Modern Architecture

- Clean, readable code following modern Python 3.10+ practices
- Type hints and modern syntax for better maintainability
- Separation of concerns with distinct classes for different responsibilities
- Comprehensive error handling with specific exception types

### User Experience Focus

- Clear error messages that guide users to solutions
- Progress tracking for bulk operations
- Emoji-enhanced feedback for better visual communication
- Comprehensive debugging tools for troubleshooting

## 🧠 Model Selection: DeepSeek V3

Why DeepSeek V3 is optimal for File Commander's command parsing requirements:

1. **Lower Hallucination Rate**: 3.9% vs alternatives - critical for reliable command interpretation
2. **Optimized for Structured Output**: Specifically enhanced for JSON parsing accuracy
3. **Cost-Effective**: Better economics for simple command-to-JSON conversion
4. **Fast Processing**: Ideal for responsive CLI interactions
5. **Right Tool for the Task**: Excels at pattern matching for file operations
6. **Proven Reliability**: Better accuracy for structured tasks vs complex reasoning models

## 🎯 Why Choose File Commander?

### Speed & Efficiency for Power Users

- **Single-command workflows**: "move all PDFs from Downloads to Documents" vs multiple GUI clicks
- **Multi-step operations**: Complex file management in seconds instead of minutes
- **Bulk pattern matching**: "find all budget files and organize them" handles dozens of files instantly

### Natural Language Advantage

- **Express intent naturally**: "find my tax documents from 2023" vs browsing through folders
- **No GUI memorization**: Describe what you want instead of remembering menu locations
- **Intuitive commands**: Works how you think, not how traditional file explorers are designed

### Batch Operations Made Simple

- **Pattern-based processing**: Operations across many files that would be tedious in GUI
- **Smart file matching**: Finds files by content, name patterns, or date ranges
- **Scalable workflows**: Handles small tasks and large reorganization projects equally well

File Commander isn't meant to replace GUI file explorers - it's designed for users who prefer natural language efficiency over traditional point-and-click interfaces.

## 📜 License

This project is open-source and available under the MIT License.

## 🙏 Acknowledgements

- **OpenRouter** for providing access to powerful language models
- **Typer** for the elegant command-line interface
- **Rich** for beautiful terminal output
- **DeepSeek** for their excellent V3 language model

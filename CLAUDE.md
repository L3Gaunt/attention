# CLAUDE.md

## Build and Run Commands
```
poetry install                                 # Install dependencies
poetry run flask --app attention run --port 8004  # Run the web server
```

## Project Structure
- `attention/attention.py` - Core functionality for attention visualization
- `attention/static/` - Web interface files

## Code Style Guidelines
- **Imports**: Standard library first, then third-party, then local
- **Formatting**: Use docstrings for functions ('''function description''')
- **Types**: No explicit type hints used, but consider adding them
- **Naming**: Use snake_case for functions and variables
- **Comments**: Include comments for complex logic (like matrix operations)
- **Error Handling**: Currently minimal - consider adding more robust handling
- **Libraries**: Uses PyTorch, Transformers, Flask
- **Python Version**: ^3.11

## Demo Access
After running the server, access the demo at: http://127.0.0.1:8004/static/index.html
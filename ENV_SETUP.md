# Environment Variables Setup

This application now supports loading API keys and configuration from environment variables instead of prompting for them each time.

## Setup Instructions

1. **Copy the example environment file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit the `.env` file with your actual credentials:**
   ```bash
   nano .env
   # or use your preferred editor
   ```

3. **Update the following variables in `.env`:**
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `NEO4J_PASSWORD`: Your Neo4j database password
   - `NEO4J_HOST`: Neo4j host (default: localhost)
   - `NEO4J_PORT`: Neo4j port (default: 7687)
   - `NEO4J_USERNAME`: Neo4j username (default: neo4j)

## Example `.env` file:
```
OPENAI_API_KEY=sk-your-actual-openai-key-here
NEO4J_PASSWORD=your-actual-neo4j-password
NEO4J_HOST=localhost
NEO4J_PORT=7687
NEO4J_USERNAME=neo4j
```

## Fallback Behavior

If environment variables are not set or still contain placeholder values, the application will fall back to prompting for credentials interactively, just like before.

## Security

- The `.env` file is automatically ignored by git (listed in `.gitignore`)
- Never commit your actual API keys to version control
- Use `.env.example` as a template for other developers

## Running the Application

After setting up your `.env` file, you can run the application as usual:

```bash
conda activate medgraphrag && python3 -m langgraph_integration.main
```

The application will now automatically load your credentials and show:
- ✓ OpenAI API key loaded from environment
- ✓ Neo4j password loaded from environment 
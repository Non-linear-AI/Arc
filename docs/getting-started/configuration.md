# Configuration

Arc requires configuration for API access to LLM providers. This guide will help you set up Arc's configuration.

## Overview

Arc stores configuration in `~/.arc/user-settings.json`. You can set configuration via:
1. The interactive `/config` command (recommended)
2. Environment variables
3. Direct file editing

## Quick Configuration

The easiest way to configure Arc is using the interactive command:

```bash
# Start Arc
arc chat

# Run the config command
/config
```

You'll be prompted to enter:
- **API Key** - Your LLM provider API key
- **Base URL** - API endpoint (optional, defaults vary by provider)
- **Model** - Model name to use (e.g., `gpt-4`, `claude-sonnet-4`, `deepseek-chat`)

### Example Configuration

```
◇ Configuration
  API Key            ********
  Base URL           https://api.deepseek.com/v1
  Model              deepseek-chat
```

## Supported LLM Providers

Arc works with any OpenAI-compatible API. Popular providers include:

### OpenAI

```bash
API Key:    sk-...  (from https://platform.openai.com/api-keys)
Base URL:   https://api.openai.com/v1  (default)
Model:      gpt-4, gpt-4-turbo, gpt-3.5-turbo
```

### Anthropic Claude

```bash
API Key:    sk-ant-...  (from https://console.anthropic.com/)
Base URL:   https://api.anthropic.com/v1
Model:      claude-sonnet-4, claude-opus-3
```

**Note**: Requires OpenAI-compatible wrapper or use Anthropic's native SDK.

### Google Gemini

```bash
API Key:    Your Gemini API key
Base URL:   https://generativelanguage.googleapis.com/v1beta/openai/
Model:      gemini-pro, gemini-ultra
```

### DeepSeek

```bash
API Key:    Your DeepSeek API key
Base URL:   https://api.deepseek.com/v1
Model:      deepseek-chat, deepseek-coder
```

### Local Models (OpenAI-compatible servers)

You can use local models via OpenAI-compatible servers like:
- **Ollama** with OpenAI compatibility
- **LM Studio**
- **LocalAI**
- **vLLM**

```bash
API Key:    (not required for local)
Base URL:   http://localhost:11434/v1  (example for Ollama)
Model:      llama3, mistral, mixtral, etc.
```

## Configuration Methods

### Method 1: Interactive Config (Recommended)

```bash
arc chat
/config
```

Follow the prompts to enter your settings.

### Method 2: Environment Variables

Set environment variables before starting Arc:

```bash
# Set API configuration
export ARC_API_KEY="your-api-key-here"
export ARC_BASE_URL="https://api.openai.com/v1"
export ARC_MODEL="gpt-4"

# Start Arc
arc chat
```

Environment variables take precedence over the settings file.

### Method 3: Direct File Editing

Edit `~/.arc/user-settings.json` directly:

```bash
# Create/edit the file
nano ~/.arc/user-settings.json
```

```json
{
  "apiKey": "your-api-key-here",
  "baseURL": "https://api.openai.com/v1",
  "model": "gpt-4"
}
```

Save and restart Arc for changes to take effect.

## Configuration File Location

Arc stores configuration in:
- **macOS/Linux**: `~/.arc/user-settings.json`
- **Windows**: `%USERPROFILE%\.arc\user-settings.json`

## Viewing Current Configuration

To view your current configuration:

```bash
arc chat
/config
```

This shows your current settings (with API key masked for security).

## Security Best Practices

### Protect Your API Key

- **Never commit** your API key to Git
- **Never share** your `user-settings.json` file
- **Use environment variables** in CI/CD environments
- **Rotate keys regularly** if exposed

### Using `.env` Files (Alternative)

For development, you can use a `.env` file:

```bash
# Create .env file (don't commit!)
echo "ARC_API_KEY=your-key" > .env
echo "ARC_BASE_URL=https://api.openai.com/v1" >> .env
echo "ARC_MODEL=gpt-4" >> .env

# Add to .gitignore
echo ".env" >> .gitignore
```

Then load it before starting Arc:

```bash
# Load environment variables
source .env  # or `set -a; source .env; set +a`

# Start Arc
arc chat
```

## Troubleshooting

### API Key Not Working

If your API key isn't working:

1. **Verify the key is valid**
   - Check your provider's dashboard
   - Regenerate the key if necessary

2. **Check the base URL**
   - Ensure it matches your provider
   - Include `/v1` suffix if required

3. **Test the key directly**
   ```bash
   curl -H "Authorization: Bearer YOUR_KEY" \
        https://api.openai.com/v1/models
   ```

### Configuration Not Loading

If Arc doesn't recognize your configuration:

1. **Check file location**
   ```bash
   ls -la ~/.arc/user-settings.json
   ```

2. **Verify JSON syntax**
   ```bash
   cat ~/.arc/user-settings.json | python -m json.tool
   ```

3. **Check file permissions**
   ```bash
   chmod 600 ~/.arc/user-settings.json
   ```

### Environment Variables Not Working

If environment variables aren't being recognized:

```bash
# Verify they're set
echo $ARC_API_KEY
echo $ARC_BASE_URL
echo $ARC_MODEL

# Export them if needed
export ARC_API_KEY="your-key"
```

## Advanced Configuration

### Custom Timeout Settings

For slower models or networks, you may need to adjust timeouts. This currently requires code modification but may be configurable in future versions.

### Multiple Configurations

To switch between different providers:

```bash
# Save current config
cp ~/.arc/user-settings.json ~/.arc/config-openai.json

# Switch to different provider
cp ~/.arc/config-deepseek.json ~/.arc/user-settings.json

# Restart Arc
arc chat
```

### Team/Organization Setup

For teams:
1. Use environment variables in shared environments
2. Document required configuration in your team's README
3. Never share actual API keys - each team member uses their own

## Next Steps

Once configured, continue to:
- **[Quick Start Tutorial](quickstart.md)** - Build your first model
- **[CLI Commands Reference](../api-reference/cli-commands.md)** - Learn all commands

## Need Help?

- Check [GitHub Issues](https://github.com/non-linear-ai/arc/issues)
- Open a [new issue](https://github.com/non-linear-ai/arc/issues/new) with configuration problems

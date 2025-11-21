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

## Next Steps

Once configured, continue to:
- **[Quick Start Tutorial](quickstart.md)** - Build your first model
- **[CLI Commands Reference](../api-reference/cli-commands.md)** - Learn all commands

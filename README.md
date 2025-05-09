# MathLearner

The MathLearner system helps solve mathematical problems based on learning from previously solved problems.

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd MathLearner
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

## Usage

### Training the Model

```
python main.py train --dataset path/to/dataset.json --limit 10
```

Where:
- `--dataset`: Path to the JSON file containing problem data and solutions
- `--limit`: (Optional) Maximum number of problems to process

### Solving a Problem

```
python main.py solve --problem "Problem to solve goes here"
```

Example:
```
python main.py solve --problem "Find the value of x that satisfies the equation x^2 - 5x + 6 = 0."
```

### Evaluating the Model

```
python main.py evaluate --test-problems path/to/test_problems.json
```

## System Structure

The MathLearner system consists of 2 main modules:

1. **Learning Module**: Learns from problems and solutions, creates standardized solutions, Python code, and problem features.

2. **Application Module**: Uses the knowledge learned to solve new problems by searching for similar problems.

### LLM Integration

The system uses an abstracted LLM interface that supports multiple providers:

- **BaseLLM**: Abstract base class that defines the interface for LLM interactions
- **OpenAILLM**: Implementation for OpenAI models (e.g., GPT-4)
- **AnthropicLLM**: Implementation for Anthropic models (e.g., Claude)
- **OpenRouterLLM**: Implementation for OpenRouter API, which provides access to various models across different providers
- **LLMFactory**: Factory class for creating LLM instances

To use a specific model, you can specify it when running the commands or modify the defaults in the code.

## Data Format

The input JSON file should have the following format:

```json
[
  {
    "problem": "Problem content...",
    "solution": "Detailed solution..."
  },
  ...
]
```

## Metrics

- **Global Accuracy**: The proportion of problems successfully solved out of all problems.
- **Precision Accuracy**: The proportion of problems successfully solved among problems that found similar problems.
- **Profitability**: Measures the benefit of finding similar problems.


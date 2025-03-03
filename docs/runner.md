# `runner` helper script

This script generates and executes command-line commands based on provided arguments. It supports parallel execution using multiple runners and handles incompatible argument groups.

## Features
- Parses command-line arguments and constructs all possible valid command variations.
- Supports filtering out incompatible argument combinations.
- Allows parallel execution of commands using multiple runners.
- Provides an option to display generated commands without execution.

## Usage

```bash
./runner <base_command> [options]
```

### Options
- `--runners <num>`: Specifies the number of parallel runners to execute commands.
- `--runner-info`: Enables runner information display without execution.
- `--runner-filter <arg1,arg2,...>`: Defines groups of incompatible arguments to avoid executing conflicting combinations.
- `--`: Marks the beginning of command-line arguments to be recorded.

## Example
```bash
./runner ./docs/nothing --runners 2 --runner-filter A1,B2 -- --optionA A1 A2 --optionB B1 B2 B3
```
This will generate all possible valid command variations based on provided arguments and execute them using two parallel runners.

## How It Works
1. Parses the base command and options.
2. Constructs all possible argument combinations.
3. Filters out incompatible combinations.
4. Executes the valid commands using multiprocessing.

## Output
- Displays the total number of generated commands.
- Executes commands using the specified number of parallel runners.
- Provides a structured JSON output of the command variations.

import re
import os
import subprocess
import sys
from typing import Any
import json

def extract_python_code(response: str) -> str:
    code_pattern = r"```(?:python)?(.*?)```"
    matches = re.findall(code_pattern, response, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return response.strip()

def run_python_code(code: str, timeout: int = 30) -> tuple[bool, str, str]:
    try:
        with open("temp_solution.py", "w") as f:
            f.write(code)
            
        result = subprocess.run(
            [sys.executable, "temp_solution.py"], 
            capture_output=True, 
            text=True, 
            timeout=timeout
        )
        
        if result.returncode != 0:
            return False, "", result.stderr
        else:
            return True, result.stdout.strip(), ""
    except Exception as e:
        return False, "", str(e)
    finally:
        if os.path.exists("temp_solution.py"):
            try:
                os.remove("temp_solution.py")
            except:
                pass


def check_solutions_match(output: str, expected: str) -> tuple[bool, str]:
    output = str(output).strip()
    expected = str(expected).strip()

    # Direct comparison
    if output == expected:
        return True, "Direct match - Solution is correct"
    
    # Numerical comparison 
    try:
        if float(output) == float(expected):
            return True, "Numerical match - Solution is correct"
    except:
        pass
    
    return False, f"Expected {expected}, but got {output}"

def save_solution(code: str, filename: str = "final_solution.py") -> None:
    with open(filename, "w") as f:
        f.write(code)

def save_training_examples(examples: list[dict[str, Any]], filename: str = "data/training_examples.json") -> None:
    with open(filename, 'w') as f:
        json.dump(examples, f, indent=2)
    
    print(f"Training examples saved to {filename}")

def load_training_examples(filename: str = "data/training_examples.json") -> list[dict[str, Any]]:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []



import gradio as gr
import json
from application_module import ApplicationModule
from embedding import EmbeddingManager

class MathSolverInterface:
    def __init__(self):
        self.app = ApplicationModule()
        self.embedding_manager = EmbeddingManager()
        
        with open("./data/training_examples.json", "r") as f:
            self.training_examples = json.load(f)

    def find_similar_questions(self, question: str, num_results: int = 3) -> list[dict]:
        """Find similar questions from training examples"""
        
        # First extract features from the question
        features = self.embedding_manager.extract_features_from_question(question)
        
        # Get similar questions using both question and features
        similar_indices = self.embedding_manager.search_similar_problems(
            question=question,
            features=features,
            n_results=num_results
        )
        
        # Return the similar examples
        return similar_indices

    def solve_problem(self, question: str) -> tuple[str, str, str]:
        """
        Solve the problem and return the formatted output
        Returns: solution_output, solution_code, similar_questions_text
        """
        result = self.app.process_question(question)
        
        # Clean up the solution output formatting - remove leading whitespace
        solution_output = f"""### Solution Result:
{result['output']}

### Success: {result['success']}

### Verification:
```json
{json.dumps(result['verification'], indent=2)}
```
"""
        
        # Get similar questions
        similar_examples = self.find_similar_questions(question)
        
        # Format similar questions output with proper markdown
        similar_questions_text = "### Similar Questions and Their Solutions:\n\n"
        for i, example in enumerate(similar_examples, 1):
            if isinstance(example, dict):
                # First retrieve the solution code
                solution_code = example.get('solution', example.get('modified_solution', 'Solution code not available'))
                
                # Find the answer by extracting it from the solution's output (last print statement)
                answer = "Answer not available"
                
                # Try to get the answer from the training examples based on the question
                for training_example in self.training_examples:
                    if training_example.get('question') == example.get('question'):
                        answer = training_example.get('answer', 'Answer not available')
                        break
                
                similar_questions_text += f"""#### {i}. Question:
{example.get('question', 'Question not available')}

Solution:
```python
{solution_code}
```

Output: {answer}
---
"""
            else:
                similar_questions_text += f"""#### {i}. Example:
{str(example)}
---
"""
        
        return (
            solution_output,
            result['solution_code'],
            similar_questions_text
        )

def create_interface():
    solver = MathSolverInterface()
    
    def process_input(question: str) -> tuple[str, str, str]:
        try:
            return solver.solve_problem(question)
        except Exception as e:
            return (
                f"Error processing question: {str(e)}",
                "Error generating solution code",
                "Could not retrieve similar questions"
            )
    
    # Create the Gradio interface
    iface = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Textbox(
                label="Enter your math problem",
                placeholder="Type your math problem here...",
                lines=3
            )
        ],
        outputs=[
            gr.Markdown(label="Solution"),
            gr.Code(language="python", label="Solution Code"),
            gr.Markdown(label="Similar Questions")
        ],
        title="Math Problem Solver",
        description="""
        Enter a math problem in natural language and get:
        1. The solution with verification
        2. The Python code used to solve it
        3. Similar problems from the training database with their solutions
        """,
        examples=[
            ["If a triangle has sides of length 3, 4, and 5, what is its area?"],
            ["After selling 10 peaches to her friends for $2 each and 4 peaches to her relatives for $1.25 each, while keeping one peach for herself, how much money did Lilia earn from selling a total of 14 peaches?"],
            ["If x + 2 = 7, what is the value of x?"]
        ],
        theme=gr.themes.Soft()
    )
    
    return iface

if __name__ == "__main__":
    # Create and launch the interface
    iface = create_interface()
    iface.launch(
        share=True,  # Creates a public link
        server_name="0.0.0.0",  # Makes the server publicly accessible
        server_port=7860  # Default Gradio port
    )